"""Automatic processing-environment bootstrap for the OCTolyzer desktop launcher.

When ``gui/environment.py`` finds no compatible Python environment, this
module can set one up from scratch: download a pinned release of `uv`
(https://github.com/astral-sh/uv), use it to install a managed CPython
build, create a dedicated virtual environment, and install OCTolyzer's
pinned requirements into it -- all without requiring the user to already
have Python, conda, or any package manager installed. An internet
connection is required; nothing here bundles the heavy scientific stack.

Framework-agnostic by design (importable and testable without PySide6),
matching ``gui/environment.py``; ``gui/app.py`` wraps ``run_bootstrap`` in a
QThread worker so it doesn't block the UI.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from gui.environment import EnvironmentCandidate, _cache_directory

# Bump periodically; check https://github.com/astral-sh/uv/releases for the
# current version and re-run this module's tests (which mock the network, so
# a bump only needs the version string and asset-name scheme to stay valid).
UV_VERSION = "0.12.6"
UV_RELEASE_BASE = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}"
PYTHON_VERSION = "3.11"
# pip has historically resolved GPU-capable torch/torchvision wheels from
# PyPI directly on Windows/Linux, falling back to CPU-only execution at
# runtime when no CUDA device is present (see gui/environment.py's probe,
# which already reports torch.cuda.is_available()/torch.backends.mps).
# requirements.txt also carries this as its first line; passing it here too
# is a harmless, defensive no-op if uv already honored the embedded one.
TORCH_FIND_LINKS = "https://download.pytorch.org/whl/torch_stable.html"
DOWNLOAD_CHUNK_SIZE = 1 << 16  # 64 KiB
REQUEST_TIMEOUT = 30


class BootstrapStage(str, Enum):
    CHECKING_TOOLS = "checking_tools"
    DOWNLOADING_MANAGER = "downloading_manager"
    INSTALLING_PYTHON = "installing_python"
    CREATING_ENVIRONMENT = "creating_environment"
    INSTALLING_PACKAGES = "installing_packages"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"


class BootstrapError(Exception):
    """Raised when the automatic environment setup cannot proceed."""

    def __init__(self, message: str, *, remediation: str | None = None):
        super().__init__(message)
        self.message = message
        self.remediation = remediation

    def __str__(self) -> str:
        return f"{self.message} {self.remediation}" if self.remediation else self.message


class BootstrapCancelled(Exception):
    """Raised when a cancellation flag is observed mid-step."""


@dataclass(frozen=True)
class _UvAsset:
    archive_name: str


def bootstrap_tools_dir() -> Path:
    """Directory the downloaded uv binary is cached in."""
    return _cache_directory() / "bootstrap" / "uv"


def runtime_env_dir() -> Path:
    """Directory the auto-created processing environment lives in."""
    return _cache_directory() / "runtime-env"


def _normalize_arch(machine: str) -> str:
    machine = machine.lower()
    if machine in ("arm64", "aarch64"):
        return "aarch64"
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    raise BootstrapError(
        f"Unsupported processor architecture: {machine}.",
        remediation="Install Python and dependencies manually; see the README.",
    )


def _platform_asset() -> _UvAsset:
    system = platform.system()
    arch = _normalize_arch(platform.machine())
    if system == "Windows":
        return _UvAsset(f"uv-{arch}-pc-windows-msvc.zip")
    if system == "Darwin":
        return _UvAsset(f"uv-{arch}-apple-darwin.tar.gz")
    if system == "Linux":
        # gnu (glibc) builds only; musl-based distributions (e.g. Alpine)
        # aren't supported by this automatic path and need the manual
        # conda/pip setup documented in the README.
        return _UvAsset(f"uv-{arch}-unknown-linux-gnu.tar.gz")
    raise BootstrapError(
        f"Unsupported operating system: {system}.",
        remediation="Install Python and dependencies manually; see the README.",
    )


def _uv_executable_name() -> str:
    return "uv.exe" if platform.system() == "Windows" else "uv"


def find_local_uv() -> Path | None:
    """Look for an already-downloaded or system-installed uv binary."""
    cached = bootstrap_tools_dir() / _uv_executable_name()
    if cached.is_file():
        return cached
    system_uv = shutil.which("uv")
    return Path(system_uv) if system_uv else None


def download_uv(
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_flag: threading.Event | None = None,
) -> Path:
    """Download, checksum-verify, and extract the pinned uv release.

    Returns the path to the extracted ``uv``/``uv.exe`` executable, cached
    under ``bootstrap_tools_dir()`` for future launches.
    """
    asset = _platform_asset()
    tools_dir = bootstrap_tools_dir()
    tools_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)
        archive_path = staging_path / asset.archive_name
        archive_url = f"{UV_RELEASE_BASE}/{asset.archive_name}"
        checksum_url = f"{archive_url}.sha256"

        expected_checksum = _download_text(checksum_url).split()[0].lower()
        _download_file(archive_path, archive_url, on_progress=on_progress, cancel_flag=cancel_flag)
        actual_checksum = _sha256(archive_path)
        if actual_checksum.lower() != expected_checksum:
            raise BootstrapError(
                "The downloaded uv archive failed checksum verification.",
                remediation="Check your internet connection and try again.",
            )

        extracted_dir = staging_path / "extracted"
        _extract_archive(archive_path, extracted_dir)
        executable_name = _uv_executable_name()
        found = next(extracted_dir.rglob(executable_name), None)
        if found is None:
            raise BootstrapError("The downloaded uv archive did not contain a uv executable.")

        destination = tools_dir / executable_name
        shutil.copy2(found, destination)
        if platform.system() != "Windows":
            mode = destination.stat().st_mode
            destination.chmod(mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return destination


def _download_text(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise BootstrapError(
            f"Could not reach {url}.",
            remediation="Check your internet connection and try again.",
        ) from error


def _download_file(
    destination: Path,
    url: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_flag: threading.Event | None = None,
) -> None:
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
            total = int(response.headers.get("Content-Length", 0))
            read = 0
            with open(destination, "wb") as handle:
                while True:
                    if cancel_flag is not None and cancel_flag.is_set():
                        raise BootstrapCancelled()
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    read += len(chunk)
                    if on_progress is not None:
                        on_progress(read, total)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise BootstrapError(
            f"Download failed: {url}.",
            remediation="Check your internet connection and try again.",
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    with tarfile.open(archive_path) as archive:
        _safe_extract_tar(archive, destination)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    # Defense in depth against path traversal, even though the archive comes
    # from a pinned, checksum-verified GitHub release.
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if destination_resolved not in member_path.parents and member_path != destination_resolved:
            raise BootstrapError("The downloaded uv archive contains unsafe paths.")
    archive.extractall(destination)


def build_python_install_command(uv_executable: str | os.PathLike[str]) -> list[str]:
    return [str(uv_executable), "python", "install", PYTHON_VERSION]


def build_venv_command(uv_executable: str | os.PathLike[str], target_dir: str | os.PathLike[str]) -> list[str]:
    return [str(uv_executable), "venv", str(target_dir), "--python", PYTHON_VERSION]


def venv_python_path(target_dir: str | os.PathLike[str]) -> Path:
    target = Path(target_dir)
    if platform.system() == "Windows":
        return target / "Scripts" / "python.exe"
    return target / "bin" / "python"


def torch_preinstall_command(
    uv_executable: str | os.PathLike[str],
    venv_python: str | os.PathLike[str],
) -> list[str]:
    """Plain `torch`/`torchvision` install -- no CUDA/CPU index selection.

    Default PyPI wheels on Windows/Linux ship with CUDA support and run
    fine CPU-only when no GPU is present; the codebase already does
    CUDA/MPS device selection at runtime, so no branching is needed here.
    """
    return [str(uv_executable), "pip", "install", "--python", str(venv_python), "torch", "torchvision"]


def build_install_command(
    uv_executable: str | os.PathLike[str],
    venv_python: str | os.PathLike[str],
    requirements_path: str | os.PathLike[str],
    *,
    find_links: str | None = TORCH_FIND_LINKS,
) -> list[str]:
    command = [
        str(uv_executable), "pip", "install",
        "--python", str(venv_python),
        "-r", str(requirements_path),
    ]
    if find_links:
        command.extend(["--find-links", find_links])
    return command


def run_streaming(
    command: list[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    cancel_flag: threading.Event | None = None,
) -> int:
    """Run a subprocess, streaming merged stdout/stderr line-by-line."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if on_line is not None:
                on_line(line.rstrip("\n"))
            if cancel_flag is not None and cancel_flag.is_set():
                process.terminate()
                break
        process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if cancel_flag is not None and cancel_flag.is_set():
        raise BootstrapCancelled()
    return process.returncode


def assemble_candidate(venv_python: str | os.PathLike[str]) -> EnvironmentCandidate:
    """Wrap the freshly-created venv as a candidate for the normal discovery/probe flow.

    Label and source ("bootstrap") must match what
    gui.environment._bootstrapped_environment() produces for the same path --
    that's the durable discovery source for this venv (found again on every
    future refresh/restart); this is only the immediate, same-session
    shortcut so the UI doesn't have to wait on a full re-discovery pass right
    after setup finishes.
    """
    return EnvironmentCandidate(Path(venv_python), "OCTolyzer (auto-installed)", "bootstrap")


def run_bootstrap(
    *,
    requirements_path: str | os.PathLike[str],
    on_stage: Callable[[BootstrapStage], None] | None = None,
    on_output: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_flag: threading.Event | None = None,
) -> EnvironmentCandidate:
    """Run the full bootstrap sequence, returning a candidate ready to probe.

    Does not itself call ``probe_environment`` -- the caller (the GUI's
    bootstrap worker) hands the returned candidate back into the existing
    discovery/probe pipeline in ``gui/environment.py`` rather than this
    module inventing a parallel verification path.
    """

    def stage(value: BootstrapStage) -> None:
        if on_stage is not None:
            on_stage(value)

    def emit(line: str) -> None:
        if on_output is not None:
            on_output(line)

    def run_step(command: list[str]) -> None:
        exit_code = run_streaming(command, on_line=emit, cancel_flag=cancel_flag)
        if exit_code != 0:
            raise BootstrapError(
                f"Command failed (exit code {exit_code}): {' '.join(command)}",
                remediation="Review the log above, or set up the environment manually per the README.",
            )

    stage(BootstrapStage.CHECKING_TOOLS)
    uv_executable = find_local_uv()
    if uv_executable is None:
        stage(BootstrapStage.DOWNLOADING_MANAGER)
        emit(f"Downloading uv {UV_VERSION}...")
        uv_executable = download_uv(on_progress=on_progress, cancel_flag=cancel_flag)
        emit(f"uv ready at {uv_executable}")
    else:
        emit(f"Using uv at {uv_executable}")

    stage(BootstrapStage.INSTALLING_PYTHON)
    emit(f"Installing managed Python {PYTHON_VERSION}...")
    run_step(build_python_install_command(uv_executable))

    stage(BootstrapStage.CREATING_ENVIRONMENT)
    target_dir = runtime_env_dir()
    if target_dir.exists():
        shutil.rmtree(target_dir)
    emit(f"Creating environment at {target_dir}...")
    run_step(build_venv_command(uv_executable, target_dir))
    venv_python = venv_python_path(target_dir)

    stage(BootstrapStage.INSTALLING_PACKAGES)
    emit("Installing torch and torchvision...")
    run_step(torch_preinstall_command(uv_executable, venv_python))
    emit("Installing OCTolyzer's remaining dependencies (this can take several minutes)...")
    run_step(build_install_command(uv_executable, venv_python, requirements_path))

    stage(BootstrapStage.VERIFYING)
    candidate = assemble_candidate(venv_python)
    stage(BootstrapStage.DONE)
    return candidate
