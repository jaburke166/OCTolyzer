"""Cross-platform auto-updater for OCTolyzer source and dependencies."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_REPOSITORY = "jaburke166/OCTolyzer"
DEFAULT_BRANCH = "main"
VERSION_FILENAME = ".octolyzer_version"
PRESERVED_NAMES = {
    ".git",
    ".venv",
    "venv",
    ".env",
    "env",
    "__pycache__",
    VERSION_FILENAME,
    "config.txt",
}


@dataclass(frozen=True)
class UpdateResult:
    updated: bool
    local_sha: str | None = None
    remote_sha: str | None = None
    message: str = ""
    error: str | None = None


def check_and_apply_update(
    root: str | os.PathLike[str],
    *,
    repository: str | None = None,
    branch: str | None = None,
    timeout: float = 4.0,
    sync_dependencies: bool = True,
    status_callback: Callable[[str], None] | None = None,
) -> UpdateResult:
    """Check GitHub for a newer commit and update code + dependencies if available."""
    if os.environ.get("OCTOLYZER_NO_AUTO_UPDATE", "").strip().lower() in {"1", "true", "yes"}:
        return UpdateResult(updated=False, message="Auto-update disabled via OCTOLYZER_NO_AUTO_UPDATE.")

    repo_root = Path(root).expanduser().resolve()
    repo_slug = repository or os.environ.get("OCTOLYZER_UPDATE_REPO", DEFAULT_REPOSITORY)
    target_branch = branch or os.environ.get("OCTOLYZER_UPDATE_BRANCH", DEFAULT_BRANCH)

    def _notify(text: str) -> None:
        if status_callback is not None:
            status_callback(text)

    _notify("Checking for OCTolyzer updates...")
    local_sha = get_local_commit_sha(repo_root)
    remote_sha = fetch_latest_commit_sha(repo_slug, target_branch, timeout=timeout)
    if remote_sha is None:
        return UpdateResult(
            updated=False,
            local_sha=local_sha,
            message="Offline or unable to reach GitHub; skipping update check.",
        )

    if local_sha and local_sha == remote_sha:
        return UpdateResult(
            updated=False,
            local_sha=local_sha,
            remote_sha=remote_sha,
            message="OCTolyzer is up to date.",
        )

    _notify(f"Updating OCTolyzer to {remote_sha[:7]}...")
    try:
        if _can_update_with_git(repo_root):
            _update_with_git(repo_root, target_branch)
        else:
            _update_from_archive(repo_root, repo_slug, target_branch, timeout=max(timeout, 30.0))
        _write_version_file(repo_root, remote_sha)
        if sync_dependencies:
            _notify("Updating Python dependencies...")
            sync_environment_dependencies(repo_root)
    except Exception as exc:
        return UpdateResult(
            updated=False,
            local_sha=local_sha,
            remote_sha=remote_sha,
            message="Update failed; continuing with current version.",
            error=str(exc),
        )

    return UpdateResult(
        updated=True,
        local_sha=local_sha,
        remote_sha=remote_sha,
        message=f"Updated OCTolyzer to {remote_sha[:7]}.",
    )


def get_local_commit_sha(root: Path) -> str | None:
    """Return the local commit SHA from git or the .octolyzer_version marker."""
    version_file = root / VERSION_FILENAME
    if (root / ".git").exists() and shutil.which("git"):
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            sha = completed.stdout.strip()
            if completed.returncode == 0 and len(sha) == 40:
                return sha
        except (OSError, subprocess.SubprocessError):
            pass

    if version_file.is_file():
        try:
            sha = version_file.read_text(encoding="utf-8").strip()
            if sha:
                return sha
        except OSError:
            pass
    return None


def fetch_latest_commit_sha(
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    *,
    timeout: float = 4.0,
) -> str | None:
    """Fetch the latest commit SHA for the given GitHub repository and branch."""
    url = f"https://api.github.com/repos/{repository}/commits/{branch}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OCTolyzer-AutoUpdater",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            sha = payload.get("sha")
            return str(sha) if isinstance(sha, str) and sha else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _can_update_with_git(root: Path) -> bool:
    if not (root / ".git").exists() or shutil.which("git") is None:
        return False
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return status.returncode == 0 and not status.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False


def _update_with_git(root: Path, branch: str) -> None:
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", branch],
        cwd=root,
        check=True,
        timeout=30.0,
    )
    subprocess.run(
        ["git", "merge", "--ff-only", f"origin/{branch}"],
        cwd=root,
        check=True,
        timeout=30.0,
    )


def _update_from_archive(root: Path, repository: str, branch: str, *, timeout: float = 30.0) -> None:
    archive_url = f"https://github.com/{repository}/archive/refs/heads/{branch}.tar.gz"
    request = urllib.request.Request(
        archive_url,
        headers={"User-Agent": "OCTolyzer-AutoUpdater"},
    )
    with tempfile.TemporaryDirectory(prefix="octolyzer-update-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_file = temp_path / "repo.tar.gz"
        with urllib.request.urlopen(request, timeout=timeout) as response:
            archive_file.write_bytes(response.read())

        extract_dir = temp_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive_file, "r:gz") as tar:
            _safe_extract_tar(tar, extract_dir)

        top_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
        if not top_dirs:
            raise RuntimeError("Downloaded update archive was empty.")
        source_root = top_dirs[0]
        _copy_updated_tree(source_root, root)


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    dest_resolved = destination.resolve()
    for member in tar.getmembers():
        member_path = (dest_resolved / member.name).resolve()
        if not str(member_path).startswith(str(dest_resolved)):
            raise RuntimeError(f"Unsafe path in update archive: {member.name}")
    tar.extractall(destination)


def _copy_updated_tree(source_root: Path, target_root: Path) -> None:
    for item in source_root.iterdir():
        if item.name in PRESERVED_NAMES and (target_root / item.name).exists():
            continue
        destination = target_root / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _write_version_file(root: Path, sha: str) -> None:
    try:
        (root / VERSION_FILENAME).write_text(f"{sha}\n", encoding="utf-8")
    except OSError:
        pass


def find_uv_executable() -> str | None:
    """Locate uv on PATH or in standard user installation directories."""
    uv_on_path = shutil.which("uv")
    if uv_on_path:
        return uv_on_path

    exe_name = "uv.exe" if os.name == "nt" else "uv"
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / exe_name,
        home / ".cargo" / "bin" / exe_name,
    ]
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidates.append(Path(local_appdata) / "uv" / exe_name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def sync_environment_dependencies(root: Path, *, python_executable: str | None = None) -> None:
    """Sync requirements.txt into the active environment using uv (or pip fallback)."""
    requirements_file = root / "requirements.txt"
    if not requirements_file.is_file():
        return

    exe_path = Path(python_executable or sys.executable).expanduser()
    if exe_path.name.lower() == "pythonw.exe":
        console_exe = exe_path.with_name("python.exe")
        if console_exe.is_file():
            exe_path = console_exe
    interpreter = str(exe_path)
    uv_bin = find_uv_executable()
    if uv_bin is not None:
        binary_pkgs = (
            "torch,torchvision,numpy,scipy,matplotlib,opencv-python,"
            "scikit-image,scikit-learn,SimpleITK,numba,llvmlite,"
            "bottleneck,PySide6,pandas,Pillow"
        )
        subprocess.run(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                interpreter,
                "--only-binary",
                binary_pkgs,
                "-r",
                str(requirements_file),
            ],
            cwd=root,
            check=True,
            timeout=600.0,
        )
    else:
        subprocess.run(
            [interpreter, "-m", "pip", "install", "--quiet", "-r", str(requirements_file)],
            cwd=root,
            check=True,
            timeout=600.0,
        )


def restart_current_process() -> None:
    """Restart the current Python process after an in-place code update."""
    env = os.environ.copy()
    env["OCTOLYZER_UPDATED_ONCE"] = "1"
    os.execve(sys.executable, [sys.executable, "-m", "gui.app", *sys.argv[1:]], env)


def main(argv: list[str] | None = None) -> int:
    """CLI hook used by launcher scripts to check/apply updates before opening GUI."""
    import argparse

    parser = argparse.ArgumentParser(description="Update OCTolyzer code and dependencies.")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Update repository files only; let the caller sync dependencies.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    result = check_and_apply_update(
        repo_root,
        sync_dependencies=not args.code_only,
        status_callback=print,
    )
    if result.message:
        print(result.message)
    if result.error:
        print(f"Update warning: {result.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
