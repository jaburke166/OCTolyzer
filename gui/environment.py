"""Cross-platform discovery and compatibility checks for Python environments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "eyepy": "eyepy",
    "SimpleITK": "SimpleITK",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "opencv-python": "cv2",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "timm": "timm",
    "numba": "numba",
    "openpyxl": "openpyxl",
}

CACHE_VERSION = 1
DISCOVERY_CACHE_NAME = "environment-discovery.json"
PROBE_CACHE_NAME = "environment-probes.json"
IGNORED_DIRECTORY_NAMES = {".git", ".hg", ".svn", "__pycache__", "node_modules", "pkgs"}
LOCAL_ENVIRONMENT_NAMES = (".venv", "venv", ".env", "env")

PROBE_SCRIPT = r'''
import importlib
import json
import platform
import sys

modules = {module_name: import_name for module_name, import_name in MODULES.items()}
result = {
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "executable": sys.executable,
    "packages": {},
    "octolyzer": None,
    "torch": {"version": None, "cuda": False, "mps": False},
}

for package_name, import_name in modules.items():
    try:
        module = importlib.import_module(import_name)
        result["packages"][package_name] = {
            "ok": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as error:
        result["packages"][package_name] = {"ok": False, "error": str(error)}

try:
    import octolyzer
    result["octolyzer"] = {"ok": True, "location": getattr(octolyzer, "__file__", None)}
except Exception as error:
    result["octolyzer"] = {"ok": False, "error": str(error)}

try:
    import torch
    result["torch"]["version"] = torch.__version__
    result["torch"]["cuda"] = bool(torch.cuda.is_available())
    result["torch"]["mps"] = bool(torch.backends.mps.is_available())
except Exception:
    pass

print(json.dumps(result))
'''


@dataclass(frozen=True)
class EnvironmentCandidate:
    executable: Path
    label: str
    source: str

    def __str__(self) -> str:
        return self.label


@dataclass
class EnvironmentProbe:
    executable: Path
    python_version: str | None = None
    python_implementation: str | None = None
    packages: dict[str, dict[str, Any]] = field(default_factory=dict)
    octolyzer: dict[str, Any] | None = None
    torch: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    error: str | None = None

    @property
    def missing_packages(self) -> list[str]:
        return [
            package_name
            for package_name, status in self.packages.items()
            if not status.get("ok", False)
        ]

    @property
    def summary(self) -> str:
        if self.error:
            return self.error
        if self.ok:
            backend = "CPU"
            if self.torch.get("cuda"):
                backend = "CUDA"
            elif self.torch.get("mps"):
                backend = "Apple GPU"
            return f"Compatible: Python {self.python_version}, Torch backend {backend}"
        missing = ", ".join(self.missing_packages)
        if missing:
            return f"Missing or broken packages: {missing}"
        return "Environment is not compatible with OCTolyzer."


def discover_environments(
    workspace_root: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
) -> list[EnvironmentCandidate]:
    """Find Python environments using bounded, provider-aware locations."""
    candidates: list[EnvironmentCandidate] = []

    for root in _workspace_roots(workspace_root):
        candidates.extend(_workspace_environments(root))

    executable_candidates = [
        (Path(sys.executable), "Current Python", "current"),
        (_which("python"), "python on PATH", "PATH"),
        (_which("python3"), "python3 on PATH", "PATH"),
    ]
    for executable, label, source in executable_candidates:
        if executable is not None:
            candidates.append(EnvironmentCandidate(executable, label, source))

    candidates.extend(_conda_environments())
    candidates.extend(_managed_environments())
    candidates.extend(_system_python_environments())
    # A venv's bin/python is a symlink chain that fully resolves to the
    # exact same physical interpreter binary as whatever bare interpreter
    # built it -- e.g. the "python3 on PATH" entry above, or the "uv:
    # <interpreter>" entry _managed_environments() finds under
    # ~/.local/share/uv/python/. Scan order here doesn't matter: when that
    # collision happens, _deduplicate_candidates() is what guarantees the
    # populated venv wins over whichever generically-labeled interpreter it
    # happens to share a binary with, not whoever runs first.
    candidates.extend(_bootstrapped_environment())

    unique = _deduplicate_candidates(candidates)
    save_discovery_cache(unique)
    return unique


def _bootstrapped_environment() -> list[EnvironmentCandidate]:
    """Find the venv created by the automatic setup flow (gui/bootstrap.py).

    This has to be a real, persistent discovery source -- not just the
    in-memory candidate gui/app.py splices in right after bootstrap
    finishes -- so the environment survives an app restart or a manual
    refresh instead of silently vanishing on the next discovery scan.
    """
    runtime_env = _cache_directory() / "runtime-env"
    executable = _environment_python(runtime_env)
    if executable is None:
        return []
    return [EnvironmentCandidate(executable, "OCTolyzer (auto-installed)", "bootstrap")]


def load_cached_environments() -> list[EnvironmentCandidate]:
    """Load still-existing environment paths without running any discovery tools."""
    payload = _read_cache(DISCOVERY_CACHE_NAME)
    if payload.get("version") != CACHE_VERSION:
        return []

    candidates: list[EnvironmentCandidate] = []
    for item in payload.get("environments", []):
        if not isinstance(item, dict):
            continue
        executable = Path(str(item.get("executable", ""))).expanduser()
        try:
            stat = executable.stat()
        except (OSError, ValueError):
            continue
        if item.get("mtime_ns") != stat.st_mtime_ns or item.get("size") != stat.st_size:
            continue
        candidates.append(
            EnvironmentCandidate(
                executable,
                str(item.get("label") or executable.parent),
                str(item.get("source") or "cache"),
            )
        )
    return _deduplicate_candidates(candidates)


def probe_environment(
    executable: str | os.PathLike[str],
    *,
    runtime_root: str | os.PathLike[str] | None = None,
    timeout: float = 45,
) -> EnvironmentProbe:
    """Run an isolated compatibility probe using the selected interpreter."""
    executable_path = Path(executable).expanduser()
    environment = os.environ.copy()
    if runtime_root is not None:
        runtime_path = str(Path(runtime_root).expanduser())
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            runtime_path
            if not existing_pythonpath
            else os.pathsep.join((runtime_path, existing_pythonpath))
        )

    script = f"MODULES = {REQUIRED_MODULES!r}\n{PROBE_SCRIPT}"
    try:
        completed = subprocess.run(
            [str(executable_path), "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return EnvironmentProbe(executable_path, error=str(error))

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return EnvironmentProbe(
            executable_path,
            error=f"Probe failed with exit code {completed.returncode}: {detail}",
        )

    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return EnvironmentProbe(executable_path, error=f"Probe returned invalid data: {error}. {detail}")

    packages = payload.get("packages", {})
    octolyzer_status = payload.get("octolyzer") or {}
    probe = EnvironmentProbe(
        executable=executable_path,
        python_version=payload.get("python_version"),
        python_implementation=payload.get("python_implementation"),
        packages=packages,
        octolyzer=octolyzer_status,
        torch=payload.get("torch", {}),
    )
    probe.ok = bool(
        probe.python_implementation == "CPython"
        and all(status.get("ok", False) for status in packages.values())
        and octolyzer_status.get("ok", False)
    )
    return probe


def load_cached_probe(
    executable: str | os.PathLike[str],
    *,
    runtime_root: str | os.PathLike[str] | None = None,
) -> EnvironmentProbe | None:
    """Return a cached probe when the interpreter and its environment are unchanged."""
    executable_path = _resolved_path(Path(executable).expanduser())
    payload = _read_cache(PROBE_CACHE_NAME)
    if payload.get("version") != CACHE_VERSION:
        return None
    signature = _environment_signature(executable_path)
    runtime = str(_resolved_path(Path(runtime_root).expanduser())) if runtime_root is not None else ""
    for item in payload.get("probes", []):
        if not isinstance(item, dict):
            continue
        if item.get("executable") != str(executable_path):
            continue
        if item.get("signature") != signature or item.get("runtime_root", "") != runtime:
            continue
        try:
            return EnvironmentProbe(
                executable=executable_path,
                python_version=item.get("python_version"),
                python_implementation=item.get("python_implementation"),
                packages=item.get("packages") or {},
                octolyzer=item.get("octolyzer"),
                torch=item.get("torch") or {},
                ok=bool(item.get("ok")),
                error=item.get("error"),
            )
        except (AttributeError, TypeError):
            return None
    return None


def save_probe_cache(
    probe: EnvironmentProbe,
    *,
    runtime_root: str | os.PathLike[str] | None = None,
) -> None:
    """Persist a compatibility result for reuse by the next GUI session."""
    executable_path = _resolved_path(Path(probe.executable).expanduser())
    runtime = str(_resolved_path(Path(runtime_root).expanduser())) if runtime_root is not None else ""
    payload = _read_cache(PROBE_CACHE_NAME)
    entries = [item for item in payload.get("probes", []) if isinstance(item, dict)]
    entries = [item for item in entries if item.get("executable") != str(executable_path)]
    entries.append(
        {
            "executable": str(executable_path),
            "signature": _environment_signature(executable_path),
            "runtime_root": runtime,
            "python_version": probe.python_version,
            "python_implementation": probe.python_implementation,
            "packages": probe.packages,
            "octolyzer": probe.octolyzer,
            "torch": probe.torch,
            "ok": probe.ok,
            "error": probe.error,
            "saved_at": time.time(),
        }
    )
    _write_cache(PROBE_CACHE_NAME, {"version": CACHE_VERSION, "probes": entries[-64:]})


def clear_cached_probe(
    executable: str | os.PathLike[str],
    *,
    runtime_root: str | os.PathLike[str] | None = None,
) -> None:
    """Remove the cached result for one interpreter and runtime payload."""
    executable_path = _resolved_path(Path(executable).expanduser())
    runtime = str(_resolved_path(Path(runtime_root).expanduser())) if runtime_root is not None else ""
    payload = _read_cache(PROBE_CACHE_NAME)
    entries = [
        item
        for item in payload.get("probes", [])
        if not (
            isinstance(item, dict)
            and item.get("executable") == str(executable_path)
            and item.get("runtime_root", "") == runtime
        )
    ]
    _write_cache(PROBE_CACHE_NAME, {"version": CACHE_VERSION, "probes": entries})


def _conda_environments() -> list[EnvironmentCandidate]:
    candidates: list[EnvironmentCandidate] = []
    for conda_executable in _conda_executables():
        try:
            completed = subprocess.run(
                [str(conda_executable), "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if completed.returncode != 0:
                continue
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue

        for environment_path in payload.get("envs", []):
            environment = Path(environment_path)
            executable = _environment_python(environment)
            if executable is not None:
                candidates.append(EnvironmentCandidate(executable, f"Conda: {environment.name}", "Conda"))
    return candidates


def _conda_executables() -> list[Path]:
    """Find Conda directly when the GUI was launched outside an activated shell."""
    home = Path.home()
    configured = os.environ.get("CONDA_EXE")
    candidates = [Path(configured).expanduser()] if configured else []
    path_conda = shutil.which("conda")
    if path_conda:
        candidates.append(Path(path_conda))

    roots = [
        home / "anaconda3",
        home / "miniconda3",
        home / "miniforge3",
        home / "mambaforge",
        home / "anaconda",
        home / "miniconda",
        Path("/opt") / "anaconda3",
        Path("/opt") / "miniconda3",
        Path("/opt") / "miniforge3",
        Path("/opt") / "mambaforge",
    ]
    if os.name == "nt":
        local_appdata = Path(os.environ.get("LOCALAPPDATA", home)).expanduser()
        program_data = Path(os.environ.get("PROGRAMDATA", home)).expanduser()
        roots.extend(
            [
                local_appdata / "Anaconda3",
                local_appdata / "Miniconda3",
                local_appdata / "Miniforge3",
                program_data / "Anaconda3",
                program_data / "Miniconda3",
            ]
        )

    executable_name = "conda.exe" if os.name == "nt" else "conda"
    candidates.extend(root / "bin" / executable_name for root in roots)
    if os.name == "nt":
        candidates.extend(root / "Scripts" / executable_name for root in roots)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = _resolved_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _environment_python(environment: Path) -> Path | None:
    if os.name == "nt":
        executable = environment / "python.exe"
    else:
        executable = environment / "bin" / "python"
    return executable if executable.is_file() else None


def _workspace_roots(
    workspace_root: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None,
) -> list[Path]:
    if workspace_root is None:
        roots = [Path.cwd()]
    elif isinstance(workspace_root, (str, os.PathLike)):
        roots = [Path(workspace_root).expanduser()]
    else:
        roots = [Path(root).expanduser() for root in workspace_root]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve(strict=False)
        except OSError:
            resolved = root
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _workspace_environments(root: Path) -> list[EnvironmentCandidate]:
    candidates: list[EnvironmentCandidate] = []
    for name in LOCAL_ENVIRONMENT_NAMES:
        candidates.extend(_environment_directory_candidate(root / name, f"Workspace: {name}", "workspace"))

    candidates.extend(_pixi_environments(root))

    try:
        children = list(root.iterdir())
    except OSError:
        children = []
    for child in children:
        if not child.is_dir() or child.name in IGNORED_DIRECTORY_NAMES:
            continue
        for name in LOCAL_ENVIRONMENT_NAMES:
            candidates.extend(
                _environment_directory_candidate(
                    child / name,
                    f"Workspace: {child.name}/{name}",
                    "workspace",
                )
            )
        candidates.extend(_pixi_environments(child))
    return candidates


def _pixi_environments(root: Path) -> list[EnvironmentCandidate]:
    pixi_root = root / ".pixi" / "envs"
    if not pixi_root.is_dir():
        return []
    candidates: list[EnvironmentCandidate] = []
    try:
        environments = list(pixi_root.iterdir())
    except OSError:
        return []
    for environment in environments:
        if environment.is_dir():
            candidates.extend(
                _environment_directory_candidate(
                    environment,
                    f"Pixi: {environment.name}",
                    "pixi",
                )
            )
    return candidates


def _managed_environments() -> list[EnvironmentCandidate]:
    candidates: list[EnvironmentCandidate] = []
    home = Path.home()
    local_appdata = Path(os.environ.get("LOCALAPPDATA", home)).expanduser()

    pyenv_root = Path(os.environ.get("PYENV_ROOT", home / ".pyenv")).expanduser()
    versions_root = pyenv_root / "versions"
    for version in _directories(versions_root):
        candidates.extend(_environment_directory_candidate(version, f"pyenv: {version.name}", "pyenv"))
        for virtual_environment in _directories(version / "envs"):
            candidates.extend(
                _environment_directory_candidate(
                    virtual_environment,
                    f"pyenv-virtualenv: {version.name}/{virtual_environment.name}",
                    "pyenv-virtualenv",
                )
            )

    poetry_roots = [
        home / ".cache" / "pypoetry" / "virtualenvs",
        home / ".local" / "share" / "pypoetry" / "virtualenvs",
    ]
    pipenv_roots = [home / ".local" / "share" / "virtualenvs", home / ".virtualenvs"]
    for root in poetry_roots:
        candidates.extend(_managed_environment_directories(root, "Poetry", "poetry"))
    for root in pipenv_roots:
        candidates.extend(_managed_environment_directories(root, "Pipenv", "pipenv"))

    hatch_root = home / ".local" / "share" / "hatch" / "env" / "virtual"
    for project in _directories(hatch_root):
        for environment in _directories(project):
            candidates.extend(
                _environment_directory_candidate(
                    environment,
                    f"Hatch: {project.name}/{environment.name}",
                    "hatch",
                )
            )

    uv_roots = [
        home / ".local" / "share" / "uv" / "python",
        home / ".cache" / "uv" / "python",
        home / "Library" / "Application Support" / "uv" / "python",
        local_appdata / "uv" / "python",
    ]
    for root in uv_roots:
        for environment in _directories(root):
            candidates.extend(
                _environment_directory_candidate(environment, f"uv: {environment.name}", "uv")
            )

    for root in (home / ".venvs",):
        candidates.extend(_managed_environment_directories(root, "Virtualenv", "virtualenv"))
    return candidates


def _system_python_environments() -> list[EnvironmentCandidate]:
    candidates: list[EnvironmentCandidate] = []
    home = Path.home()
    if os.name == "nt":
        roots = [
            Path(os.environ.get("LOCALAPPDATA", home)).expanduser() / "Programs" / "Python",
            Path(os.environ.get("PROGRAMFILES", home)).expanduser() / "Python",
        ]
        for root in roots:
            for installation in _directories(root):
                executable = installation / "python.exe"
                if executable.is_file():
                    candidates.append(
                        EnvironmentCandidate(executable, f"System Python: {installation.name}", "system")
                    )
        return candidates

    executable_paths = [
        Path("/usr/bin/python"),
        Path("/usr/bin/python3"),
        Path("/usr/local/bin/python"),
        Path("/usr/local/bin/python3"),
        Path("/opt/homebrew/bin/python3"),
        Path.home() / ".local" / "bin" / "python",
        Path.home() / ".local" / "bin" / "python3",
    ]
    for executable in executable_paths:
        if executable.is_file():
            candidates.append(
                EnvironmentCandidate(executable, f"System Python: {executable}", "system")
            )
    return candidates


def _common_python_paths(
    workspace_root: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
) -> list[Path]:
    """Return bounded provider paths kept for compatibility with older callers."""
    candidates: list[EnvironmentCandidate] = []
    for root in _workspace_roots(workspace_root):
        candidates.extend(_workspace_environments(root))
    candidates.extend(_managed_environments())
    candidates.extend(_system_python_environments())
    return [candidate.executable for candidate in _deduplicate_candidates(candidates)]


def _managed_environment_directories(
    root: Path,
    provider: str,
    source: str,
) -> list[EnvironmentCandidate]:
    return [
        candidate
        for directory in _directories(root)
        for candidate in _environment_directory_candidate(directory, f"{provider}: {directory.name}", source)
    ]


def _environment_directory_candidate(
    environment: Path,
    label: str,
    source: str,
) -> list[EnvironmentCandidate]:
    executable = _environment_python(environment)
    if executable is None:
        return []
    return [EnvironmentCandidate(executable, label, source)]


def _directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return [child for child in root.iterdir() if child.is_dir() and child.name not in IGNORED_DIRECTORY_NAMES]
    except OSError:
        return []


def _which(command: str) -> Path | None:
    resolved = shutil.which(command)
    return Path(resolved) if resolved else None


# Sources that should always win a collision, regardless of scan order: a
# venv's own interpreter binary is a symlink chain that fully resolves to the
# exact same real file as whatever bare interpreter created it (uv-managed
# Python, system Python, pyenv, etc. -- not just one specific source), so
# relying on which source happens to run first is fragile. Any of these
# purpose-built, more-informative labels should win over a generic one.
_PREFERRED_CANDIDATE_SOURCES = frozenset({"manual", "bootstrap"})


def _deduplicate_candidates(candidates: list[EnvironmentCandidate]) -> list[EnvironmentCandidate]:
    order: list[Path] = []
    best: dict[Path, EnvironmentCandidate] = {}
    for candidate in candidates:
        executable = _resolved_path(Path(candidate.executable).expanduser())
        if not executable.is_file() or _is_ignored_executable(executable):
            continue
        existing = best.get(executable)
        if existing is None:
            order.append(executable)
            best[executable] = EnvironmentCandidate(executable, candidate.label, candidate.source)
        elif candidate.source in _PREFERRED_CANDIDATE_SOURCES and existing.source not in _PREFERRED_CANDIDATE_SOURCES:
            best[executable] = EnvironmentCandidate(executable, candidate.label, candidate.source)
    return [best[executable] for executable in order]


def _is_ignored_executable(executable: Path) -> bool:
    return "pkgs" in {part.casefold() for part in executable.parts}


def _resolved_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path


def _cache_directory() -> Path:
    configured = os.environ.get("OCTOLYZER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OCTolyzer"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "octolyzer"


def _read_cache(name: str) -> dict[str, Any]:
    try:
        payload = json.loads((_cache_directory() / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_cache(name: str, payload: dict[str, Any]) -> None:
    cache_directory = _cache_directory()
    try:
        cache_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_directory,
            prefix=f".{name}.",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary)
            temporary_path = Path(temporary.name)
        temporary_path.replace(cache_directory / name)
    except OSError:
        return


def save_discovery_cache(candidates: list[EnvironmentCandidate]) -> None:
    """Persist only lightweight path metadata; discovery remains safe to skip on startup."""
    environments = []
    for candidate in candidates:
        try:
            stat = candidate.executable.stat()
        except OSError:
            continue
        environments.append(
            {
                "executable": str(candidate.executable),
                "label": candidate.label,
                "source": candidate.source,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    _write_cache(DISCOVERY_CACHE_NAME, {"version": CACHE_VERSION, "environments": environments})


def _environment_signature(executable: Path) -> str:
    paths = [executable]
    prefix = executable.parent.parent
    paths.extend((prefix / "pyvenv.cfg", prefix / "conda-meta" / "history"))
    for site_packages in (prefix / "Lib" / "site-packages", prefix / "lib"):
        if site_packages.is_dir():
            paths.append(site_packages)
    signature_parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        signature_parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(signature_parts)
