"""Manage the OS-specific double-click launcher in the repository root."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

LAUNCHER_TEMPLATES_DIR = Path(__file__).resolve().parent / "launchers"
PLATFORM_LAUNCHERS = {
    "windows": "OCTolyzer.bat",
    "darwin": "OCTolyzer.command",
    "linux": "OCTolyzer.sh",
}
ALL_LAUNCHER_FILENAMES = tuple(PLATFORM_LAUNCHERS.values())


def current_platform_key(platform_name: str | None = None) -> str:
    """Normalize sys.platform / os.name to 'windows', 'darwin', or 'linux'."""
    plat = (sys.platform if platform_name is None else platform_name).lower()
    if plat.startswith("win") or plat == "nt":
        return "windows"
    if plat == "darwin":
        return "darwin"
    return "linux"


def expected_launcher_filename(platform_name: str | None = None) -> str:
    """Return the launcher filename appropriate for the target operating system."""
    return PLATFORM_LAUNCHERS[current_platform_key(platform_name)]


def ensure_os_launcher(
    root: str | os.PathLike[str],
    *,
    platform_name: str | None = None,
) -> Path:
    """Place only the current OS's double-click launcher in the repository root."""
    repo_root = Path(root).expanduser().resolve()
    target_filename = expected_launcher_filename(platform_name)
    template_path = LAUNCHER_TEMPLATES_DIR / target_filename
    if not template_path.is_file():
        raise FileNotFoundError(f"Launcher template not found: {template_path}")

    target_path = repo_root / target_filename
    template_bytes = template_path.read_bytes()
    if not target_path.is_file() or target_path.read_bytes() != template_bytes:
        target_path.write_bytes(template_bytes)

    if current_platform_key(platform_name) != "windows":
        try:
            current_mode = target_path.stat().st_mode
            target_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass

    for other_filename in ALL_LAUNCHER_FILENAMES:
        if other_filename == target_filename:
            continue
        other_path = repo_root / other_filename
        if other_path.is_file():
            try:
                other_path.unlink()
            except OSError:
                pass

    return target_path
