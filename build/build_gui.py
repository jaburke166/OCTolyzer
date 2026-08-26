"""Build the OCTolyzer desktop launcher with Nuitka."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIRECTORY = ROOT / "dist" / "gui"
# gui/assets is bundled alongside octolyzer/figures so MainWindow can resolve
# its window icon (gui/app.py:_apply_window_icon) the same way in dev
# checkouts and frozen Nuitka builds -- both look under the runtime payload.
RUNTIME_DIRECTORIES = ("octolyzer", "figures", "gui/assets")
DEFAULT_PRODUCT_VERSION = "0.0.0-dev"
LAUNCHER_NAME = "OCTolyzerGUI"
ICON_ICO = ROOT / "gui" / "assets" / "icon.ico"
ICON_ICNS = ROOT / "gui" / "assets" / "icon.icns"


def build_launcher(
    output_directory: Path,
    *,
    clean: bool = False,
    product_version: str = DEFAULT_PRODUCT_VERSION,
) -> Path:
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if clean:
        for pattern in ("*.dist", "*.build", "*.app", "OCTolyzerGUI*", "runtime-staging"):
            for path in output_directory.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    staged_runtime = output_directory / "runtime-staging"
    _copy_runtime_payload(staged_runtime)
    report_path = output_directory / "nuitka-compilation-report.xml"
    # "app" produces a real .app bundle on macOS (standalone folder layout under
    # Contents/MacOS); everywhere else it is equivalent to "onefile", which is what
    # we want for a single .exe on Windows and a single binary to wrap in an
    # AppImage on Linux.
    packaging_mode = "app" if sys.platform == "darwin" else "onefile"
    command = [
        sys.executable,
        "-m",
        "nuitka",
        f"--mode={packaging_mode}",
        # Avoids blocking on interactive prompts in CI when Nuitka needs to fetch
        # a helper tool (ccache, dependency walker, etc.).
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        "--include-package=gui",
        "--nofollow-import-to=octolyzer.analyse",
        "--nofollow-import-to=octolyzer.analyse_slo",
        "--nofollow-import-to=octolyzer.collate_data",
        "--nofollow-import-to=octolyzer.measure",
        "--nofollow-import-to=octolyzer.segment",
        f"--output-dir={output_directory}",
        f"--output-filename={LAUNCHER_NAME}",
        # --include-data-dir silently drops .py/.so/.dylib files as "code", which
        # strips the octolyzer package out of the payload. --include-raw-dir copies
        # everything verbatim, which is what a source tree meant for an external
        # interpreter needs.
        f"--include-raw-dir={staged_runtime}=runtime",
        f"--report={report_path}",
        "--product-name=OCTolyzer",
        # Windows PE version resources require a numeric dotted version, so a
        # tag like v1.2.3-rc1 is normalized before reaching Nuitka.
        f"--product-version={_numeric_product_version(product_version)}",
        "--file-description=OCTolyzer desktop launcher",
        str(ROOT / "gui" / "app.py"),
    ]
    if os.name == "nt":
        command.append("--windows-console-mode=disable")
        if ICON_ICO.is_file():
            command.append(f"--windows-icon-from-ico={ICON_ICO}")
    if sys.platform == "darwin" and ICON_ICNS.is_file():
        command.append(f"--macos-app-icon={ICON_ICNS}")

    subprocess.run(command, cwd=ROOT, check=True)
    shutil.rmtree(staged_runtime)

    if sys.platform == "darwin":
        # Nuitka names the bundle after the input file (app.app), not
        # --output-filename, so rename it to match.
        application = output_directory / f"{LAUNCHER_NAME}.app"
        generated_applications = sorted(output_directory.glob("*.app"))
        if not application.is_dir() and len(generated_applications) == 1:
            generated_applications[0].rename(application)
        if not application.is_dir():
            raise RuntimeError(f"Nuitka did not produce the expected app bundle: {application}")
        return application

    executable = output_directory / (LAUNCHER_NAME + (".exe" if os.name == "nt" else ""))
    if not executable.is_file():
        raise RuntimeError(f"Nuitka did not produce the expected launcher: {executable}")
    if sys.platform.startswith("linux"):
        return _create_linux_appimage(executable, output_directory)
    return executable


def _copy_runtime_payload(runtime_directory: Path) -> None:
    """Stage files that the GUI launches in the selected processing environment."""
    if runtime_directory.exists():
        shutil.rmtree(runtime_directory)
    runtime_directory.mkdir()

    # --include-raw-dir copies the staged tree verbatim (no code-file filtering),
    # so strip bytecode caches here rather than shipping stale .pyc files.
    ignore_cache = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for directory_name in RUNTIME_DIRECTORIES:
        shutil.copytree(
            ROOT / directory_name, runtime_directory / directory_name, ignore=ignore_cache
        )
    shutil.copy2(ROOT / "config.txt", runtime_directory / "config.txt")
    # Bundled so the auto-bootstrap flow (gui/bootstrap.py) can install the
    # exact pinned dependency set even when the user has no git checkout --
    # only the installed app itself.
    shutil.copy2(ROOT / "requirements.txt", runtime_directory / "requirements.txt")


def _numeric_product_version(version: str) -> str:
    """Reduce a version string (e.g. a git tag) to Windows PE's N.N.N.N form."""
    match = re.match(r"v?(\d+(?:\.\d+){0,3})", version)
    return match.group(1) if match else "0.0.0"


def _create_linux_appimage(executable: Path, output_directory: Path) -> Path:
    appimagetool = shutil.which("appimagetool")
    if appimagetool is None:
        raise RuntimeError("Linux builds require appimagetool on PATH to create an AppImage.")
    appimage = output_directory / f"{LAUNCHER_NAME}.AppImage"
    with tempfile.TemporaryDirectory() as staging_directory:
        app_dir = Path(staging_directory) / "OCTolyzer.AppDir"
        application_directory = app_dir / "usr" / "bin"
        application_directory.mkdir(parents=True)
        shutil.copy2(executable, application_directory / LAUNCHER_NAME)
        (application_directory / LAUNCHER_NAME).chmod(0o755)
        # The AppImage runtime execv()s $APPDIR/AppRun on launch -- without
        # this file present, running the AppImage fails immediately with
        # "execv error: No such file or directory". appimagetool does not
        # create or require one at build time, so it has to be written here.
        app_run = app_dir / "AppRun"
        app_run.write_text(
            "#!/bin/sh\n"
            'HERE="$(dirname "$(readlink -f "${0}")")"\n'
            f'exec "${{HERE}}/usr/bin/{LAUNCHER_NAME}" "$@"\n',
            encoding="utf-8",
        )
        app_run.chmod(0o755)
        (app_dir / "OCTolyzer.desktop").write_text(
            "[Desktop Entry]\nName=OCTolyzer\nExec=OCTolyzerGUI\nIcon=OCTolyzer\n"
            "Type=Application\nCategories=Science;\n",
            encoding="utf-8",
        )
        icon_source = ROOT / "gui" / "assets" / "icon-256.png"
        if icon_source.is_file():
            # appimagetool looks for the icon named after the .desktop file's
            # Icon= key at the AppDir root; also install it into the standard
            # hicolor theme path for desktop environments that read that instead.
            shutil.copy2(icon_source, app_dir / "OCTolyzer.png")
            themed_icon_directory = app_dir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
            themed_icon_directory.mkdir(parents=True)
            shutil.copy2(icon_source, themed_icon_directory / "OCTolyzer.png")
        subprocess.run([appimagetool, str(app_dir), str(appimage)], check=True)
    executable.unlink()
    return appimage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the OCTolyzer GUI with Nuitka.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BUILD_DIRECTORY,
        help="Directory in which to place the packaged artifact.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous Nuitka distribution and compiler outputs before compiling.",
    )
    parser.add_argument(
        "--product-version",
        default=DEFAULT_PRODUCT_VERSION,
        help="Version to embed in the built artifact (e.g. a git tag with the leading 'v' stripped).",
    )
    arguments = parser.parse_args(argv)
    artifact = build_launcher(
        arguments.output_dir,
        clean=arguments.clean,
        product_version=arguments.product_version,
    )
    print(f"Built OCTolyzer GUI artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
