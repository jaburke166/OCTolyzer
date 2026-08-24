"""Build the OCTolyzer desktop launcher with Nuitka."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIRECTORY = ROOT / "dist" / "gui"
RUNTIME_DIRECTORIES = ("octolyzer", "figures")


def build_launcher(output_directory: Path, *, clean: bool = False) -> Path:
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if clean:
        for pattern in ("*.dist", "*.build", "OCTolyzerGUI*"):
            for path in output_directory.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    report_path = output_directory / "nuitka-compilation-report.xml"
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--mode=standalone",
        "--enable-plugin=pyside6",
        "--include-package=gui",
        "--nofollow-import-to=octolyzer.analyse",
        "--nofollow-import-to=octolyzer.analyse_slo",
        "--nofollow-import-to=octolyzer.collate_data",
        "--nofollow-import-to=octolyzer.measure",
        "--nofollow-import-to=octolyzer.segment",
        f"--output-dir={output_directory}",
        "--output-filename=OCTolyzerGUI",
        f"--report={report_path}",
        "--product-name=OCTolyzer",
        "--product-version=1.0.0",
        "--file-description=OCTolyzer desktop launcher",
        str(ROOT / "gui" / "app.py"),
    ]
    if os.name == "nt":
        command.append("--windows-console-mode=disable")

    subprocess.run(command, cwd=ROOT, check=True)
    distribution_directory = _find_distribution_directory(output_directory)
    _copy_runtime_payload(distribution_directory)
    return distribution_directory


def _find_distribution_directory(output_directory: Path) -> Path:
    candidates = sorted(output_directory.glob("*.dist"))
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(f"Expected one Nuitka distribution directory, found: {names}")
    return candidates[0]


def _copy_runtime_payload(distribution_directory: Path) -> None:
    runtime_directory = distribution_directory / "runtime"
    if runtime_directory.exists():
        shutil.rmtree(runtime_directory)
    runtime_directory.mkdir()

    for directory_name in RUNTIME_DIRECTORIES:
        shutil.copytree(ROOT / directory_name, runtime_directory / directory_name)
    shutil.copy2(ROOT / "config.txt", runtime_directory / "config.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the OCTolyzer GUI with Nuitka.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BUILD_DIRECTORY,
        help="Directory in which to place the Nuitka distribution.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous Nuitka distribution and compiler outputs before compiling.",
    )
    arguments = parser.parse_args(argv)
    distribution_directory = build_launcher(arguments.output_dir, clean=arguments.clean)
    print(f"Built OCTolyzer GUI in {distribution_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
