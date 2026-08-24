"""Asynchronous process runner used by the OCTolyzer desktop launcher."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal
except ImportError:  # Keep environment discovery usable without GUI dependencies.
    QObject = object
    QProcess = None
    QProcessEnvironment = None
    Signal = None


def build_command(
    executable: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> list[str]:
    """Build the platform-neutral OCTolyzer subprocess command."""
    return [
        str(Path(executable).expanduser()),
        "-u",
        "-m",
        "octolyzer.main",
        "--config",
        str(Path(config_path).expanduser()),
    ]


def build_environment(
    runtime_root: str | os.PathLike[str],
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Prepare an environment in which the selected interpreter sees OCTolyzer."""
    environment = dict(os.environ if base_environment is None else base_environment)
    runtime_path = str(Path(runtime_root).expanduser().resolve())
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        runtime_path
        if not existing_pythonpath
        else os.pathsep.join((runtime_path, existing_pythonpath))
    )
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


if QProcess is not None:

    class ProcessRunner(QObject):
        """Run OCTolyzer without blocking the Qt event loop."""

        output_received = Signal(str)
        state_changed = Signal(str)
        completed = Signal(int, bool)
        failed = Signal(str)

        def __init__(self, parent: QObject | None = None):
            super().__init__(parent)
            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self._read_stdout)
            self.process.readyReadStandardError.connect(self._read_stderr)
            self.process.stateChanged.connect(self._state_changed)
            self.process.finished.connect(self._finished)
            self.process.errorOccurred.connect(self._error_occurred)
            self._stopped = False

        @property
        def running(self) -> bool:
            return self.process.state() != QProcess.ProcessState.NotRunning

        def start(
            self,
            executable: str | os.PathLike[str],
            config_path: str | os.PathLike[str],
            runtime_root: str | os.PathLike[str],
        ) -> None:
            if self.running:
                raise RuntimeError("An OCTolyzer analysis is already running.")

            command = build_command(executable, config_path)
            process_environment = QProcessEnvironment.systemEnvironment()
            for key, value in build_environment(runtime_root).items():
                process_environment.insert(key, value)
            self.process.setProcessEnvironment(process_environment)
            self.process.setWorkingDirectory(str(Path(runtime_root).expanduser().resolve()))
            self._stopped = False
            self.state_changed.emit("starting")
            self.process.start(command[0], command[1:])

        def stop(self) -> None:
            if not self.running:
                return
            self._stopped = True
            self.state_changed.emit("stopping")
            self.process.terminate()

        def _read_stdout(self) -> None:
            text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if text:
                self.output_received.emit(text)

        def _read_stderr(self) -> None:
            text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            if text:
                self.output_received.emit(text)

        def _state_changed(self, state: QProcess.ProcessState) -> None:
            names = {
                QProcess.ProcessState.Starting: "starting",
                QProcess.ProcessState.Running: "running",
                QProcess.ProcessState.NotRunning: "idle",
            }
            self.state_changed.emit(names.get(state, "unknown"))

        def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
            self._read_stdout()
            self._read_stderr()
            was_stopped = self._stopped
            self._stopped = False
            self.completed.emit(exit_code, was_stopped)

        def _error_occurred(self, error: QProcess.ProcessError) -> None:
            if error == QProcess.ProcessError.UnknownError:
                return
            self.failed.emit(self.process.errorString())

else:

    class ProcessRunner:
        """Placeholder that gives a useful error when PySide6 is unavailable."""

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "The OCTolyzer GUI requires PySide6. Install requirements-gui.txt "
                "in the GUI/build environment."
            )
