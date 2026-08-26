import hashlib
import io
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from gui.bootstrap import (BootstrapCancelled, BootstrapError,
                           BootstrapStage, assemble_candidate,
                           build_install_command, build_python_install_command,
                           build_venv_command, download_uv, find_local_uv,
                           run_bootstrap, run_streaming,
                           torch_preinstall_command, venv_python_path)
from gui.environment import EnvironmentCandidate


class _FakeResponse:
    """Minimal stand-in for the object urllib.request.urlopen() returns."""

    def __init__(self, data: bytes, headers: dict | None = None):
        self._data = data
        self.headers = headers or {}
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._position:]
            self._position = len(self._data)
            return chunk
        chunk = self._data[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _tar_gz_bytes(member_name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _zip_bytes(member_name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member_name, content)
    return buffer.getvalue()


class CommandBuilderTests(unittest.TestCase):
    def test_python_install_command(self):
        self.assertEqual(
            build_python_install_command("/tmp/uv"),
            ["/tmp/uv", "python", "install", "3.11"],
        )

    def test_venv_command(self):
        self.assertEqual(
            build_venv_command("/tmp/uv", "/tmp/env"),
            ["/tmp/uv", "venv", "/tmp/env", "--python", "3.11"],
        )

    def test_venv_python_path_uses_platform_layout(self):
        with patch("gui.bootstrap.platform.system", return_value="Windows"):
            self.assertEqual(venv_python_path("C:/env"), Path("C:/env/Scripts/python.exe"))
        with patch("gui.bootstrap.platform.system", return_value="Linux"):
            self.assertEqual(venv_python_path("/tmp/env"), Path("/tmp/env/bin/python"))

    def test_torch_preinstall_command_has_no_index_url_selection(self):
        command = torch_preinstall_command("/tmp/uv", "/tmp/env/bin/python")
        self.assertEqual(
            command,
            ["/tmp/uv", "pip", "install", "--python", "/tmp/env/bin/python", "torch", "torchvision"],
        )
        self.assertNotIn("--index-url", command)

    def test_install_command_includes_find_links_by_default(self):
        command = build_install_command("/tmp/uv", "/tmp/env/bin/python", "requirements.txt")
        self.assertIn("--find-links", command)
        self.assertIn("-r", command)
        self.assertIn("requirements.txt", command)

    def test_install_command_find_links_can_be_disabled(self):
        command = build_install_command(
            "/tmp/uv", "/tmp/env/bin/python", "requirements.txt", find_links=None
        )
        self.assertNotIn("--find-links", command)


class BootstrapErrorTests(unittest.TestCase):
    def test_str_includes_remediation_when_present(self):
        error = BootstrapError("Download failed.", remediation="Check your internet connection.")
        self.assertEqual(str(error), "Download failed. Check your internet connection.")

    def test_str_is_message_only_without_remediation(self):
        error = BootstrapError("Download failed.")
        self.assertEqual(str(error), "Download failed.")


class FindLocalUvTests(unittest.TestCase):
    def test_prefers_cached_binary_over_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory) / "cache"
            tools_directory = cache_directory / "bootstrap" / "uv"
            tools_directory.mkdir(parents=True)
            executable_name = "uv.exe" if sys.platform == "win32" else "uv"
            cached_uv = tools_directory / executable_name
            cached_uv.touch()

            with patch("gui.bootstrap.bootstrap_tools_dir", return_value=tools_directory), patch(
                "gui.bootstrap.shutil.which", return_value="/usr/local/bin/uv"
            ):
                self.assertEqual(find_local_uv(), cached_uv)

    def test_falls_back_to_path_when_nothing_cached(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tools_directory = Path(temporary_directory) / "bootstrap" / "uv"
            with patch("gui.bootstrap.bootstrap_tools_dir", return_value=tools_directory), patch(
                "gui.bootstrap.shutil.which", return_value="/usr/local/bin/uv"
            ):
                self.assertEqual(find_local_uv(), Path("/usr/local/bin/uv"))

    def test_returns_none_when_nothing_found(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tools_directory = Path(temporary_directory) / "bootstrap" / "uv"
            with patch("gui.bootstrap.bootstrap_tools_dir", return_value=tools_directory), patch(
                "gui.bootstrap.shutil.which", return_value=None
            ):
                self.assertIsNone(find_local_uv())


class DownloadUvTests(unittest.TestCase):
    def test_downloads_verifies_and_extracts_tar_archive(self):
        archive_bytes = _tar_gz_bytes("uv-x86_64-unknown-linux-gnu/uv", b"fake-uv-binary")
        checksum = hashlib.sha256(archive_bytes).hexdigest()

        responses = [
            _FakeResponse(f"{checksum}  uv-x86_64-unknown-linux-gnu.tar.gz\n".encode()),
            _FakeResponse(archive_bytes, headers={"Content-Length": str(len(archive_bytes))}),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            tools_directory = Path(temporary_directory) / "bootstrap" / "uv"
            with patch("gui.bootstrap.bootstrap_tools_dir", return_value=tools_directory), patch(
                "gui.bootstrap.platform.system", return_value="Linux"
            ), patch("gui.bootstrap.platform.machine", return_value="x86_64"), patch(
                "gui.bootstrap.urllib.request.urlopen", side_effect=responses
            ):
                progress_calls = []
                destination = download_uv(on_progress=lambda read, total: progress_calls.append((read, total)))

            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes(), b"fake-uv-binary")
            self.assertTrue(progress_calls)

    def test_downloads_and_extracts_zip_archive_on_windows(self):
        archive_bytes = _zip_bytes("uv-x86_64-pc-windows-msvc/uv.exe", b"fake-uv-exe")
        checksum = hashlib.sha256(archive_bytes).hexdigest()

        responses = [
            _FakeResponse(f"{checksum}  uv-x86_64-pc-windows-msvc.zip\n".encode()),
            _FakeResponse(archive_bytes),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            tools_directory = Path(temporary_directory) / "bootstrap" / "uv"
            with patch("gui.bootstrap.bootstrap_tools_dir", return_value=tools_directory), patch(
                "gui.bootstrap.platform.system", return_value="Windows"
            ), patch("gui.bootstrap.platform.machine", return_value="AMD64"), patch(
                "gui.bootstrap.urllib.request.urlopen", side_effect=responses
            ):
                destination = download_uv()

            self.assertEqual(destination.name, "uv.exe")
            self.assertEqual(destination.read_bytes(), b"fake-uv-exe")

    def test_checksum_mismatch_raises_bootstrap_error(self):
        archive_bytes = _tar_gz_bytes("uv-x86_64-unknown-linux-gnu/uv", b"fake-uv-binary")
        responses = [
            _FakeResponse(b"0" * 64 + "  uv-x86_64-unknown-linux-gnu.tar.gz\n".encode()),
            _FakeResponse(archive_bytes),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            tools_directory = Path(temporary_directory) / "bootstrap" / "uv"
            with patch("gui.bootstrap.bootstrap_tools_dir", return_value=tools_directory), patch(
                "gui.bootstrap.platform.system", return_value="Linux"
            ), patch("gui.bootstrap.platform.machine", return_value="x86_64"), patch(
                "gui.bootstrap.urllib.request.urlopen", side_effect=responses
            ):
                with self.assertRaises(BootstrapError):
                    download_uv()

    def test_unsupported_architecture_raises_bootstrap_error(self):
        with patch("gui.bootstrap.platform.system", return_value="Linux"), patch(
            "gui.bootstrap.platform.machine", return_value="sparc64"
        ):
            with self.assertRaises(BootstrapError):
                download_uv()

    def test_download_failure_raises_bootstrap_error_with_remediation(self):
        import urllib.error

        with patch("gui.bootstrap.platform.system", return_value="Linux"), patch(
            "gui.bootstrap.platform.machine", return_value="x86_64"
        ), patch("gui.bootstrap.urllib.request.urlopen", side_effect=urllib.error.URLError("no network")):
            with self.assertRaises(BootstrapError) as context:
                download_uv()
            self.assertIsNotNone(context.exception.remediation)


class RunStreamingTests(unittest.TestCase):
    def test_streams_output_lines_and_returns_exit_code(self):
        lines = []
        exit_code = run_streaming(
            [sys.executable, "-c", "print('first'); print('second')"],
            on_line=lines.append,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(lines, ["first", "second"])

    def test_nonzero_exit_code_is_returned_not_raised(self):
        exit_code = run_streaming([sys.executable, "-c", "raise SystemExit(3)"])
        self.assertEqual(exit_code, 3)

    def test_cancellation_raises_bootstrap_cancelled(self):
        cancel_flag = threading.Event()

        def on_line(_line: str) -> None:
            cancel_flag.set()

        script = "import time\nfor _ in range(50):\n    print('tick', flush=True)\n    time.sleep(0.05)\n"
        with self.assertRaises(BootstrapCancelled):
            run_streaming([sys.executable, "-c", script], on_line=on_line, cancel_flag=cancel_flag)


class AssembleCandidateTests(unittest.TestCase):
    def test_builds_uv_sourced_candidate(self):
        candidate = assemble_candidate("/tmp/env/bin/python")
        self.assertIsInstance(candidate, EnvironmentCandidate)
        self.assertEqual(candidate.executable, Path("/tmp/env/bin/python"))
        self.assertEqual(candidate.source, "uv")


class RunBootstrapOrchestrationTests(unittest.TestCase):
    def test_happy_path_sequences_stages_and_returns_candidate(self):
        stages = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_directory = Path(temporary_directory) / "runtime-env"
            with patch("gui.bootstrap.find_local_uv", return_value=Path("/usr/local/bin/uv")), patch(
                "gui.bootstrap.runtime_env_dir", return_value=env_directory
            ), patch("gui.bootstrap.run_streaming", return_value=0) as mock_run_streaming:
                candidate = run_bootstrap(
                    requirements_path="requirements.txt",
                    on_stage=stages.append,
                )

        self.assertEqual(
            stages,
            [
                BootstrapStage.CHECKING_TOOLS,
                BootstrapStage.INSTALLING_PYTHON,
                BootstrapStage.CREATING_ENVIRONMENT,
                BootstrapStage.INSTALLING_PACKAGES,
                BootstrapStage.VERIFYING,
                BootstrapStage.DONE,
            ],
        )
        self.assertEqual(candidate.source, "uv")
        self.assertEqual(mock_run_streaming.call_count, 4)  # python install, venv, torch, requirements

    def test_downloads_uv_when_not_found_locally(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_directory = Path(temporary_directory) / "runtime-env"
            with patch("gui.bootstrap.find_local_uv", return_value=None), patch(
                "gui.bootstrap.download_uv", return_value=Path("/tmp/uv")
            ) as mock_download, patch(
                "gui.bootstrap.runtime_env_dir", return_value=env_directory
            ), patch("gui.bootstrap.run_streaming", return_value=0):
                run_bootstrap(requirements_path="requirements.txt")

        mock_download.assert_called_once()

    def test_failed_step_raises_bootstrap_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_directory = Path(temporary_directory) / "runtime-env"
            with patch("gui.bootstrap.find_local_uv", return_value=Path("/usr/local/bin/uv")), patch(
                "gui.bootstrap.runtime_env_dir", return_value=env_directory
            ), patch("gui.bootstrap.run_streaming", return_value=1):
                with self.assertRaises(BootstrapError):
                    run_bootstrap(requirements_path="requirements.txt")


if __name__ == "__main__":
    unittest.main()
