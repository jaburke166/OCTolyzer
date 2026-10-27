import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.environment import (EnvironmentCandidate, EnvironmentProbe,
                             _common_python_paths, _conda_environments,
                             _deduplicate_candidates, _environment_python,
                             clear_cached_probe, load_cached_environments,
                             load_cached_probe, probe_environment,
                             save_discovery_cache, save_probe_cache)
from gui.launcher import ensure_os_launcher
from gui.runner import build_command, build_environment
from gui.updater import (VERSION_FILENAME, check_and_apply_update,
                         get_local_commit_sha)


class GuiSupportTests(unittest.TestCase):
    def test_build_command_does_not_depend_on_shell_quoting(self):
        command = build_command("/tmp/python", "/tmp/run with spaces/config.txt")
        self.assertEqual(
            command,
            ["/tmp/python", "-u", "-m", "octolyzer.main", "--config", "/tmp/run with spaces/config.txt"],
        )

    def test_build_environment_prepends_runtime_path(self):
        environment = build_environment(
            "/tmp/runtime",
            base_environment={"PYTHONPATH": "/tmp/existing"},
        )
        self.assertEqual(environment["PYTHONPATH"], os.pathsep.join(["/tmp/runtime", "/tmp/existing"]))
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")

    def test_probe_summary_reports_missing_packages(self):
        probe = EnvironmentProbe(
            executable="/tmp/python",
            python_version="3.11.0",
            packages={"torch": {"ok": False}},
        )
        self.assertEqual(probe.missing_packages, ["torch"])
        self.assertIn("torch", probe.summary)

    def test_bounded_paths_include_workspace_and_ignore_conda_package_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace_python = root / ".venv" / "bin" / "python"
            pixi_python = root / ".pixi" / "envs" / "default" / "bin" / "python"
            cache_python = root / "anaconda3" / "pkgs" / "python-3.11" / "bin" / "python"
            for executable in (workspace_python, pixi_python, cache_python):
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()

            with patch("gui.environment._managed_environments", return_value=[]), patch(
                "gui.environment._system_python_environments", return_value=[]
            ):
                paths = _common_python_paths(root)

            self.assertIn(workspace_python, paths)
            self.assertIn(pixi_python, paths)
            self.assertNotIn(cache_python, paths)

    def test_environment_candidates_are_deduplicated_by_resolved_executable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "python"
            executable.touch()
            alias = executable.parent / "alias"
            alias.symlink_to(executable)
            candidates = _deduplicate_candidates(
                [
                    EnvironmentCandidate(executable, "first", "test"),
                    EnvironmentCandidate(alias, "second", "test"),
                ]
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].label, "first")

    def test_virtualenv_symlink_is_preserved_not_dereferenced_to_base_python(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            base_python = root / "base" / "bin" / "python3.14"
            base_python.parent.mkdir(parents=True)
            base_python.touch()

            venv_root = root / ".venv"
            venv_bin = venv_root / "bin"
            venv_bin.mkdir(parents=True)
            (venv_root / "pyvenv.cfg").write_text("home = /opt/homebrew/bin\n", encoding="utf-8")
            venv_python = venv_bin / "python"
            venv_python.symlink_to(base_python)

            candidates = _deduplicate_candidates(
                [
                    EnvironmentCandidate(venv_python, "Workspace: .venv", "workspace"),
                    EnvironmentCandidate(base_python, "System Python", "system"),
                ]
            )
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0].executable, venv_python)
            self.assertEqual(candidates[1].executable, base_python)

    def test_conda_is_found_from_standard_installation_without_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "anaconda3"
            conda_executable = root / "bin" / "conda"
            environment_python = root / "envs" / "oct-analysis" / "bin" / "python"
            conda_executable.parent.mkdir(parents=True)
            environment_python.parent.mkdir(parents=True)
            conda_executable.touch()
            environment_python.touch()
            completed = unittest.mock.Mock(
                returncode=0,
                stdout=json.dumps({"envs": [str(root), str(environment_python.parents[1])]}),
            )

            with patch.dict(os.environ, {"HOME": temporary_directory}, clear=True), patch(
                "gui.environment.shutil.which", return_value=None
            ), patch("gui.environment.subprocess.run", return_value=completed):
                candidates = _conda_environments()

            self.assertEqual([candidate.executable for candidate in candidates], [environment_python])

    def test_environment_and_probe_caches_use_current_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory) / "cache"
            executable = Path(temporary_directory) / "python"
            executable.touch()
            candidate = EnvironmentCandidate(executable, "Test Python", "test")
            probe = EnvironmentProbe(executable, python_version="3.11", ok=True)
            with patch.dict(os.environ, {"OCTOLYZER_CACHE_DIR": str(cache_directory)}):
                save_discovery_cache([candidate])
                save_probe_cache(probe, runtime_root=temporary_directory)
                self.assertEqual(load_cached_environments()[0].label, "Test Python")
                self.assertTrue(load_cached_probe(executable, runtime_root=temporary_directory).ok)
                clear_cached_probe(executable, runtime_root=temporary_directory)
                self.assertIsNone(load_cached_probe(executable, runtime_root=temporary_directory))

                executable.write_bytes(b"changed")
                self.assertEqual(load_cached_environments(), [])
                self.assertIsNone(load_cached_probe(executable, runtime_root=temporary_directory))

    def test_package_main_launches_gui_by_default_and_cli_when_args_given(self):
        from octolyzer import __main__ as pkg_main

        fake_gui_app = unittest.mock.MagicMock(main=unittest.mock.MagicMock(return_value=0))
        with patch.dict("sys.modules", {"gui.app": fake_gui_app}):
            self.assertEqual(pkg_main.main([]), 0)
            fake_gui_app.main.assert_called_once_with()

        fake_main = unittest.mock.MagicMock(main=unittest.mock.MagicMock(return_value=0))
        with patch.dict("sys.modules", {"octolyzer.main": fake_main}):
            self.assertEqual(pkg_main.main(["--config", "config.txt"]), 0)
            fake_main.main.assert_called_once_with(["--config", "config.txt"])

    def test_auto_update_skips_when_already_at_remote_sha(self):
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / VERSION_FILENAME).write_text(f"{sha}\n", encoding="utf-8")
            with patch("gui.updater.fetch_latest_commit_sha", return_value=sha):
                result = check_and_apply_update(root)
            self.assertFalse(result.updated)
            self.assertEqual(result.local_sha, sha)
            self.assertEqual(result.remote_sha, sha)

    def test_auto_update_applies_archive_and_preserves_local_state(self):
        old_sha = "1" * 40
        new_sha = "2" * 40
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / VERSION_FILENAME).write_text(f"{old_sha}\n", encoding="utf-8")
            (root / "config.txt").write_text("custom_user_config\n", encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv" / "marker.txt").write_text("keep\n", encoding="utf-8")

            def fake_archive_update(target_root, _repo, _branch, **_kwargs):
                (target_root / "new_file.py").write_text("VALUE = 1\n", encoding="utf-8")

            with patch("gui.updater.fetch_latest_commit_sha", return_value=new_sha), patch(
                "gui.updater._can_update_with_git", return_value=False
            ), patch("gui.updater._update_from_archive", side_effect=fake_archive_update), patch(
                "gui.updater.sync_environment_dependencies"
            ) as sync_deps:
                result = check_and_apply_update(root)

            self.assertTrue(result.updated)
            self.assertEqual(get_local_commit_sha(root), new_sha)
            self.assertTrue((root / "new_file.py").is_file())
            self.assertEqual((root / "config.txt").read_text(encoding="utf-8"), "custom_user_config\n")
            self.assertTrue((root / ".venv" / "marker.txt").is_file())
            sync_deps.assert_called_once_with(root.resolve())

    def test_auto_update_respects_disable_env_var(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {"OCTOLYZER_NO_AUTO_UPDATE": "1"}):
                result = check_and_apply_update(temporary_directory)
            self.assertFalse(result.updated)

    def test_ensure_os_launcher_places_only_matching_os_launcher(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "OCTolyzer.bat").write_text("old", encoding="utf-8")
            (root / "OCTolyzer.sh").write_text("old", encoding="utf-8")

            mac_launcher = ensure_os_launcher(root, platform_name="darwin")
            self.assertEqual(mac_launcher.name, "OCTolyzer.command")
            self.assertTrue(mac_launcher.is_file())
            self.assertFalse((root / "OCTolyzer.bat").exists())
            self.assertFalse((root / "OCTolyzer.sh").exists())

            win_launcher = ensure_os_launcher(root, platform_name="win32")
            self.assertEqual(win_launcher.name, "OCTolyzer.bat")
            self.assertTrue(win_launcher.is_file())
            self.assertFalse((root / "OCTolyzer.command").exists())
            self.assertFalse((root / "OCTolyzer.sh").exists())

    def test_windows_pythonw_is_normalized_to_python_exe_and_scripts_venv_is_found(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            scripts_dir = root / ".venv" / "Scripts"
            scripts_dir.mkdir(parents=True)
            (root / ".venv" / "pyvenv.cfg").write_text("home = C:\\Python313\n", encoding="utf-8")
            python_exe = scripts_dir / "python.exe"
            pythonw_exe = scripts_dir / "pythonw.exe"
            python_exe.touch()
            pythonw_exe.touch()

            with patch("gui.environment.os.name", "nt"):
                self.assertEqual(_environment_python(root / ".venv"), python_exe)

            candidates = _deduplicate_candidates(
                [
                    EnvironmentCandidate(python_exe, "Workspace: .venv", "workspace"),
                    EnvironmentCandidate(pythonw_exe, "Current Python", "current"),
                ]
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].executable, python_exe)
            self.assertEqual(candidates[0].label, "Workspace: .venv")

            command = build_command(pythonw_exe, root / "config.txt")
            self.assertEqual(Path(command[0]), python_exe)

    def test_probe_timeout_returns_concise_message_without_dumping_script(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "python"
            executable.touch()
            timeout_exc = unittest.mock.MagicMock(
                side_effect=__import__("subprocess").TimeoutExpired(cmd=["python", "-c", "HUGE_SCRIPT"], timeout=180)
            )
            with patch("gui.environment.subprocess.run", timeout_exc):
                probe = probe_environment(executable, timeout=180)
            self.assertFalse(probe.ok)
            self.assertIn("timed out after 180 seconds", probe.summary)
            self.assertNotIn("HUGE_SCRIPT", probe.summary)


if __name__ == "__main__":
    unittest.main()
