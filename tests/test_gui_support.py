import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.environment import (EnvironmentCandidate, EnvironmentProbe,
                             _common_python_paths, _conda_environments,
                             _deduplicate_candidates, clear_cached_probe,
                             load_cached_environments, load_cached_probe,
                             save_discovery_cache, save_probe_cache)
from gui.runner import build_command, build_environment


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


if __name__ == "__main__":
    unittest.main()
