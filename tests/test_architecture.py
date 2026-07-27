"""Tests for dependency and distribution guardrails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


architecture = _load_script("check_architecture")
wheel_check = _load_script("check_wheel")


class ArchitectureGuardrailTests(unittest.TestCase):
    def test_repository_respects_layer_direction(self):
        violations = architecture.validate_repository(PROJECT_ROOT)
        self.assertEqual(
            [violation.render(PROJECT_ROOT) for violation in violations],
            [],
        )

    def test_checker_reports_core_to_gui_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layer in architecture.SOURCE_LAYERS:
                (root / layer).mkdir()
                (root / layer / "__init__.py").write_text("", encoding="utf-8")
            bad_module = root / "core" / "bad.py"
            bad_module.write_text(
                "from gui.main_window import RobotGuiMainWindow\n",
                encoding="utf-8",
            )

            violations = architecture.validate_repository(root)

        self.assertEqual(len(violations), 1)
        self.assertIn("core must not import gui", violations[0].message)

    def test_wheel_validator_checks_modules_and_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "ghostgui.whl"
            names = set(wheel_check.REQUIRED_MODULES)
            names.update(
                f"ghostgui-0.1.0{suffix}"
                for suffix in wheel_check.REQUIRED_METADATA_SUFFIXES
            )
            with zipfile.ZipFile(wheel_path, "w") as archive:
                for name in names:
                    content = (
                        "[console_scripts]\n"
                        "ghostgui = application.launcher:main\n"
                        if name.endswith("entry_points.txt")
                        else ""
                    )
                    archive.writestr(name, content)

            errors = wheel_check.validate_wheel(wheel_path)

        self.assertEqual(errors, [])

    def test_wheel_validator_can_require_runtime_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel_path = Path(directory) / "ghostgui.whl"
            names = set(wheel_check.REQUIRED_MODULES)
            names.update(
                f"ghostgui-0.1.0{suffix}"
                for suffix in wheel_check.REQUIRED_METADATA_SUFFIXES
            )
            with zipfile.ZipFile(wheel_path, "w") as archive:
                for name in names:
                    content = (
                        "[console_scripts]\n"
                        "ghostgui = application.launcher:main\n"
                        if name.endswith("entry_points.txt")
                        else ""
                    )
                    archive.writestr(name, content)

            errors = wheel_check.validate_wheel(
                wheel_path,
                require_resources=True,
            )

        self.assertTrue(any("runtime resources" in error for error in errors))

    def test_ci_runs_compatibility_release_and_visual_gates(self):
        workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        ).read_text(encoding="utf-8")

        for required in (
            'python-version: ["3.10", "3.13"]',
            "scripts/run_test_suite.py",
            "scripts/check_architecture.py",
            "scripts/check_docs.py",
            "scripts/check_wheel.py",
            "--require-resources",
            "scripts/smoke_installed_package.py",
            "xvfb-run",
            "tests.test_visual_smoke",
        ):
            self.assertIn(required, workflow)


if __name__ == "__main__":
    unittest.main()
