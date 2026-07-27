"""Recovery, migration, path-safety, and resource-location contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from application.project_manager import (
    DEFAULT_PROJECT_FILES,
    GhostGUIProject,
    PROJECT_FILENAME,
    PROJECT_SCHEMA_VERSION,
    ProjectFormatError,
    ProjectSaveTransaction,
    TRANSACTION_JOURNAL_FILENAME,
    migrate_project_metadata,
    recover_project_transactions,
)
from core.resources import bundled_resource_root, resource_path
from core.trajectory import TargetFrame, Trajectory


class FakeTimeline:
    def __init__(self, times=(0.0,), width=3):
        self.times = np.asarray(times, dtype=float)
        self.qpos = np.zeros((len(self.times), width), dtype=float)

    def save_npz(self, path):
        np.savez(
            path,
            schema_version=np.asarray([1]),
            times=self.times,
            qpos=self.qpos,
        )


def schema_v1_metadata(project_name="Legacy"):
    return {
        "schema_version": 1,
        "project_name": project_name,
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-01T00:00:00Z",
        "application": {
            "name": "GhostGUI",
            "project_format": "ghostgui.project.v1",
        },
        "robot": {"model_key": "g1", "model_name": "Unitree G1"},
        "files": dict(DEFAULT_PROJECT_FILES),
    }


class ProjectSchemaTests(unittest.TestCase):
    def test_v1_metadata_migrates_in_memory_to_current_schema(self):
        migrated = migrate_project_metadata(schema_v1_metadata())

        self.assertEqual(migrated["schema_version"], PROJECT_SCHEMA_VERSION)
        self.assertEqual(
            migrated["application"]["project_format"],
            "ghostgui.project.v2",
        )
        self.assertIn("autosave_files", migrated)
        self.assertIn("persistence", migrated)

    def test_open_reports_source_schema_and_save_persists_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / PROJECT_FILENAME).write_text(
                json.dumps(schema_v1_metadata()),
                encoding="utf-8",
            )

            project = GhostGUIProject.open(root)
            self.assertEqual(project.source_schema_version, 1)
            project.save_metadata(update_modified=False)
            persisted = json.loads(
                (root / PROJECT_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(persisted["schema_version"], PROJECT_SCHEMA_VERSION)

    def test_future_schema_is_rejected_with_supported_version(self):
        metadata = schema_v1_metadata()
        metadata["schema_version"] = PROJECT_SCHEMA_VERSION + 1
        with self.assertRaisesRegex(ProjectFormatError, "newer"):
            migrate_project_metadata(metadata)


class ProjectPathSafetyTests(unittest.TestCase):
    def test_metadata_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = migrate_project_metadata(schema_v1_metadata())
            metadata["files"]["workspace"] = "../../outside.json"
            with self.assertRaisesRegex(ProjectFormatError, "safe relative"):
                GhostGUIProject(Path(directory), metadata)

    def test_absolute_and_duplicate_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = migrate_project_metadata(schema_v1_metadata())
            metadata["files"]["workspace"] = "/tmp/outside.json"
            with self.assertRaisesRegex(ProjectFormatError, "must be relative"):
                GhostGUIProject(Path(directory), metadata)

            metadata = migrate_project_metadata(schema_v1_metadata())
            metadata["files"]["workspace"] = metadata["files"]["target_trajectory"]
            with self.assertRaisesRegex(ProjectFormatError, "more than once"):
                GhostGUIProject(Path(directory), metadata)

    def test_existing_symlink_cannot_redirect_a_project_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "redirect").symlink_to(outside, target_is_directory=True)
            metadata = migrate_project_metadata(schema_v1_metadata())
            metadata["files"]["workspace"] = "redirect/workspace.json"

            with self.assertRaisesRegex(ProjectFormatError, "escapes"):
                GhostGUIProject(root, metadata)


class ProjectTransactionTests(unittest.TestCase):
    def test_save_bundle_replaces_a_coherent_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle.ghostgui"
            project = GhostGUIProject.create(root, "Bundle", "g1")
            trajectory = Trajectory()
            trajectory.add_frame(TargetFrame(time=0.0, x=0.25))
            timeline = FakeTimeline(times=(0.0, 0.5))
            workspace = {"current_time": 0.5}

            project.save_bundle(trajectory, timeline, workspace)

            self.assertEqual(
                project.read_trajectory_dict()["tracks"]["pelvis"][0]["x"],
                0.25,
            )
            self.assertEqual(project.read_workspace(), workspace)
            with np.load(project.paths.qpos_timeline) as payload:
                np.testing.assert_allclose(payload["times"], [0.0, 0.5])
            self.assertFalse(
                (root / ".ghostgui-transactions").exists()
            )

    def test_failed_replace_rolls_back_every_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "data" / "first.json"
            second = root / "data" / "second.json"
            first.parent.mkdir()
            first.write_text('{"value": "old-first"}\n', encoding="utf-8")
            second.write_text('{"value": "old-second"}\n', encoding="utf-8")
            transaction = ProjectSaveTransaction(root)
            transaction.stage_json(first, {"value": "new-first"})
            transaction.stage_json(second, {"value": "new-second"})
            replacements = 0

            def fail_second(source, destination):
                nonlocal replacements
                replacements += 1
                if replacements == 2:
                    raise OSError("simulated disk failure")
                os.replace(source, destination)

            with self.assertRaisesRegex(OSError, "simulated"):
                transaction.commit(replace=fail_second)

            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8"))["value"],
                "old-first",
            )
            self.assertEqual(
                json.loads(second.read_text(encoding="utf-8"))["value"],
                "old-second",
            )

    def test_open_recovers_interrupted_replacement_from_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "data" / "workspace.json"
            destination.parent.mkdir(parents=True)
            destination.write_text('{"value": "old"}\n', encoding="utf-8")

            transaction = ProjectSaveTransaction(root)
            transaction.stage_json(destination, {"value": "new"})
            backup = transaction.backup_root / "data" / "workspace.json"
            backup.parent.mkdir(parents=True)
            shutil.copy2(destination, backup)
            journal = {
                "schema_version": 1,
                "transaction_id": transaction.identifier,
                "status": "replacing",
                "operations": ["data/workspace.json"],
                "absent_before": [],
            }
            transaction.journal_path.parent.mkdir(parents=True, exist_ok=True)
            transaction.journal_path.write_text(
                json.dumps(journal),
                encoding="utf-8",
            )
            os.replace(
                transaction.staging_root / "data" / "workspace.json",
                destination,
            )

            recovered = recover_project_transactions(root)

            self.assertEqual(recovered, [transaction.identifier])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["value"],
                "old",
            )
            self.assertFalse(
                (transaction.transaction_root / TRANSACTION_JOURNAL_FILENAME).exists()
            )

    def test_recovery_removes_destination_that_was_absent_before_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "data" / "new.json"
            transaction = ProjectSaveTransaction(root)
            transaction.stage_json(destination, {"value": "new"})
            journal = {
                "schema_version": 1,
                "transaction_id": transaction.identifier,
                "status": "replacing",
                "operations": ["data/new.json"],
                "absent_before": ["data/new.json"],
            }
            transaction.journal_path.parent.mkdir(parents=True, exist_ok=True)
            transaction.journal_path.write_text(
                json.dumps(journal),
                encoding="utf-8",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(
                transaction.staging_root / "data" / "new.json",
                destination,
            )

            recover_project_transactions(root)

            self.assertFalse(destination.exists())
            self.assertFalse(transaction.transaction_root.exists())

    def test_committed_transaction_cleanup_keeps_new_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "data" / "workspace.json"
            transaction = ProjectSaveTransaction(root)
            transaction.stage_json(destination, {"value": "new"})
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(
                transaction.staging_root / "data" / "workspace.json",
                destination,
            )
            transaction.journal_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "transaction_id": transaction.identifier,
                        "status": "committed",
                        "operations": ["data/workspace.json"],
                        "absent_before": ["data/workspace.json"],
                    }
                ),
                encoding="utf-8",
            )

            recovered = recover_project_transactions(root)

            self.assertEqual(recovered, [])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"value": "new"},
            )
            self.assertFalse(transaction.transaction_root.exists())

    def test_corrupt_or_unsafe_journal_is_preserved_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transaction = ProjectSaveTransaction(root)
            transaction.transaction_root.mkdir(parents=True)
            transaction.journal_path.write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(ProjectFormatError, "invalid journal"):
                recover_project_transactions(root)
            self.assertTrue(transaction.transaction_root.exists())

            transaction.journal_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "transaction_id": transaction.identifier,
                        "status": "replacing",
                        "operations": ["../outside.json"],
                        "absent_before": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProjectFormatError, "safe relative"):
                recover_project_transactions(root)
            self.assertTrue(transaction.transaction_root.exists())

    def test_recovery_rejects_symlinked_transaction_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / ".ghostgui-transactions").symlink_to(
                outside,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ProjectFormatError, "symbolic link"):
                ProjectSaveTransaction(root)
            with self.assertRaisesRegex(ProjectFormatError, "symbolic link"):
                recover_project_transactions(root)

            self.assertTrue(outside.exists())

    def test_prejournal_crash_directory_is_safe_to_remove(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transaction = ProjectSaveTransaction(root)
            transaction.stage_json(root / "data" / "value.json", {"value": 1})

            recovered = recover_project_transactions(root)

            self.assertEqual(recovered, [])
            self.assertFalse(transaction.transaction_root.exists())

    def test_manual_and_autosave_bundles_round_trip_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recovery.ghostgui"
            project = GhostGUIProject.create(root, "Recovery", "g1")
            saved = Trajectory()
            saved.add_frame(TargetFrame(time=0.0, x=0.1))
            autosaved = Trajectory()
            autosaved.add_frame(TargetFrame(time=0.0, x=0.9))
            timeline = FakeTimeline(times=(0.0, 0.25))
            project.save_bundle(saved, timeline, {"source": "manual"})
            project.write_autosave(
                autosaved,
                timeline,
                {"source": "autosave"},
                "g1",
            )

            reopened = GhostGUIProject.open(root)

            self.assertEqual(
                reopened.read_trajectory_dict()["tracks"]["pelvis"][0]["x"],
                0.1,
            )
            self.assertEqual(
                reopened.read_trajectory_dict(autosave=True)["tracks"][
                    "pelvis"
                ][0]["x"],
                0.9,
            )
            self.assertEqual(reopened.read_workspace(), {"source": "manual"})
            self.assertEqual(
                reopened.read_workspace(autosave=True),
                {"source": "autosave"},
            )
            self.assertTrue(reopened.autosave_exists())
            reopened.clear_autosave()
            self.assertFalse(reopened.autosave_exists())


class ResourceLocationTests(unittest.TestCase):
    def test_checkout_resources_are_found_independent_of_cwd(self):
        old_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                model = resource_path("models/g1_29dof.xml", required=True)
                guide = resource_path("docs/user_guide.md", required=True)
        finally:
            os.chdir(old_cwd)

        self.assertTrue(model.is_file())
        self.assertTrue(guide.is_file())

    def test_resource_override_is_validated_and_traversal_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            with patch.dict(
                os.environ,
                {"GHOSTGUI_RESOURCE_DIR": str(root)},
            ):
                self.assertEqual(bundled_resource_root(), root.resolve())
                with self.assertRaises(ValueError):
                    resource_path("../outside")


if __name__ == "__main__":
    unittest.main()
