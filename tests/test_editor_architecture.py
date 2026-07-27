"""Qt-free contracts for document, controller, command, and session ownership."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from application.editor_commands import (
    AddKeyframe,
    ClearTrajectory,
    DeleteKeyframe,
    DeleteTimeslice,
    ReplaceTrajectoryFrames,
    UpdateKeyframe,
    UpsertKeyframe,
)
from application.editor_controller import EditorController
from application.editor_events import (
    DocumentChanged,
    DocumentDirtyChanged,
    EditorEvent,
    EditorEventBus,
)
from application.editor_session import EditorSession, EditorSessionState
from application.project_document import ProjectDocument
from core.trajectory import TargetFrame


def frame(time=0.0, x=0.0, name="pelvis"):
    return TargetFrame(time=time, frame_name=name, x=x)


class ProjectDocumentTests(unittest.TestCase):
    def test_snapshot_round_trip_is_independent(self):
        document = ProjectDocument("g1")
        controller = EditorController(document)
        controller.execute(AddKeyframe(frame(x=0.25)))

        restored = ProjectDocument.from_snapshot(document.snapshot())
        restored.trajectory.frames[0].x = 0.75

        self.assertEqual(document.trajectory.frames[0].x, 0.25)
        self.assertEqual(restored.active_index, 0)
        self.assertEqual(restored.revision, 1)
        self.assertTrue(restored.dirty)

    def test_document_validates_time_contract(self):
        with self.assertRaises(ValueError):
            ProjectDocument("g1", timeline_duration=0.0)
        with self.assertRaises(ValueError):
            ProjectDocument("g1", current_time=-0.1)


class EditorControllerTests(unittest.TestCase):
    def setUp(self):
        self.document = ProjectDocument("g1")
        self.events = EditorEventBus()
        self.controller = EditorController(self.document, self.events)
        self.changed = []
        self.dirty = []
        self.events.subscribe(DocumentChanged, self.changed.append)
        self.events.subscribe(DocumentDirtyChanged, self.dirty.append)

    def test_commands_mutate_only_through_document_and_emit_typed_events(self):
        result = self.controller.execute(AddKeyframe(frame(x=0.1)))
        self.controller.execute(UpsertKeyframe(frame(x=0.2)))
        self.controller.execute(UpdateKeyframe(0, frame(x=0.3)))

        self.assertTrue(result.changed)
        self.assertEqual(len(self.document.trajectory.frames), 1)
        self.assertEqual(self.document.trajectory.frames[0].x, 0.3)
        self.assertEqual(self.document.revision, 3)
        self.assertEqual(
            [event.operation for event in self.changed],
            ["add_keyframe", "upsert_keyframe", "update_keyframe"],
        )
        self.assertEqual([event.dirty for event in self.dirty], [True])

    def test_delete_replace_and_clear_report_affected_counts(self):
        replacement = ReplaceTrajectoryFrames(
            (frame(0.0), frame(0.5, name="left_hand"))
        )
        self.assertEqual(
            self.controller.execute(replacement).affected_count,
            2,
        )
        self.assertEqual(
            self.controller.execute(DeleteTimeslice(0.5)).affected_count,
            1,
        )
        self.assertEqual(
            self.controller.execute(DeleteKeyframe(0)).affected_count,
            1,
        )
        self.assertFalse(self.controller.execute(ClearTrajectory()).changed)

    def test_mark_saved_and_current_time_have_separate_events(self):
        self.controller.execute(AddKeyframe(frame()))
        self.assertTrue(self.controller.mark_saved())
        self.assertFalse(self.document.dirty)
        self.assertTrue(self.controller.set_current_time(0.5))
        self.assertEqual(self.document.current_time, 0.5)
        self.assertFalse(self.document.dirty)
        self.assertEqual([event.dirty for event in self.dirty], [True, False])

    def test_faulty_subscriber_does_not_rollback_successful_command(self):
        def fail(_event):
            raise RuntimeError("presentation failed")

        self.events.subscribe(EditorEvent, fail)

        self.controller.execute(AddKeyframe(frame()))

        self.assertEqual(len(self.document.trajectory.frames), 1)
        self.assertEqual(len(self.controller.dispatch_failures), 3)


class EditorEventBusTests(unittest.TestCase):
    def test_subscription_is_idempotently_removable(self):
        events = EditorEventBus()
        received = []
        subscription = events.subscribe(DocumentChanged, received.append)
        event = DocumentChanged("doc", "g1", 1, "test", 0)
        events.publish(event)
        subscription.unsubscribe()
        subscription.unsubscribe()
        events.publish(event)
        self.assertEqual(received, [event])

    def test_rejects_untyped_events(self):
        events = EditorEventBus()
        with self.assertRaises(TypeError):
            events.publish(object())


class EditorSessionTests(unittest.TestCase):
    @dataclass
    class Viewer:
        state_timeline: object

    def test_session_owns_document_and_lifecycle(self):
        timeline = object()
        document = ProjectDocument("g1")
        session = EditorSession(
            "g1",
            adapter=object(),
            backend=object(),
            reference=object(),
            viewer_3d=self.Viewer(timeline),
            document=document,
        )

        self.assertIs(document.qpos_timeline, timeline)
        session.activate()
        self.assertEqual(session.state, EditorSessionState.ACTIVE)
        session.deactivate()
        self.assertEqual(session.state, EditorSessionState.INACTIVE)
        session.close()
        with self.assertRaises(RuntimeError):
            session.activate()

    def test_session_rejects_mismatched_document(self):
        with self.assertRaises(ValueError):
            EditorSession(
                "go2",
                adapter=object(),
                backend=object(),
                reference=object(),
                viewer_3d=self.Viewer(None),
                document=ProjectDocument("g1"),
            )


if __name__ == "__main__":
    unittest.main()
