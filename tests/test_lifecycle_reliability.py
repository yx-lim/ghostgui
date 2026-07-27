"""Cancellation, rendering, resource, and session teardown contracts."""

from __future__ import annotations

from threading import Event
from time import monotonic
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from application.background_jobs import (
    JobState,
    SerializedBackgroundJobs,
)
from application.editor_session import EditorSession, EditorSessionState
from application.project_document import ProjectDocument
from gui.render_scheduler import RenderRequestCoalescer
from gui.viewers.robot_canvas_3d import RobotCanvas3D


def process_until(predicate, timeout=2.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
    return predicate()


class BackgroundJobReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_jobs_are_serial_and_callbacks_arrive_on_gui_thread(self):
        jobs = SerializedBackgroundJobs()
        work_order = []
        results = []
        first = jobs.submit_handle(
            "first",
            lambda: work_order.append("first") or 1,
            lambda result: results.append(result),
            self.fail,
        )
        second = jobs.submit_handle(
            "second",
            lambda: work_order.append("second") or 2,
            lambda result: results.append(result),
            self.fail,
        )

        self.assertTrue(process_until(lambda: not jobs.is_busy()))
        self.assertEqual(work_order, ["first", "second"])
        self.assertEqual(results, [1, 2])
        self.assertEqual(first.state, JobState.DELIVERED)
        self.assertEqual(second.state, JobState.DELIVERED)
        self.assertTrue(jobs.shutdown())

    def test_cooperative_cancellation_has_separate_callback(self):
        jobs = SerializedBackgroundJobs()
        started = Event()
        release = Event()
        cancelled = []
        failed = []

        def work(token):
            started.set()
            release.wait(1.0)
            token.raise_if_cancelled()
            return "late"

        handle = jobs.submit_cancellable(
            "cancellable",
            work,
            lambda _result: self.fail("cancelled job succeeded"),
            failed.append,
            cancelled=lambda: cancelled.append(True),
        )
        self.assertTrue(started.wait(1.0))
        self.assertTrue(handle.cancel())
        release.set()

        self.assertTrue(process_until(lambda: not jobs.is_busy()))
        self.assertEqual(cancelled, [True])
        self.assertEqual(failed, [])
        self.assertEqual(handle.state, JobState.CANCELLED)
        self.assertTrue(jobs.shutdown())

    def test_shutdown_suppresses_late_callbacks_and_can_be_rejoined(self):
        jobs = SerializedBackgroundJobs()
        started = Event()
        release = Event()
        callbacks = []

        def work():
            started.set()
            release.wait(1.0)
            return "done"

        jobs.submit("slow", work, callbacks.append, callbacks.append)
        self.assertTrue(started.wait(1.0))
        self.assertFalse(jobs.shutdown(timeout=0.001))
        release.set()
        self.assertTrue(jobs.shutdown(timeout=1.0))
        QApplication.processEvents()
        self.assertEqual(callbacks, [])
        self.assertFalse(jobs.is_busy())


class RenderLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_render_requests_are_coalesced_and_shutdown_drops_pending(self):
        renders = []
        scheduler = RenderRequestCoalescer(lambda: renders.append("render"))
        scheduler.request()
        scheduler.request()
        scheduler.request()
        self.assertTrue(scheduler.pending)
        QApplication.processEvents()
        self.assertEqual(renders, ["render"])

        scheduler.request()
        scheduler.shutdown()
        QApplication.processEvents()
        self.assertEqual(renders, ["render"])
        self.assertFalse(scheduler.request())

    def test_canvas_deletes_unique_context_resources_once(self):
        canvas = RobotCanvas3D()
        canvas._geom_lists = [3, 4, 3, None]
        canvas._mesh_display_lists = {0: 4, 1: 5}
        canvas._quadric = object()
        with (
            patch(
                "gui.viewers.robot_canvas_3d.GL.glDeleteLists"
            ) as delete_lists,
            patch(
                "gui.viewers.robot_canvas_3d.GLU.gluDeleteQuadric"
            ) as delete_quadric,
        ):
            canvas.cleanup_gl_resources(context_current=True)
            canvas.cleanup_gl_resources(context_current=True)

        self.assertEqual(
            {call.args for call in delete_lists.call_args_list},
            {(3, 1), (4, 1), (5, 1)},
        )
        self.assertEqual(delete_lists.call_count, 3)
        delete_quadric.assert_called_once()
        canvas.shutdown()


class SessionCleanupTests(unittest.TestCase):
    class Viewer:
        state_timeline = object()

        def __init__(self):
            self.shutdown_count = 0

        def shutdown(self):
            self.shutdown_count += 1

    def test_session_close_is_idempotent_and_detaches_runtime_state(self):
        viewer = self.Viewer()
        document = ProjectDocument("g1")
        session = EditorSession(
            "g1",
            adapter=object(),
            backend=object(),
            reference=object(),
            viewer_3d=viewer,
            document=document,
        )

        session.close()
        session.close()

        self.assertEqual(session.state, EditorSessionState.CLOSED)
        self.assertEqual(viewer.shutdown_count, 1)
        self.assertIsNone(document.qpos_timeline)


if __name__ == "__main__":
    unittest.main()
