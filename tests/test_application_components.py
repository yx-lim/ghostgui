"""Focused tests for extracted application-level UI support components."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from application.history import HistoryStack
from application.paths import writable_data_root
from application.playback import PlaybackClock
from gui.viewers.camera import OrbitCamera


class HistoryStackTests(unittest.TestCase):
    def test_bounded_record_undo_redo_and_branching(self):
        history = HistoryStack[str](max_depth=2)
        history.set_baseline("zero")
        history.record("one", before="zero", after="one")
        history.record("two", before="one", after="two")
        history.record("three", before="two", after="three")

        self.assertEqual(
            [entry.description for entry in history.undo_entries],
            ["two", "three"],
        )
        transition = history.undo("three")
        self.assertEqual(transition.target, "two")
        self.assertEqual(transition.direction, "undo")
        transition = history.redo("two")
        self.assertEqual(transition.target, "three")

        history.undo("three")
        history.record("branch", before="two", after="branch")
        self.assertEqual(history.redo_entries, [])
        self.assertEqual(history.baseline, "branch")

    def test_empty_and_invalid_history_are_explicit(self):
        with self.assertRaises(ValueError):
            HistoryStack(max_depth=0)
        history = HistoryStack()
        self.assertIsNone(history.undo("current"))
        self.assertIsNone(history.redo("current"))


class PlaybackClockTests(unittest.TestCase):
    def test_elapsed_uses_monotonic_clock_and_supplied_values(self):
        ticks = iter((10.0, 10.25))
        clock = PlaybackClock(now=lambda: next(ticks))
        clock.start()
        self.assertAlmostEqual(clock.elapsed(0.033), 0.25)
        self.assertAlmostEqual(clock.elapsed(0.033, supplied=-1.0), 0.0)
        clock.stop()
        self.assertIsNone(clock.last_tick)

    def test_advance_scales_and_wraps_elapsed_time(self):
        self.assertAlmostEqual(
            PlaybackClock.advance(0.2, 0.2, 0.8, 0.15, speed=2.0),
            0.5,
        )
        self.assertAlmostEqual(
            PlaybackClock.advance(0.7, 0.2, 0.8, 0.2),
            0.3,
        )

    def test_advance_rejects_invalid_timeline_contracts(self):
        with self.assertRaises(ValueError):
            PlaybackClock.advance(0.0, 1.0, 1.0, 0.1)
        with self.assertRaises(ValueError):
            PlaybackClock.advance(0.0, 0.0, 1.0, 0.1, speed=0.0)
        with self.assertRaises(ValueError):
            PlaybackClock.advance(0.0, 0.0, 1.0, float("nan"))


class OrbitCameraTests(unittest.TestCase):
    def test_orbit_pan_zoom_keep_navigation_constraints(self):
        camera = OrbitCamera()
        start_eye = camera.eye().copy()

        camera.orbit(10.0, 1000.0)
        camera.pan(20.0, -10.0, viewport_height=500)
        camera.zoom(-100.0)

        self.assertEqual(camera.pitch, 85.0)
        self.assertEqual(camera.distance, 0.5)
        self.assertFalse((camera.eye() == start_eye).all())
        right, up, forward = camera.basis()
        self.assertAlmostEqual(float(right @ up), 0.0, places=7)
        self.assertAlmostEqual(float(up @ forward), 0.0, places=7)


class RuntimePathTests(unittest.TestCase):
    def test_explicit_user_data_root_overrides_checkout_defaults(self):
        with patch.dict(
            "os.environ",
            {"GHOSTGUI_USER_DATA_DIR": "/tmp/ghostgui-test-data"},
        ):
            self.assertEqual(
                str(writable_data_root()),
                "/tmp/ghostgui-test-data",
            )


if __name__ == "__main__":
    unittest.main()
