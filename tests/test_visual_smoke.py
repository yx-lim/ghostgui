"""Software-OpenGL visual smoke and stable layout regression checks."""

from __future__ import annotations

import os
import unittest

from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui.main_window import RobotGuiMainWindow


VISUAL_TESTS_ENABLED = os.environ.get("GHOSTGUI_VISUAL_TESTS") == "1"


@unittest.skipUnless(
    VISUAL_TESTS_ENABLED,
    "set GHOSTGUI_VISUAL_TESTS=1 and run under Xvfb",
)
class MainWindowVisualSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_renders_and_matches_structural_baseline(self):
        self.assertNotEqual(QGuiApplication.platformName(), "offscreen")
        window = RobotGuiMainWindow("g1")
        try:
            window.resize(1200, 800)
            window.show()
            for _ in range(100):
                self.app.processEvents()
                if window.viewer_3d.canvas.isValid():
                    break
                QTest.qWait(20)

            self.assertTrue(window.isVisible())
            self.assertTrue(window.viewer_3d.canvas.isValid())
            self.assertEqual(
                [window.viewer_tabs.tabText(index) for index in range(2)],
                ["3D Pose", "Simulation"],
            )
            self.assertEqual(
                [
                    section.title
                    for section in window.left_sidebar_content.sections
                ],
                ["Target", "Editing Mode", "Planning"],
            )
            self.assertEqual(
                [
                    section.title
                    for section in window.right_sidebar_content.sections
                ],
                ["Status", "IK / Constraints"],
            )
            self.assertEqual(
                [
                    action.text()
                    for action in (
                        window.preview_action,
                        window.slice_action,
                        window.generate_action,
                        window.playback_action,
                        window.reset_action,
                        window.clear_action,
                        window.move_action,
                        window.rotate_action,
                    )
                ],
                [
                    "Preview Path",
                    "Commit Keyframe",
                    "Generate",
                    "Play",
                    "Reset",
                    "Clear",
                    "Move",
                    "Rotate",
                ],
            )

            image = window.grab().toImage()
            self.assertFalse(image.isNull())
            self.assertGreaterEqual(image.width(), 1100)
            self.assertGreaterEqual(image.height(), 700)
            colors = {
                image.pixelColor(x, y).rgba()
                for x in range(0, image.width(), 30)
                for y in range(0, image.height(), 30)
            }
            self.assertGreater(
                len(colors),
                12,
                "captured window is visually blank or failed to composite",
            )
            self.assertEqual(image.pixelColor(0, 0).alpha(), 255)
        finally:
            window.current_project = None
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
