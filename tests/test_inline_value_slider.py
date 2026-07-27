import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui.widgets.inline_value_slider import InlineValueSlider
from gui.widgets.joint_controls import IKInfluenceControl, JointControl
from gui.theme import DARK_THEME, LIGHT_THEME


class InlineValueSliderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_slider(self, **kwargs):
        slider = InlineValueSlider(
            kwargs.pop("minimum", -1.0),
            kwargs.pop("maximum", 1.0),
            kwargs.pop("value", 0.0),
            single_step=kwargs.pop("single_step", 0.1),
            decimals=kwargs.pop("decimals", 2),
            suffix=kwargs.pop("suffix", ""),
            display_scale=kwargs.pop("display_scale", 1.0),
        )
        slider.resize(240, 28)
        slider.show()
        self.app.processEvents()
        self.addCleanup(slider.close)
        return slider

    def test_left_and_right_click_step_without_jumping_to_pointer(self):
        slider = self.make_slider()

        QTest.mouseClick(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(slider.width() - 6, slider.height() // 2),
        )
        self.assertAlmostEqual(slider.logical_value(), 0.1, places=3)

        QTest.mouseClick(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(6, slider.height() // 2),
        )
        self.assertAlmostEqual(slider.logical_value(), 0.0, places=3)

    def test_keyboard_step_home_end_and_page_keys_use_logical_units(self):
        slider = self.make_slider()
        slider.setFocus()

        QTest.keyClick(slider, Qt.Key.Key_Right)
        self.assertAlmostEqual(slider.logical_value(), 0.1, places=3)
        QTest.keyClick(slider, Qt.Key.Key_PageUp)
        self.assertAlmostEqual(slider.logical_value(), 1.0, places=3)
        QTest.keyClick(slider, Qt.Key.Key_Home)
        self.assertAlmostEqual(slider.logical_value(), -1.0, places=3)
        QTest.keyClick(slider, Qt.Key.Key_End)
        self.assertAlmostEqual(slider.logical_value(), 1.0, places=3)

    def test_f2_direct_entry_commits_and_escape_cancels(self):
        slider = self.make_slider()
        self.assertFalse(slider.editor.isVisible())

        slider.setFocus()
        QTest.keyClick(slider, Qt.Key.Key_F2)
        self.assertTrue(slider.editor.isVisible())
        slider.editor.setText("0.75")
        QTest.keyClick(slider.editor, Qt.Key.Key_Return)

        self.assertFalse(slider.editor.isVisible())
        self.assertAlmostEqual(slider.logical_value(), 0.75, places=3)

        QTest.mouseDClick(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(slider.width() // 2, slider.height() // 2),
        )
        self.assertTrue(slider.editor.isVisible())
        slider.editor.setText("-0.5")
        QTest.keyClick(slider.editor, Qt.Key.Key_Escape)

        self.assertFalse(slider.editor.isVisible())
        self.assertAlmostEqual(slider.logical_value(), 0.75, places=3)

    def test_invalid_direct_entry_stays_open_until_cancelled(self):
        slider = self.make_slider(value=0.25)
        slider.begin_inline_edit()
        slider.editor.setText("not a number")

        QTest.keyClick(slider.editor, Qt.Key.Key_Return)

        self.assertTrue(slider.editor.isVisible())
        self.assertAlmostEqual(slider.logical_value(), 0.25, places=3)
        QTest.keyClick(slider.editor, Qt.Key.Key_Escape)
        self.assertFalse(slider.editor.isVisible())

    def test_dragging_updates_live_instead_of_applying_click_step(self):
        slider = self.make_slider()
        emitted = []
        slider.logical_value_changed.connect(emitted.append)
        y = slider.height() / 2.0
        start = QPointF(8.0, y)
        finish = QPointF(slider.width() - 8.0, y)

        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            start,
            slider.mapToGlobal(start.toPoint()),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            finish,
            slider.mapToGlobal(finish.toPoint()),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            finish,
            slider.mapToGlobal(finish.toPoint()),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(slider, press)
        QApplication.sendEvent(slider, move)
        QApplication.sendEvent(slider, release)

        self.assertGreater(slider.logical_value(), 0.8)
        self.assertGreater(len(emitted), 0)

    def test_degree_display_and_entry_keep_radians_internally(self):
        slider = self.make_slider(
            minimum=-math.pi,
            maximum=math.pi,
            single_step=math.pi / 180.0,
            decimals=1,
            suffix="°",
            display_scale=180.0 / math.pi,
        )
        slider.set_logical_value(math.pi / 2.0)
        self.assertEqual(slider.format_value(), "90.0°")

        slider.begin_inline_edit()
        slider.editor.setText("-45")
        QTest.keyClick(slider.editor, Qt.Key.Key_Return)

        self.assertAlmostEqual(slider.logical_value(), -math.pi / 4.0, places=3)

    def test_percent_display_and_clamping(self):
        slider = self.make_slider(
            minimum=0.0,
            maximum=1.0,
            value=0.35,
            single_step=0.01,
            decimals=0,
            suffix="%",
            display_scale=100.0,
        )
        self.assertEqual(slider.format_value(), "35%")
        slider.set_logical_value(2.0)
        self.assertAlmostEqual(slider.logical_value(), 1.0)

    def test_custom_background_tracks_active_theme(self):
        slider = self.make_slider(value=-1.0)
        sample = QPoint(slider.width() - 12, slider.height() // 2)

        with patch(
            "gui.widgets.inline_value_slider.current_theme",
            return_value=LIGHT_THEME,
        ):
            slider.update()
            self.app.processEvents()
            light_background = slider.grab().toImage().pixelColor(sample)

        with patch(
            "gui.widgets.inline_value_slider.current_theme",
            return_value=DARK_THEME,
        ):
            slider.update()
            self.app.processEvents()
            dark_background = slider.grab().toImage().pixelColor(sample)

        self.assertEqual(light_background, QColor(LIGHT_THEME.panel_bg))
        self.assertEqual(dark_background, QColor(DARK_THEME.panel_bg))
        self.assertNotEqual(light_background, dark_background)

    def test_joint_and_weight_controls_emit_logical_values(self):
        joint = JointControl("joint", (-math.pi, math.pi), 0.0)
        influence = IKInfluenceControl("joint", 1.0)
        self.addCleanup(joint.close)
        self.addCleanup(influence.close)
        joint_values = []
        influence_values = []
        joint.value_changed.connect(lambda _name, value: joint_values.append(value))
        influence.value_changed.connect(
            lambda _name, value: influence_values.append(value)
        )

        joint.slider.step_logical_value(1)
        influence.slider.step_logical_value(1)

        self.assertAlmostEqual(joint_values[-1], math.pi / 180.0, places=3)
        self.assertAlmostEqual(influence_values[-1], 1.01, places=3)


if __name__ == "__main__":
    unittest.main()
