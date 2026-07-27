import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QWidget

from gui.app_sidebars import AppSidebar
from gui.theme import (
    DARK_THEME,
    LIGHT_THEME,
    application_stylesheet,
    current_theme,
    ensure_application_theme,
    section_stylesheet,
    tutorial_card_stylesheet,
)
from gui.widgets.compact import compact_spinbox


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original_palette = QPalette(self.app.palette())
        self.original_font = QFont(self.app.font())

    def tearDown(self):
        self.app.setPalette(self.original_palette)
        self.app.setFont(self.original_font)

    def set_window_lightness(self, color):
        palette = QPalette(self.app.palette())
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.app.setPalette(palette)

    def test_section_header_states_pin_readable_foreground_colors(self):
        self.set_window_lightness("#f8fafc")
        light_style = section_stylesheet()

        self.assertIs(current_theme(), LIGHT_THEME)
        self.assertIn(f"background: {LIGHT_THEME.section_header_bg};", light_style)
        self.assertIn(f"color: {LIGHT_THEME.section_header_text};", light_style)
        self.assertIn(
            f"background: {LIGHT_THEME.section_header_hover_bg};", light_style
        )
        self.assertIn(
            f"color: {LIGHT_THEME.section_header_hover_text};", light_style
        )

        self.set_window_lightness("#0f172a")
        dark_style = section_stylesheet()

        self.assertIs(current_theme(), DARK_THEME)
        self.assertIn(f"background: {DARK_THEME.section_header_bg};", dark_style)
        self.assertIn(f"color: {DARK_THEME.section_header_text};", dark_style)
        self.assertIn(
            f"background: {DARK_THEME.section_header_hover_bg};", dark_style
        )
        self.assertIn(f"color: {DARK_THEME.section_header_hover_text};", dark_style)

    def test_common_controls_receive_dark_mode_text_and_background(self):
        self.set_window_lightness("#0f172a")
        style = application_stylesheet()

        self.assertIn(f"color: {DARK_THEME.text};", style)
        self.assertIn(f"background: {DARK_THEME.sidebar_bg};", style)
        self.assertIn(f"background: {DARK_THEME.elevated_bg};", style)
        self.assertIn(f"border: 1px solid {DARK_THEME.border};", style)
        self.assertIn(f"selection-background-color: {DARK_THEME.accent};", style)
        self.assertNotIn("QPushButton, QToolButton", style)

    def test_theme_stylesheet_does_not_override_fonts(self):
        style = application_stylesheet()

        self.assertNotIn("font-", style)
        self.assertNotIn("font:", style)
        self.assertNotIn("line-height", style)

    def test_tabs_spinboxes_and_scroller_arrows_are_explicitly_themed(self):
        self.set_window_lightness("#0f172a")
        style = application_stylesheet()

        self.assertIn("QTabWidget#ikEditorTabs", style)
        self.assertIn("QScrollArea#ikEditorScroll", style)
        self.assertIn("QWidget#ikEditorTabContent", style)
        self.assertIn("QSpinBox::up-arrow", style)
        self.assertIn("QSpinBox::down-arrow", style)
        self.assertIn("padding-right: 14px;", style)
        self.assertIn("width: 14px;", style)
        self.assertIn("QTabBar QToolButton::left-arrow", style)
        self.assertIn("QTabBar QToolButton::right-arrow", style)
        self.assertIn("chevron-up-dark.svg", style)
        self.assertIn("chevron-down-dark.svg", style)
        self.assertIn("chevron-left-dark.svg", style)
        self.assertIn("chevron-right-dark.svg", style)
        self.assertNotIn("border-bottom: 5px solid", style)
        self.assertNotIn("border-top: 5px solid", style)

    def test_compact_spinbox_leaves_room_for_themed_arrows(self):
        spinbox = QDoubleSpinBox()
        try:
            compact_spinbox(spinbox)
            self.assertGreaterEqual(spinbox.maximumWidth(), 78)
        finally:
            spinbox.close()

    def test_sidebar_shell_widgets_are_theme_selectable(self):
        sidebar = AppSidebar()
        try:
            content = QWidget()
            section = sidebar.add_section("Example", content)

            self.assertEqual(sidebar.objectName(), "AppSidebar")
            self.assertEqual(sidebar.scroll.objectName(), "appSidebarScroll")
            self.assertEqual(
                sidebar.scroll.viewport().objectName(), "appSidebarViewport"
            )
            self.assertEqual(sidebar.body.objectName(), "appSidebarBody")
            self.assertEqual(section.content.objectName(), "sectionContent")
        finally:
            sidebar.close()

    def test_tutorial_card_uses_theme_tokens(self):
        self.set_window_lightness("#0f172a")
        style = tutorial_card_stylesheet()

        self.assertIn(f"background-color: {DARK_THEME.overlay_card_bg};", style)
        self.assertIn(f"color: {DARK_THEME.text};", style)
        self.assertIn(f"color: {DARK_THEME.disabled_text};", style)

    def test_theme_synchronizer_reacts_to_palette_changes(self):
        synchronizer = ensure_application_theme(self.app)
        original_font = QFont(self.app.font())

        self.set_window_lightness("#f8fafc")
        QApplication.processEvents()
        synchronizer.apply()
        self.assertIn(f"background: {LIGHT_THEME.window_bg};", self.app.styleSheet())
        self.assertIn(f"color: {LIGHT_THEME.text};", self.app.styleSheet())
        self.assertIn(
            f"background: {LIGHT_THEME.section_header_bg};", self.app.styleSheet()
        )

        self.set_window_lightness("#0f172a")
        QApplication.processEvents()
        self.assertIn(f"background: {DARK_THEME.window_bg};", self.app.styleSheet())
        self.assertIn(f"color: {DARK_THEME.text};", self.app.styleSheet())
        self.assertIn(
            f"background: {DARK_THEME.section_header_bg};", self.app.styleSheet()
        )
        self.assertEqual(self.app.font(), original_font)


if __name__ == "__main__":
    unittest.main()
