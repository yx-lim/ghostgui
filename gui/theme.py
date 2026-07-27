"""Palette-aware style tokens for GhostGUI widgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtGui import QFont, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from core.resources import resource_path

@dataclass(frozen=True)
class Theme:
    window_bg: str
    sidebar_bg: str
    panel_bg: str
    panel_hover_bg: str
    elevated_bg: str
    text: str
    muted_text: str
    disabled_text: str
    border: str
    focus_border: str
    accent: str
    accent_text: str
    section_bg: str
    section_header_bg: str
    section_header_active_bg: str
    section_header_hover_bg: str
    section_header_text: str
    section_header_hover_text: str
    overlay_scrim: str
    overlay_card_bg: str
    overlay_progress_bg: str
    quick_panel_bg: str
    quick_panel_border: str
    icon_variant: str


LIGHT_THEME = Theme(
    window_bg="#f8fafc",
    sidebar_bg="#f1f5f9",
    panel_bg="#ffffff",
    panel_hover_bg="#e8eef6",
    elevated_bg="#f1f5f9",
    text="#111827",
    muted_text="#475569",
    disabled_text="#94a3b8",
    border="#cbd5e1",
    focus_border="#2563eb",
    accent="#2563eb",
    accent_text="#ffffff",
    section_bg="#ffffff",
    section_header_bg="#eef2f7",
    section_header_active_bg="#e0e7f0",
    section_header_hover_bg="#dbeafe",
    section_header_text="#111827",
    section_header_hover_text="#0f172a",
    overlay_scrim="rgba(15, 23, 42, 132)",
    overlay_card_bg="#f8fafc",
    overlay_progress_bg="#e2e8f0",
    quick_panel_bg="rgba(248, 250, 252, 232)",
    quick_panel_border="rgba(71, 85, 105, 135)",
    icon_variant="light",
)

DARK_THEME = Theme(
    window_bg="#0f172a",
    sidebar_bg="#0b1220",
    panel_bg="#111827",
    panel_hover_bg="#263244",
    elevated_bg="#1f2937",
    text="#f8fafc",
    muted_text="#cbd5e1",
    disabled_text="#64748b",
    border="#334155",
    focus_border="#60a5fa",
    accent="#3b82f6",
    accent_text="#f8fafc",
    section_bg="#111827",
    section_header_bg="#1f2937",
    section_header_active_bg="#253246",
    section_header_hover_bg="#334155",
    section_header_text="#f8fafc",
    section_header_hover_text="#ffffff",
    overlay_scrim="rgba(2, 6, 23, 166)",
    overlay_card_bg="#111827",
    overlay_progress_bg="#0f172a",
    quick_panel_bg="rgba(17, 24, 39, 232)",
    quick_panel_border="rgba(148, 163, 184, 125)",
    icon_variant="dark",
)

THEME_ASSET_DIR = resource_path("gui/assets/theme")


def _icon_url(theme: Theme, name: str) -> str:
    return (THEME_ASSET_DIR / f"{name}-{theme.icon_variant}.svg").as_posix()


def theme_icon(name: str, widget: QWidget | None = None) -> QIcon:
    """Return the light/dark SVG icon matching the active application palette."""
    return QIcon(_icon_url(current_theme(widget), name))


def _palette_for(widget: QWidget | None = None) -> QPalette | None:
    if widget is not None:
        return widget.palette()
    app = QApplication.instance()
    return app.palette() if app is not None else None


def is_dark_mode(widget: QWidget | None = None) -> bool:
    app = QApplication.instance()
    if app is not None:
        color_scheme = app.styleHints().colorScheme()
        if color_scheme == Qt.ColorScheme.Dark:
            return True
        if color_scheme == Qt.ColorScheme.Light:
            return False

    palette = _palette_for(widget)
    if palette is None:
        return False
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def current_theme(widget: QWidget | None = None) -> Theme:
    return DARK_THEME if is_dark_mode(widget) else LIGHT_THEME


def application_stylesheet(widget: QWidget | None = None) -> str:
    theme = current_theme(widget)
    status_success = "#4ade80" if theme.icon_variant == "dark" else "#15803d"
    status_warning = "#fbbf24" if theme.icon_variant == "dark" else "#b45309"
    status_error = "#f87171" if theme.icon_variant == "dark" else "#b91c1c"
    return (
        f"""
        QMainWindow, QDialog {{
            background: {theme.window_bg};
            color: {theme.text};
        }}
        QWidget#AppSidebar, QScrollArea#appSidebarScroll,
        QWidget#appSidebarViewport, QWidget#appSidebarBody {{
            background: {theme.sidebar_bg};
            color: {theme.text};
        }}
        QSplitter::handle:horizontal {{
            background: {theme.window_bg};
            border-left: 1px solid {theme.border};
            border-right: 1px solid {theme.border};
        }}
        QToolButton#leftSidebarCollapseButton,
        QToolButton#rightSidebarCollapseButton {{
            background: {theme.elevated_bg};
            border: 1px solid {theme.border};
            border-radius: 3px;
            padding: 0;
        }}
        QToolButton#leftSidebarCollapseButton:hover,
        QToolButton#rightSidebarCollapseButton:hover {{
            background: {theme.panel_hover_bg};
            border-color: {theme.focus_border};
        }}
        QToolButton#leftSidebarCollapseButton::left-arrow,
        QToolButton#rightSidebarCollapseButton::left-arrow {{
            image: url("{_icon_url(theme, "chevron-left")}");
            width: 9px;
            height: 9px;
        }}
        QToolButton#leftSidebarCollapseButton::right-arrow,
        QToolButton#rightSidebarCollapseButton::right-arrow {{
            image: url("{_icon_url(theme, "chevron-right")}");
            width: 9px;
            height: 9px;
        }}
        QLabel, QCheckBox, QRadioButton, QGroupBox {{
            color: {theme.text};
        }}
        QLabel#statusSeverityIcon {{
            color: {theme.accent};
        }}
        QLabel#statusSeverityIcon[severity="success"] {{
            color: {status_success};
        }}
        QLabel#statusSeverityIcon[severity="warning"] {{
            color: {status_warning};
        }}
        QLabel#statusSeverityIcon[severity="error"] {{
            color: {status_error};
        }}
        QLabel#statusEventTitle {{
            color: {theme.text};
        }}
        QLabel#statusEventMessage {{
            color: {theme.muted_text};
        }}
        QGroupBox {{
            background: {theme.panel_bg};
            border: 1px solid {theme.border};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: {theme.muted_text};
        }}
        QPushButton {{
            color: {theme.text};
            background: {theme.elevated_bg};
            border: 1px solid {theme.border};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        QPushButton:hover {{
            color: {theme.text};
            background: {theme.panel_hover_bg};
            border-color: {theme.focus_border};
        }}
        QPushButton:pressed {{
            background: {theme.section_header_active_bg};
        }}
        QPushButton:disabled {{
            color: {theme.disabled_text};
            background: {theme.panel_bg};
            border-color: {theme.border};
        }}
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QTextBrowser,
        QListWidget {{
            color: {theme.text};
            background: {theme.panel_bg};
            border: 1px solid {theme.border};
            border-radius: 4px;
            selection-color: {theme.accent_text};
            selection-background-color: {theme.accent};
        }}
        QSpinBox, QDoubleSpinBox {{
            padding-right: 14px;
        }}
        QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover,
        QTextEdit:hover, QTextBrowser:hover, QListWidget:hover {{
            border-color: {theme.focus_border};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            width: 14px;
            background: {theme.elevated_bg};
            border-left: 1px solid {theme.border};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-position: top right;
            border-bottom: 1px solid {theme.border};
            border-top-right-radius: 4px;
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-position: bottom right;
            border-bottom-right-radius: 4px;
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {theme.panel_hover_bg};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url("{_icon_url(theme, "chevron-up")}");
            width: 9px;
            height: 9px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url("{_icon_url(theme, "chevron-down")}");
            width: 9px;
            height: 9px;
        }}
        QTabWidget::pane {{
            background: {theme.panel_bg};
            border: 1px solid {theme.border};
            border-radius: 4px;
        }}
        QTabBar {{
            background: {theme.panel_bg};
            color: {theme.text};
        }}
        QTabWidget#ikEditorTabs, QTabWidget#ikEditorTabs::pane,
        QScrollArea#ikEditorScroll, QWidget#ikEditorViewport,
        QWidget#ikEditorTabContent {{
            background: {theme.panel_bg};
            color: {theme.text};
        }}
        QTabBar::tab {{
            color: {theme.muted_text};
            background: {theme.elevated_bg};
            border: 1px solid {theme.border};
            padding: 5px 8px;
        }}
        QTabBar::tab:selected {{
            color: {theme.text};
            background: {theme.panel_bg};
            border-color: {theme.focus_border};
        }}
        QTabBar QToolButton {{
            color: {theme.text};
            background: {theme.elevated_bg};
            border: 1px solid {theme.border};
            border-radius: 4px;
        }}
        QTabBar QToolButton:hover {{
            background: {theme.panel_hover_bg};
            border-color: {theme.focus_border};
        }}
        QTabBar QToolButton::left-arrow {{
            image: url("{_icon_url(theme, "chevron-left")}");
            width: 10px;
            height: 10px;
        }}
        QTabBar QToolButton::right-arrow {{
            image: url("{_icon_url(theme, "chevron-right")}");
            width: 10px;
            height: 10px;
        }}
        QScrollArea {{
            background: transparent;
            border: none;
        }}
    """
        + _menu_stylesheet(theme)
        + _section_stylesheet(theme)
        + _toolbar_stylesheet(theme)
        + _render_progress_stylesheet(theme)
        + _tutorial_card_stylesheet(theme)
        + _help_button_stylesheet(theme)
    )


class ThemeSynchronizer(QObject):
    """Keeps GhostGUI's stylesheet in sync with the system color scheme."""

    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        self._original_font = QFont(app.font())
        self._pending = False
        app.installEventFilter(self)
        if hasattr(app, "paletteChanged"):
            app.paletteChanged.connect(self.schedule_apply)
        hints = app.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self.schedule_apply)
        self.apply()

    def eventFilter(self, watched, event):
        if event.type() in (
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.PaletteChange,
        ):
            self.schedule_apply()
        return False

    def schedule_apply(self, *args) -> None:
        if self._pending:
            return
        self._pending = True
        QTimer.singleShot(0, self.apply)

    def apply(self) -> None:
        self._pending = False
        style = application_stylesheet()
        if self.app.styleSheet() != style:
            self.app.setStyleSheet(style)
        self.app.setFont(self._original_font)


def ensure_application_theme(app: QApplication) -> ThemeSynchronizer:
    synchronizer = getattr(app, "_ghostgui_theme_synchronizer", None)
    if synchronizer is None:
        synchronizer = ThemeSynchronizer(app)
        app._ghostgui_theme_synchronizer = synchronizer
    else:
        synchronizer.apply()
    return synchronizer


def apply_application_theme(app: QApplication) -> None:
    ensure_application_theme(app)


def _section_stylesheet(theme: Theme) -> str:
    return f"""
        QWidget#CollapsibleSection {{
            border: 1px solid {theme.border};
            border-radius: 6px;
            background: {theme.section_bg};
        }}
        QWidget#sectionContent {{
            background: {theme.section_bg};
            color: {theme.text};
        }}
        QToolButton#sectionHeader {{
            border: none;
            border-bottom: 1px solid {theme.border};
            padding: 3px 6px;
            text-align: left;
            background: {theme.section_header_bg};
            color: {theme.section_header_text};
        }}
        QToolButton#sectionHeader:checked {{
            background: {theme.section_header_active_bg};
            color: {theme.section_header_text};
        }}
        QToolButton#sectionHeader:hover,
        QToolButton#sectionHeader:checked:hover {{
            background: {theme.section_header_hover_bg};
            color: {theme.section_header_hover_text};
        }}
        QTabBar#editingModeBar {{
            color: {theme.text};
            background: transparent;
        }}
        QTabBar#editingModeBar::tab {{
            color: {theme.muted_text};
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 3px 6px;
            margin: 0;
        }}
        QTabBar#editingModeBar::tab:selected {{
            color: {theme.text};
            background: {theme.section_header_active_bg};
            border-bottom-color: {theme.focus_border};
        }}
        QTabBar#editingModeBar::tab:hover {{
            color: {theme.section_header_hover_text};
            background: {theme.section_header_hover_bg};
        }}
        QStackedWidget#editingModeStack,
        QStackedWidget#jointEditorStack,
        QWidget#endEffectorEditorPage,
        QScrollArea#jointEditorScroll {{
            border: none;
            background: transparent;
        }}
    """


def section_stylesheet(widget: QWidget | None = None) -> str:
    return _section_stylesheet(current_theme(widget))


def _menu_stylesheet(theme: Theme) -> str:
    return f"""
        QMenuBar#appMenuBar {{
            color: {theme.text};
            background: {theme.sidebar_bg};
            border: none;
            border-bottom: 1px solid {theme.border};
            padding: 1px 4px;
        }}
        QMenuBar#appMenuBar::item {{
            background: transparent;
            padding: 3px 7px;
        }}
        QMenuBar#appMenuBar::item:selected,
        QMenuBar#appMenuBar::item:pressed {{
            color: {theme.text};
            background: {theme.panel_hover_bg};
        }}
        QMenu {{
            color: {theme.text};
            background: {theme.panel_bg};
            border: 1px solid {theme.border};
            padding: 3px;
        }}
        QMenu::item {{
            padding: 4px 26px 4px 22px;
        }}
        QMenu::item:selected {{
            color: {theme.section_header_hover_text};
            background: {theme.section_header_hover_bg};
        }}
        QMenu::item:disabled {{
            color: {theme.disabled_text};
        }}
        QMenu::separator {{
            background: {theme.border};
            height: 1px;
            margin: 4px 6px;
        }}
    """


def _toolbar_stylesheet(theme: Theme) -> str:
    return f"""
        QToolBar#workflowToolbar {{
            background: {theme.sidebar_bg};
            border: none;
            border-bottom: 1px solid {theme.border};
            spacing: 2px;
            padding: 3px 5px;
        }}
        QToolBar#workflowToolbar::separator {{
            background: {theme.border};
            width: 1px;
            margin: 4px 3px;
        }}
        QToolBar#workflowToolbar QToolButton {{
            color: {theme.text};
            background: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 3px 5px;
        }}
        QToolBar#workflowToolbar QToolButton:hover {{
            color: {theme.section_header_hover_text};
            background: {theme.section_header_hover_bg};
            border-color: {theme.focus_border};
        }}
        QToolBar#workflowToolbar QToolButton:pressed,
        QToolBar#workflowToolbar QToolButton:checked {{
            color: {theme.text};
            background: {theme.section_header_active_bg};
            border-color: {theme.focus_border};
        }}
        QToolBar#workflowToolbar QToolButton:disabled {{
            color: {theme.disabled_text};
            background: transparent;
            border-color: transparent;
        }}
    """


def toolbar_stylesheet(widget: QWidget | None = None) -> str:
    return _toolbar_stylesheet(current_theme(widget))


def _render_progress_stylesheet(theme: Theme) -> str:
    return f"""
        QWidget#renderProgressOverlay {{
            background: {theme.overlay_scrim};
        }}
        QWidget#renderProgressCard {{
            background: {theme.overlay_card_bg};
            border: 1px solid {theme.border};
            border-radius: 6px;
        }}
        QLabel#renderTitle {{
            color: {theme.text};
        }}
        QLabel#renderDetail {{
            color: {theme.muted_text};
        }}
        QProgressBar {{
            color: {theme.text};
            border: 1px solid {theme.border};
            border-radius: 4px;
            min-height: 16px;
            text-align: center;
            background: {theme.overlay_progress_bg};
        }}
        QProgressBar::chunk {{
            background: {theme.accent};
            border-radius: 3px;
        }}
    """


def render_progress_stylesheet(widget: QWidget | None = None) -> str:
    return _render_progress_stylesheet(current_theme(widget))


def _tutorial_card_stylesheet(theme: Theme) -> str:
    return f"""
        QWidget#tutorialCard {{
            background-color: {theme.overlay_card_bg};
            border: 1px solid {theme.border};
            border-radius: 7px;
        }}
        QLabel#tutorialStepLabel {{
            color: {theme.muted_text};
        }}
        QLabel#tutorialTitle {{
            color: {theme.text};
        }}
        QLabel#tutorialBody {{
            color: {theme.text};
        }}
        QPushButton {{
            color: {theme.text};
            background-color: {theme.elevated_bg};
            border: 1px solid {theme.border};
            border-radius: 4px;
            padding: 4px 10px;
        }}
        QPushButton:hover {{
            color: {theme.text};
            background-color: {theme.panel_hover_bg};
            border-color: {theme.focus_border};
        }}
        QPushButton:disabled {{
            color: {theme.disabled_text};
            background-color: {theme.panel_bg};
        }}
    """


def tutorial_card_stylesheet(widget: QWidget | None = None) -> str:
    return _tutorial_card_stylesheet(current_theme(widget))


def _help_button_stylesheet(theme: Theme) -> str:
    return f"""
        QToolButton#helpButton {{
            color: {theme.text};
            background: {theme.quick_panel_bg};
            border: 1px solid {theme.quick_panel_border};
            border-radius: 4px;
            padding: 4px 8px;
        }}
        QToolButton#helpButton:hover {{
            color: {theme.section_header_hover_text};
            background: {theme.section_header_hover_bg};
            border-color: {theme.focus_border};
        }}
    """


def help_button_stylesheet(widget: QWidget | None = None) -> str:
    return _help_button_stylesheet(current_theme(widget))
