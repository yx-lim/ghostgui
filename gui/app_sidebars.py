"""Application-level sidebar sections for the GhostGUI main window."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class SidebarSplitterHandle(QSplitterHandle):
    """Splitter handle with an optional persistent sidebar toggle."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setFixedSize(12, 140)
        self.toggle_button.hide()
        layout.addStretch(1)
        layout.addWidget(
            self.toggle_button,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        layout.addStretch(1)

    def configure_sidebar_toggle(self, side, enabled=True):
        self.sidebar_side = side
        self.toggle_button.setObjectName(
            f"{side}SidebarCollapseButton"
        )
        self.toggle_button.setVisible(bool(enabled))

    def set_sidebar_collapsed(self, collapsed):
        if self.sidebar_side == "right":
            arrow = (
                Qt.ArrowType.LeftArrow
                if collapsed
                else Qt.ArrowType.RightArrow
            )
        else:
            arrow = (
                Qt.ArrowType.RightArrow
                if collapsed
                else Qt.ArrowType.LeftArrow
            )
        self.toggle_button.setArrowType(arrow)
        self.toggle_button.setToolTip(
            f"{'Expand' if collapsed else 'Collapse'} "
            f"{self.sidebar_side} sidebar"
        )


class SidebarSplitter(QSplitter):
    """Main splitter that exposes a toggle from its left divider."""

    left_sidebar_toggle_requested = Signal()
    right_sidebar_toggle_requested = Signal()

    def createHandle(self):
        return SidebarSplitterHandle(self.orientation(), self)

    def configure_left_sidebar_handle(self):
        handle = self.handle(1)
        if isinstance(handle, SidebarSplitterHandle):
            handle.configure_sidebar_toggle("left")
            handle.toggle_button.clicked.connect(
                self.left_sidebar_toggle_requested.emit
            )

    def configure_right_sidebar_handle(self):
        handle = self.handle(2)
        if isinstance(handle, SidebarSplitterHandle):
            handle.configure_sidebar_toggle("right")
            handle.toggle_button.clicked.connect(
                self.right_sidebar_toggle_requested.emit
            )

    def set_left_sidebar_collapsed(self, collapsed):
        handle = self.handle(1)
        if isinstance(handle, SidebarSplitterHandle):
            handle.set_sidebar_collapsed(collapsed)

    def set_right_sidebar_collapsed(self, collapsed):
        handle = self.handle(2)
        if isinstance(handle, SidebarSplitterHandle):
            handle.set_sidebar_collapsed(collapsed)


class CollapsibleSection(QWidget):
    """A compact section that hides content without deleting its widgets."""

    def __init__(self, title, content, expanded=True, parent=None):
        super().__init__(parent)
        self.title = title
        self.content = content
        self.content.setObjectName(self.content.objectName() or "sectionContent")
        self.content.setMinimumWidth(0)

        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.toggled.connect(self.set_expanded)
        self.header.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.header.setMinimumWidth(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self.setObjectName("CollapsibleSection")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.set_expanded(bool(expanded))

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        self.content.setVisible(expanded)
        self.header.setChecked(expanded)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class AppSidebar(QWidget):
    """One scroll area containing all app-level sidebar sections."""

    SECTION_MAX_WIDTH = 240

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AppSidebar")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("appSidebarScroll")
        self.scroll.viewport().setObjectName("appSidebarViewport")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.body = QWidget()
        self.body.setObjectName("appSidebarBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 4, 0, 4)
        self.body_layout.setSpacing(2)
        self.scroll.setWidget(self.body)
        self.sections = []

        root.addWidget(self.scroll)

    def add_section(self, title, widget, expanded=True):
        if self.SECTION_MAX_WIDTH is not None:
            widget.setMaximumWidth(self.SECTION_MAX_WIDTH - 16)
        else:
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                widget.sizePolicy().verticalPolicy(),
            )
        section = CollapsibleSection(title, widget, expanded=expanded)
        if self.SECTION_MAX_WIDTH is not None:
            section.setMaximumWidth(self.SECTION_MAX_WIDTH)
        self.body_layout.addWidget(section)
        self.sections.append(section)
        return section

    def add_sections(self, sections):
        added = []
        for title, widget, expanded in sections:
            added.append(self.add_section(title, widget, expanded=expanded))
        return added

    def add_stretch(self):
        self.body_layout.addStretch(1)


class AppLeftSidebar(AppSidebar):
    SECTION_MAX_WIDTH = None

    def __init__(
        self,
        trajectory_controls,
        editor_tabs=None,
        project_panel=None,
        include_view=True,
        parent=None,
    ):
        super().__init__(parent)
        self.body_layout.setContentsMargins(4, 4, 0, 4)
        if project_panel is not None:
            self.add_section("Project", project_panel, expanded=True)
        self.add_sections(trajectory_controls.workflow_sections())
        self.view_panel = None
        if include_view:
            self.view_panel = self._build_view_panel(
                trajectory_controls.view_panel, editor_tabs
            )
            self.add_section("View", self.view_panel, expanded=False)
        self.add_stretch()

    def _build_view_panel(self, display_panel, editor_tabs):
        layout = display_panel.layout()
        if editor_tabs is None:
            return display_panel
        buttons = []
        for index in range(editor_tabs.count()):
            button = QPushButton(editor_tabs.tabText(index))
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, tab_index=index: editor_tabs.setCurrentIndex(
                    tab_index
                )
            )
            buttons.append(button)
            layout.addWidget(button)

        def sync_active(active_index):
            for index, button in enumerate(buttons):
                button.setChecked(index == active_index)

        editor_tabs.currentChanged.connect(sync_active)
        sync_active(editor_tabs.currentIndex())
        return display_panel


class AppRightSidebar(AppSidebar):
    SECTION_MAX_WIDTH = None

    def __init__(self, status_panel, base_sections=None, parent=None):
        super().__init__(parent)
        self.add_section("Status", status_panel, expanded=True)
        if base_sections:
            self.add_sections(base_sections)
        self.add_stretch()

    def set_context_widget(self, widget):
        return

    def current_context_widget(self):
        return None
