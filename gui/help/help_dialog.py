"""In-app help center dialog."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from .help_content import HELP_SECTIONS


class HelpCenterDialog(QDialog):
    """Static help center for first-run workflow guidance."""

    start_tutorial_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GhostGUI Help")
        self.setObjectName("helpCenterDialog")
        self.resize(840, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        self.section_list = QListWidget()
        self.section_list.setObjectName("helpSectionList")
        self.section_list.setMaximumWidth(190)
        self.section_list.setMinimumWidth(170)

        self.browser = QTextBrowser()
        self.browser.setObjectName("helpContentBrowser")
        self.browser.setOpenExternalLinks(False)

        for section in HELP_SECTIONS:
            item = QListWidgetItem(section.title)
            item.setData(Qt.ItemDataRole.UserRole, section.body)
            self.section_list.addItem(item)

        self.section_list.currentItemChanged.connect(self._show_section)

        content_layout.addWidget(self.section_list)
        content_layout.addWidget(self.browser, stretch=1)
        root.addLayout(content_layout, stretch=1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.start_tutorial_button = QPushButton("Start Tutorial")
        self.start_tutorial_button.setObjectName("startTutorialButton")
        self.start_tutorial_button.clicked.connect(self.start_tutorial_requested.emit)
        button_box.addButton(
            self.start_tutorial_button,
            QDialogButtonBox.ButtonRole.ActionRole,
        )

        button_box.rejected.connect(self.close)
        root.addWidget(button_box)

        if self.section_list.count():
            self.section_list.setCurrentRow(0)

    def _show_section(self, current, previous=None):
        if current is None:
            self.browser.clear()
            return
        self.browser.setMarkdown(current.data(Qt.ItemDataRole.UserRole))
