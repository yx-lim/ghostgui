"""Recent project preview dialog for GhostGUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from .theme import current_theme


THUMBNAIL_SIZE = QSize(180, 104)
PROJECT_CARD_SIZE = QSize(220, 172)


class ProjectBrowserDialog(QDialog):
    """Visual recent-project picker backed by saved workspace snapshots."""

    def __init__(self, project_previews, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open GhostGUI Project")
        self.setObjectName("projectBrowserDialog")
        self.resize(760, 520)
        self.selected_project_path = None
        self.browse_requested = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.project_list = QListWidget()
        self.project_list.setObjectName("projectBrowserList")
        self.project_list.setViewMode(QListView.ViewMode.IconMode)
        self.project_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.project_list.setMovement(QListView.Movement.Static)
        self.project_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.project_list.setWrapping(True)
        self.project_list.setSpacing(8)
        self.project_list.setIconSize(THUMBNAIL_SIZE)
        self.project_list.setGridSize(PROJECT_CARD_SIZE)
        self.project_list.currentItemChanged.connect(self.update_open_button)
        self.project_list.itemDoubleClicked.connect(
            lambda _item: self.open_selected_project()
        )
        root.addWidget(self.project_list, stretch=1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("openProjectPreviewButton")
        self.open_button.clicked.connect(self.open_selected_project)
        button_box.addButton(
            self.open_button,
            QDialogButtonBox.ButtonRole.ActionRole,
        )

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setObjectName("browseProjectFolderButton")
        self.browse_button.clicked.connect(self.browse_for_project)
        button_box.addButton(
            self.browse_button,
            QDialogButtonBox.ButtonRole.ActionRole,
        )

        button_box.rejected.connect(self.reject)
        root.addWidget(button_box)

        self.set_project_previews(project_previews)

    def set_project_previews(self, project_previews):
        self.project_list.clear()
        for preview in project_previews:
            path = preview.get("path")
            if not path:
                continue
            item = QListWidgetItem()
            item.setText(self.project_item_text(preview))
            item.setIcon(self.project_icon(preview))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            item.setSizeHint(PROJECT_CARD_SIZE)
            self.project_list.addItem(item)
        if self.project_list.count():
            self.project_list.setCurrentRow(0)
        self.update_open_button()

    def project_item_text(self, preview):
        path = Path(preview.get("path", ""))
        name = preview.get("project_name") or path.stem
        model = preview.get("model_name") or preview.get("model_key") or "Unknown model"
        modified = self.short_timestamp(
            preview.get("modified_at") or preview.get("last_opened_at")
        )
        lines = [name, model]
        if modified:
            lines.append(modified)
        return "\n".join(lines)

    def short_timestamp(self, value):
        if not value:
            return ""
        stamp = str(value).replace("T", " ").replace("Z", "")
        return stamp.split(".")[0]

    def project_icon(self, preview):
        snapshot_path = preview.get("snapshot_path")
        if snapshot_path:
            pixmap = QPixmap(snapshot_path)
            if not pixmap.isNull():
                return QIcon(self.framed_snapshot(pixmap))
        return QIcon(self.placeholder_snapshot())

    def framed_snapshot(self, source):
        target = QPixmap(THUMBNAIL_SIZE)
        theme = current_theme(self)
        target.fill(QColor(theme.panel_bg))
        scaled = source.scaled(
            THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (THUMBNAIL_SIZE.width() - scaled.width()) // 2
        y = (THUMBNAIL_SIZE.height() - scaled.height()) // 2
        painter = QPainter(target)
        painter.drawPixmap(x, y, scaled)
        painter.setPen(QPen(QColor(theme.border), 1))
        painter.drawRect(target.rect().adjusted(0, 0, -1, -1))
        painter.end()
        return target

    def placeholder_snapshot(self):
        pixmap = QPixmap(THUMBNAIL_SIZE)
        theme = current_theme(self)
        pixmap.fill(QColor(theme.elevated_bg))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(theme.border), 1))
        painter.drawRoundedRect(pixmap.rect().adjusted(1, 1, -2, -2), 6, 6)
        painter.setPen(QColor(theme.muted_text))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "No snapshot")
        painter.end()
        return pixmap

    def update_open_button(self, current=None, previous=None):
        self.open_button.setEnabled(self.project_list.currentItem() is not None)

    def open_selected_project(self):
        item = self.project_list.currentItem()
        if item is None:
            return
        self.selected_project_path = item.data(Qt.ItemDataRole.UserRole)
        if self.selected_project_path:
            self.accept()

    def browse_for_project(self):
        self.browse_requested = True
        self.accept()
