"""Compact status summary with optional diagnostic details."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class EditorStatusPanel(QWidget):
    """Own status widgets without owning status parsing or application state."""

    def __init__(self, model_source: str, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Retained for integrations, but intentionally not rendered because the
        # same backend appears in the status summary.
        self.backend_label = QLabel()
        self.backend_label.setWordWrap(True)

        self.status_icon_label = QLabel()
        self.status_icon_label.setObjectName("statusSeverityIcon")
        self.status_icon_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.status_icon_label.setFixedWidth(18)

        self.viewer_status_label = QLabel()
        self.viewer_status_label.setObjectName("statusEventTitle")
        self.viewer_status_label.setWordWrap(True)
        self.status_message_label = QLabel()
        self.status_message_label.setObjectName("statusEventMessage")
        self.status_message_label.setWordWrap(True)

        summary = QWidget()
        summary.setObjectName("statusEventSummary")
        summary.setAccessibleName("Current status")
        summary_layout = QHBoxLayout(summary)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(5)
        summary_layout.addWidget(self.status_icon_label)
        summary_copy = QVBoxLayout()
        summary_copy.setContentsMargins(0, 0, 0, 0)
        summary_copy.setSpacing(1)
        summary_copy.addWidget(self.viewer_status_label)
        summary_copy.addWidget(self.status_message_label)
        summary_layout.addLayout(summary_copy, stretch=1)
        layout.addWidget(summary)

        self.status_frame_label = QLabel("-")
        self.status_ik_label = QLabel("-")
        self.status_move_label = QLabel("-")
        self.viewer_time_label = QLabel()
        self.viewer_root_pose_label = QLabel()
        self.model_source_label = QLabel(model_source)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumWidth(0)
        self.status_text.setMinimumHeight(80)
        self.status_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        self.status_details_button = QToolButton()
        self.status_details_button.setText("Details")
        self.status_details_button.setCheckable(True)
        self.status_details_button.setChecked(False)
        self.status_details_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.status_details_button.setArrowType(Qt.ArrowType.RightArrow)

        self.status_details_panel = QWidget()
        details_layout = QVBoxLayout(self.status_details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        details_layout.addWidget(self.status_text)
        self.status_details_panel.setVisible(False)

        self.status_details_button.toggled.connect(self.set_details_visible)
        layout.addWidget(self.status_details_button)
        layout.addWidget(self.status_details_panel)

    def set_details_visible(self, visible):
        self.status_details_panel.setVisible(visible)
        self.status_details_button.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )

    def compatibility_widgets(self):
        """Return the stable main-window attribute surface during migration."""
        names = (
            "backend_label",
            "status_icon_label",
            "viewer_status_label",
            "status_message_label",
            "status_frame_label",
            "status_ik_label",
            "status_move_label",
            "viewer_time_label",
            "viewer_root_pose_label",
            "model_source_label",
            "status_text",
            "status_details_button",
            "status_details_panel",
        )
        return {name: getattr(self, name) for name in names}
