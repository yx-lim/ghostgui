"""Navigation card shown above the tutorial overlay."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TutorialCard(QWidget):
    """Small floating card with tutorial text and navigation controls."""

    back_requested = Signal()
    next_requested = Signal()
    skip_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tutorialCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.step_label = QLabel()
        self.step_label.setObjectName("tutorialStepLabel")

        self.title_label = QLabel()
        self.title_label.setObjectName("tutorialTitle")
        self.title_label.setWordWrap(True)

        self.body_label = QLabel()
        self.body_label.setObjectName("tutorialBody")
        self.body_label.setWordWrap(True)
        self.body_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 4, 0, 0)
        button_row.setSpacing(6)

        self.skip_button = QPushButton("Skip")
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.skip_button.setObjectName("tutorialSkipButton")
        self.back_button.setObjectName("tutorialBackButton")
        self.next_button.setObjectName("tutorialNextButton")
        self.skip_button.clicked.connect(self.skip_requested.emit)
        self.back_button.clicked.connect(self.back_requested.emit)
        self.next_button.clicked.connect(self.next_requested.emit)

        button_row.addWidget(self.skip_button)
        button_row.addStretch(1)
        button_row.addWidget(self.back_button)
        button_row.addWidget(self.next_button)

        layout.addWidget(self.step_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.body_label)
        layout.addLayout(button_row)
        self.hide()

    def set_step(self, step, index, total):
        self.step_label.setText(f"Step {index + 1} of {total}")
        self.title_label.setText(step.title)
        self.body_label.setText(step.body)
        self.back_button.setEnabled(index > 0)
        self.next_button.setText("Done" if index == total - 1 else "Next")
        self.adjustSize()
