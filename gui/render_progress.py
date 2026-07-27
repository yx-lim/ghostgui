"""Viewer-local render progress overlay."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class RenderProgressOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("renderProgressOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._allow_close = False
        if parent is not None:
            parent.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.card = QWidget(self)
        self.card.setObjectName("renderProgressCard")
        self.card.setFixedWidth(420)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(30, 24, 30, 24)
        card_layout.setSpacing(10)

        self.title_label = QLabel("Rendering robot model")
        self.title_label.setObjectName("renderTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)

        self.detail_label = QLabel("Preparing 3D geometry...")
        self.detail_label.setObjectName("renderDetail")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(300)
        self.progress_bar.setMaximumWidth(360)

        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.detail_label)
        card_layout.addWidget(
            self.progress_bar,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.card.setFixedHeight(self.card.sizeHint().height())
        layout.addStretch(1)
        layout.addWidget(self.card, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.hide()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self.update_geometry()
        return super().eventFilter(watched, event)

    def update_geometry(self):
        parent = self.parentWidget()
        if parent is None:
            return
        self.setGeometry(parent.rect())
        if not self.isHidden():
            self.raise_()

    def set_message(self, title, detail, progress=None):
        self.title_label.setText(title)
        self.detail_label.setText(detail)
        if progress is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(progress))))

    def show_rendering(self, title, detail, progress=None):
        self._allow_close = False
        self.set_message(title, detail, progress)
        self.update_geometry()
        self.show()
        self.raise_()

    def finish(self):
        self._allow_close = True
        self.hide()

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
        else:
            event.ignore()
