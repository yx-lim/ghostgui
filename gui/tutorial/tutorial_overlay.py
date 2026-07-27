"""Paint-only tutorial highlight overlay."""

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class TutorialOverlay(QWidget):
    """Dims the main window and highlights the current target widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tutorialOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.target_rect = QRect()
        self.hide()

    def set_target_rect(self, rect):
        self.target_rect = QRect(rect) if rect is not None else QRect()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        dim_path = QPainterPath()
        dim_path.setFillRule(Qt.FillRule.OddEvenFill)
        dim_path.addRect(self.rect())

        if not self.target_rect.isNull():
            highlight = self.target_rect.adjusted(-6, -6, 6, 6)
            dim_path.addRoundedRect(highlight, 8, 8)
            painter.setBrush(QColor(17, 24, 39, 145))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(dim_path)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(47, 128, 237), 3))
            painter.drawRoundedRect(highlight, 8, 8)
        else:
            painter.fillRect(self.rect(), QColor(17, 24, 39, 145))
