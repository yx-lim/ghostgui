"""A compact slider that paints and edits its numeric value inline."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QDoubleValidator,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QLineEdit, QSlider

from gui.theme import current_theme


class InlineValueSlider(QSlider):
    """A horizontal float slider with a filled bar and temporary inline editor.

    QSlider stores integers, so this widget maps a high-resolution internal
    range onto a logical floating-point range. Callers interact only with
    ``logical_value()``, ``set_logical_value()``, and
    ``logical_value_changed``.
    """

    logical_value_changed = Signal(float)
    interaction_finished = Signal()

    # Keep programmatic pose synchronization effectively lossless while still
    # using QSlider's integer storage. This remains well inside Qt's int range.
    _RAW_RESOLUTION = 100_000_000

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float = 0.0,
        *,
        single_step: float,
        decimals: int = 2,
        suffix: str = "",
        display_scale: float = 1.0,
        parent=None,
    ):
        super().__init__(Qt.Orientation.Horizontal, parent)
        if maximum <= minimum:
            raise ValueError("InlineValueSlider maximum must exceed minimum")
        if single_step <= 0:
            raise ValueError("InlineValueSlider single_step must be positive")
        if display_scale <= 0:
            raise ValueError("InlineValueSlider display_scale must be positive")

        self._logical_minimum = float(minimum)
        self._logical_maximum = float(maximum)
        self._logical_single_step = float(single_step)
        self._display_scale = float(display_scale)
        self._decimals = max(0, int(decimals))
        self._suffix = str(suffix)
        self._press_position = QPointF()
        self._dragging = False
        self._hovered = False
        self._closing_editor = False

        super().setRange(0, self._RAW_RESOLUTION)
        self.setTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumHeight(24)

        self.editor = QLineEdit(self)
        self.editor.setObjectName("inlineValueEditor")
        self.editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.editor.setFrame(True)
        self.editor.hide()
        self.editor.installEventFilter(self)
        self.editor.editingFinished.connect(self._commit_editor)

        super().valueChanged.connect(self._on_raw_value_changed)
        self.set_logical_value(value)
        self._update_editor_validator()

    def logical_minimum(self) -> float:
        return self._logical_minimum

    def logical_maximum(self) -> float:
        return self._logical_maximum

    def logical_single_step(self) -> float:
        return self._logical_single_step

    def logical_value(self) -> float:
        raw = super().value()
        if raw <= 0:
            return self._logical_minimum
        if raw >= self._RAW_RESOLUTION:
            return self._logical_maximum
        fraction = raw / float(self._RAW_RESOLUTION)
        return self._logical_minimum + (
            self._logical_maximum - self._logical_minimum
        ) * fraction

    def set_logical_value(self, value: float):
        clamped = max(
            self._logical_minimum,
            min(self._logical_maximum, float(value)),
        )
        fraction = (
            (clamped - self._logical_minimum)
            / (self._logical_maximum - self._logical_minimum)
        )
        super().setValue(round(fraction * self._RAW_RESOLUTION))
        self.update()

    def set_logical_range(self, minimum: float, maximum: float):
        if maximum <= minimum:
            raise ValueError("InlineValueSlider maximum must exceed minimum")
        current = self.logical_value()
        self._logical_minimum = float(minimum)
        self._logical_maximum = float(maximum)
        self._update_editor_validator()
        self.set_logical_value(current)

    def set_logical_single_step(self, step: float):
        if step <= 0:
            raise ValueError("InlineValueSlider single step must be positive")
        self._logical_single_step = float(step)

    def display_value(self, logical_value: float | None = None) -> float:
        value = self.logical_value() if logical_value is None else logical_value
        return float(value) * self._display_scale

    def logical_from_display(self, display_value: float) -> float:
        return float(display_value) / self._display_scale

    def format_value(self, logical_value: float | None = None) -> str:
        displayed = self.display_value(logical_value)
        zero_threshold = 0.5 * (10.0 ** -self._decimals)
        if abs(displayed) < zero_threshold:
            displayed = 0.0
        number = f"{displayed:.{self._decimals}f}"
        if not self._suffix:
            return number
        separator = "" if self._suffix in ("°", "%") else " "
        return f"{number}{separator}{self._suffix}"

    def begin_inline_edit(self):
        if not self.isEnabled():
            return
        displayed = self.display_value()
        self.editor.setText(f"{displayed:.{self._decimals}f}")
        self._position_editor()
        self.editor.show()
        self.editor.raise_()
        self.editor.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.editor.selectAll()

    def step_logical_value(self, step_count: int):
        self.set_logical_value(
            self.logical_value() + step_count * self._logical_single_step
        )

    def _raw_from_position(self, x: float) -> int:
        groove = self._groove_rect()
        if groove.width() <= 0:
            return 0
        fraction = (float(x) - groove.left()) / groove.width()
        fraction = max(0.0, min(1.0, fraction))
        return round(fraction * self._RAW_RESOLUTION)

    def _on_raw_value_changed(self, _raw_value: int):
        self.logical_value_changed.emit(self.logical_value())
        self.update()

    def _update_editor_validator(self):
        lower = self.display_value(self._logical_minimum)
        upper = self.display_value(self._logical_maximum)
        validator = QDoubleValidator(
            min(lower, upper),
            max(lower, upper),
            max(6, self._decimals),
            self.editor,
        )
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.editor.setValidator(validator)

    def _parse_editor_value(self):
        text = self.editor.text().strip()
        suffix = self._suffix.strip()
        if suffix and text.endswith(suffix):
            text = text[: -len(suffix)].strip()
        value, accepted = self.locale().toDouble(text)
        if accepted:
            return float(value)
        try:
            return float(text)
        except ValueError:
            return None

    def _commit_editor(self, keep_open_on_invalid=False):
        if not self.editor.isVisible() or self._closing_editor:
            return
        display_value = self._parse_editor_value()
        if display_value is None:
            if keep_open_on_invalid:
                self.editor.setFocus(Qt.FocusReason.OtherFocusReason)
                self.editor.selectAll()
            else:
                self._close_editor()
            return
        self.set_logical_value(self.logical_from_display(display_value))
        self._close_editor()
        self.interaction_finished.emit()

    def _cancel_editor(self):
        if not self.editor.isVisible():
            return
        self._close_editor()

    def _close_editor(self):
        self._closing_editor = True
        try:
            self.editor.hide()
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        finally:
            self._closing_editor = False

    def eventFilter(self, watched, event):
        if (
            watched is self.editor
            and event.type() == QEvent.Type.KeyPress
        ):
            if event.key() == Qt.Key.Key_Escape:
                self._cancel_editor()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._commit_editor(keep_open_on_invalid=True)
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2):
            self.begin_inline_edit()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.step_logical_value(-1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.step_logical_value(1)
        elif key == Qt.Key.Key_PageDown:
            self.step_logical_value(-10)
        elif key == Qt.Key.Key_PageUp:
            self.step_logical_value(10)
        elif key == Qt.Key.Key_Home:
            self.set_logical_value(self._logical_minimum)
        elif key == Qt.Key.Key_End:
            self.set_logical_value(self._logical_maximum)
        else:
            super().keyPressEvent(event)
            return
        self.interaction_finished.emit()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._press_position = event.position()
        self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not event.buttons() & Qt.MouseButton.LeftButton:
            super().mouseMoveEvent(event)
            return
        distance = (
            event.position() - self._press_position
        ).manhattanLength()
        if distance >= QApplication.startDragDistance():
            self._dragging = True
        if self._dragging:
            super().setValue(self._raw_from_position(event.position().x()))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._dragging:
            super().setValue(self._raw_from_position(event.position().x()))
            self.interaction_finished.emit()
        elif not self._value_text_rect().contains(event.position()):
            self.step_logical_value(
                -1 if event.position().x() < self.rect().center().x() else 1
            )
            self.interaction_finished.emit()
        self._dragging = False
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._value_text_rect().contains(event.position())
        ):
            self.begin_inline_edit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if not delta:
            event.ignore()
            return
        self.step_logical_value(1 if delta > 0 else -1)
        self.interaction_finished.emit()
        event.accept()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.editor.isVisible():
            self._position_editor()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            self.update()

    def sizeHint(self):
        base = super().sizeHint()
        text_height = self.fontMetrics().height() + 8
        return QSize(max(100, base.width()), max(24, base.height(), text_height))

    def _groove_rect(self) -> QRectF:
        margin = 2.0
        return QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)

    def _value_text_rect(self) -> QRectF:
        groove = self._groove_rect()
        text_width = self.fontMetrics().horizontalAdvance(self.format_value())
        width = min(groove.width(), max(48.0, float(text_width + 14)))
        return QRectF(
            groove.center().x() - width / 2.0,
            groove.top(),
            width,
            groove.height(),
        )

    def _position_editor(self):
        width = max(
            64,
            min(
                self.width() - 8,
                self.fontMetrics().horizontalAdvance(self.editor.text()) + 28,
            ),
        )
        height = max(22, self.height() - 4)
        self.editor.setGeometry(
            round(self.rect().center().x() - width / 2),
            round(self.rect().center().y() - height / 2),
            round(width),
            round(height),
        )

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        theme = current_theme(self)
        groove = self._groove_rect()
        radius = min(4.0, groove.height() / 2.0)

        background = QColor(theme.panel_bg)
        border = QColor(theme.border)
        fill = QColor(theme.accent)
        normal_text = QColor(theme.text)
        filled_text = QColor(theme.accent_text)
        if not self.isEnabled():
            background = QColor(theme.window_bg)
            fill = QColor(theme.border)
            normal_text = QColor(theme.disabled_text)
            filled_text = QColor(theme.disabled_text)
        elif self._hovered:
            border = QColor(theme.focus_border)

        path = QPainterPath()
        path.addRoundedRect(groove, radius, radius)
        painter.fillPath(path, background)

        fraction = super().value() / float(self._RAW_RESOLUTION)
        fill_rect = QRectF(
            groove.left(),
            groove.top(),
            groove.width() * fraction,
            groove.height(),
        )
        painter.save()
        painter.setClipPath(path)
        painter.fillRect(fill_rect, fill)
        painter.restore()

        if self._logical_minimum < 0.0 < self._logical_maximum:
            zero_fraction = (
                -self._logical_minimum
                / (self._logical_maximum - self._logical_minimum)
            )
            zero_x = groove.left() + groove.width() * zero_fraction
            zero_color = QColor(theme.border)
            zero_color.setAlpha(150)
            painter.setPen(QPen(zero_color, 1.0))
            painter.drawLine(
                QPointF(zero_x, groove.top() + 3.0),
                QPointF(zero_x, groove.bottom() - 3.0),
            )

        focus_color = QColor(theme.focus_border)
        painter.setPen(
            QPen(
                focus_color if self.hasFocus() else border,
                1.5 if self.hasFocus() else 1.0,
            )
        )
        painter.drawPath(path)

        text = self.format_value()
        alignment = Qt.AlignmentFlag.AlignCenter
        painter.save()
        painter.setClipRect(fill_rect)
        painter.setPen(filled_text)
        painter.drawText(groove, alignment, text)
        painter.restore()

        unfilled_rect = QRectF(
            fill_rect.right(),
            groove.top(),
            max(0.0, groove.right() - fill_rect.right()),
            groove.height(),
        )
        painter.save()
        painter.setClipRect(unfilled_rect)
        painter.setPen(normal_text)
        painter.drawText(groove, alignment, text)
        painter.restore()
