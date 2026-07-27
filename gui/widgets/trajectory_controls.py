"""Reusable compact controls used by the trajectory editor."""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QStackedWidget,
    QWidget,
)

from .inline_value_slider import InlineValueSlider


class LabeledSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, name, min_value, max_value, initial_value, scale=100):
        super().__init__()
        self.name = name
        self.scale = scale
        self.decimals = max(2, int(math.ceil(math.log10(max(1, scale)))))
        self._syncing = False

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.label = QLabel()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.input = QDoubleSpinBox()
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(initial_value)
        self.input.setDecimals(self.decimals)
        self.input.setRange(min_value / self.scale, max_value / self.scale)
        self.input.setSingleStep(1 / self.scale)
        self.input.setValue(initial_value / self.scale)
        self.input.setMinimumWidth(52)
        self.input.setMaximumWidth(76)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.input.valueChanged.connect(self.on_input_changed)
        self.label.setText(self.name)
        self.label.setMinimumWidth(64)
        self.label.setMaximumWidth(86)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.input)
        self.setLayout(layout)

    def value(self):
        return self.slider.value() / self.scale

    def set_value(self, value):
        self.slider.setValue(round(value * self.scale))

    def set_range(self, min_value, max_value):
        current_value = max(
            min_value / self.scale,
            min(self.value(), max_value / self.scale),
        )
        self._syncing = True
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(round(current_value * self.scale))
        self.input.setRange(min_value / self.scale, max_value / self.scale)
        self.input.setValue(current_value)
        self._syncing = False

    def on_slider_changed(self, raw_value):
        if self._syncing:
            return
        self._syncing = True
        value = raw_value / self.scale
        self.input.setValue(value)
        self._syncing = False
        self.value_changed.emit(value)

    def on_input_changed(self, value):
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(round(value * self.scale))
        self._syncing = False
        self.value_changed.emit(self.value())


class InlineLabeledSlider(QWidget):
    """Static field label plus one inline-editable numeric slider."""

    value_changed = Signal(float)
    interaction_finished = Signal()

    def __init__(
        self,
        name,
        min_value,
        max_value,
        initial_value,
        *,
        single_step,
        decimals=2,
        suffix="",
        display_scale=1.0,
    ):
        super().__init__()
        self.name = name
        self._syncing = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.label = QLabel(name)
        self.label.setMinimumWidth(64)
        self.label.setMaximumWidth(86)
        self.slider = InlineValueSlider(
            min_value,
            max_value,
            initial_value,
            single_step=single_step,
            decimals=decimals,
            suffix=suffix,
            display_scale=display_scale,
        )
        self.slider.setAccessibleName(name)
        self.slider.setAccessibleDescription(
            "Drag to adjust, click the left or right half to step, "
            "or press Enter or F2 to type a value."
        )
        self.slider.logical_value_changed.connect(self._on_value_changed)
        self.slider.interaction_finished.connect(
            self.interaction_finished.emit
        )
        layout.addWidget(self.label)
        layout.addWidget(self.slider, stretch=1)

    def value(self):
        return self.slider.logical_value()

    def set_value(self, value):
        self._syncing = True
        try:
            self.slider.set_logical_value(value)
        finally:
            self._syncing = False

    def set_range(self, min_value, max_value):
        self._syncing = True
        try:
            self.slider.set_logical_range(min_value, max_value)
        finally:
            self._syncing = False

    def _on_value_changed(self, value):
        if not self._syncing:
            self.value_changed.emit(float(value))


class CurrentPageStack(QStackedWidget):
    """A stack whose size hint follows the visible page instead of the largest."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def sizeHint(self):
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self):
        current = self.currentWidget()
        return (
            current.minimumSizeHint()
            if current is not None
            else super().minimumSizeHint()
        )


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()
