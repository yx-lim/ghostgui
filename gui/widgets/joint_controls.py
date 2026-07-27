"""Joint and IK influence controls."""

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .inline_value_slider import InlineValueSlider


class JointControl(QWidget):
    value_changed = Signal(str, float)

    def __init__(self, name, limits, value):
        super().__init__()
        self.name = name
        self.lo, self.hi = limits if limits is not None else (-3.14159, 3.14159)
        self._syncing = False
        self.slider = InlineValueSlider(
            self.lo,
            self.hi,
            value,
            single_step=math.pi / 180.0,
            decimals=1,
            suffix="°",
            display_scale=180.0 / math.pi,
        )
        self.slider.setAccessibleName(name)
        self.slider.setAccessibleDescription(
            "Joint angle in degrees. Drag to adjust, click a side to step "
            "one degree, or press Enter or F2 to type a value."
        )
        name_label = QLabel(name)
        name_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name_label)
        layout.addWidget(self.slider)
        self.slider.logical_value_changed.connect(self._changed)
        self.set_value(value)

    def _changed(self, value):
        if not self._syncing:
            self.value_changed.emit(self.name, float(value))

    def set_value(self, value):
        self._syncing = True
        try:
            self.slider.set_logical_value(value)
        finally:
            self._syncing = False


class IKInfluenceControl(QWidget):
    value_changed = Signal(str, float)

    def __init__(self, name, value=1.0):
        super().__init__()
        self.name = name
        self._syncing = False
        self.slider = InlineValueSlider(
            0.0,
            3.0,
            value,
            single_step=0.01,
            decimals=2,
        )
        self.slider.setAccessibleName(f"{name} IK influence")
        self.slider.setAccessibleDescription(
            "Joint IK influence. Drag to adjust, click a side to step by "
            "0.01, or press Enter or F2 to type a value."
        )
        name_label = QLabel(name)
        name_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name_label)
        layout.addWidget(self.slider)
        self.slider.logical_value_changed.connect(self._changed)
        self.set_value(value)

    def _changed(self, value):
        if not self._syncing:
            self.value_changed.emit(self.name, float(value))

    def set_value(self, value):
        self._syncing = True
        try:
            self.slider.set_logical_value(value)
        finally:
            self._syncing = False
