"""Compact sizing helpers for dense sidebar controls."""

from PySide6.QtWidgets import QComboBox, QSizePolicy


def compact_combo(combo, minimum_chars=10):
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(minimum_chars)
    combo.setMinimumWidth(0)
    combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)


def compact_spinbox(spinbox, width=78):
    spinbox.setMinimumWidth(0)
    spinbox.setMaximumWidth(width)
    spinbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
