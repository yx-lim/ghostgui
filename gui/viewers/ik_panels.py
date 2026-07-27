"""Focused builders for the advanced IK inspector panels."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.compact import compact_combo, compact_spinbox
from gui.widgets.joint_controls import IKInfluenceControl


def make_ik_scroll_area(content):
    scroll = QScrollArea()
    scroll.setObjectName("ikEditorScroll")
    scroll.viewport().setObjectName("ikEditorViewport")
    content.setObjectName(content.objectName() or "ikEditorTabContent")
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(0)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    return scroll


def build_solver_widget(viewer):
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)

    solver_group = QGroupBox("Solver")
    solver_group.setMinimumWidth(0)
    solver_layout = QFormLayout(solver_group)
    solver_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    viewer.ik_damping = QDoubleSpinBox()
    compact_spinbox(viewer.ik_damping)
    viewer.ik_damping.setRange(0.001, 1.0)
    viewer.ik_damping.setDecimals(4)
    viewer.ik_damping.setValue(0.04)
    viewer.ik_max_iterations = QSpinBox()
    compact_spinbox(viewer.ik_max_iterations)
    viewer.ik_max_iterations.setRange(1, 300)
    viewer.ik_max_iterations.setValue(80)
    viewer.ik_step_size = QDoubleSpinBox()
    compact_spinbox(viewer.ik_step_size)
    viewer.ik_step_size.setRange(0.01, 1.0)
    viewer.ik_step_size.setValue(0.7)
    viewer.ik_max_step = QDoubleSpinBox()
    compact_spinbox(viewer.ik_max_step)
    viewer.ik_max_step.setRange(0.001, 0.5)
    viewer.ik_max_step.setDecimals(3)
    viewer.ik_max_step.setValue(0.08)
    viewer.ik_position_tolerance = QDoubleSpinBox()
    compact_spinbox(viewer.ik_position_tolerance)
    viewer.ik_position_tolerance.setRange(0.0001, 0.1)
    viewer.ik_position_tolerance.setDecimals(4)
    viewer.ik_position_tolerance.setValue(0.005)
    viewer.ik_orientation_tolerance = QDoubleSpinBox()
    compact_spinbox(viewer.ik_orientation_tolerance)
    viewer.ik_orientation_tolerance.setRange(0.001, 1.0)
    viewer.ik_orientation_tolerance.setDecimals(3)
    viewer.ik_orientation_tolerance.setValue(0.03)
    solver_layout.addRow("Damping", viewer.ik_damping)
    solver_layout.addRow("Max iterations", viewer.ik_max_iterations)
    solver_layout.addRow("Step size", viewer.ik_step_size)
    solver_layout.addRow("Max joint step", viewer.ik_max_step)
    solver_layout.addRow("Position tolerance", viewer.ik_position_tolerance)
    solver_layout.addRow(
        "Orientation tolerance", viewer.ik_orientation_tolerance
    )
    layout.addWidget(solver_group)
    layout.addStretch()
    return make_ik_scroll_area(content)


def build_ik_tasks_widget(viewer):
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)

    task_group = QGroupBox("IK Tasks")
    task_group.setMinimumWidth(0)
    task_layout = QFormLayout(task_group)
    task_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    viewer.ik_task_controls = {}
    defaults = {
        "tcp_position": (True, 1.0),
        "tcp_orientation": (
            viewer.robot_model.model_type != "quadruped"
            or viewer.robot_model.info.key == "go2",
            0.25,
        ),
        "posture": (False, 0.05),
        "foot_lock": (False, 0.5),
        "root_orientation": (True, 0.1),
        "regularization": (False, 0.01),
    }
    labels = {
        "tcp_position": "TCP position",
        "tcp_orientation": "TCP orientation",
        "posture": "Posture preservation",
        "foot_lock": "Foot lock",
        "root_orientation": "Root/base upright",
        "regularization": "Joint regularization",
    }
    for key, (enabled, weight) in defaults.items():
        checkbox = QCheckBox()
        checkbox.setChecked(enabled)
        spin = QDoubleSpinBox()
        compact_spinbox(spin)
        spin.setRange(0.0, 10.0)
        spin.setDecimals(3)
        spin.setValue(weight)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(checkbox)
        row_layout.addWidget(spin)
        viewer.ik_task_controls[key] = (checkbox, spin)
        task_layout.addRow(labels[key], row)
    layout.addWidget(task_group)
    layout.addStretch()
    return make_ik_scroll_area(content)


def build_joint_weights_widget(viewer):
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)

    influence_group = QGroupBox("Joint Weights")
    influence_group.setMinimumWidth(0)
    influence_layout = QVBoxLayout(influence_group)
    viewer.ik_preset_box = QComboBox()
    compact_combo(viewer.ik_preset_box, minimum_chars=8)
    presets = [
        "All joints normal",
        "Root locked",
        "Selected limb only",
        "Feet planted",
    ]
    if viewer.robot_model.model_type == "humanoid":
        presets.extend(("Upper body only", "Legs only"))
    elif viewer.robot_model.model_type == "quadruped":
        presets.append("Quadruped legs only")
    viewer.ik_preset_box.addItems(presets)
    apply_preset = QPushButton("Apply")
    apply_preset.clicked.connect(viewer.apply_ik_preset)
    influence_layout.addWidget(viewer.ik_preset_box)
    influence_layout.addWidget(apply_preset)
    for name in viewer.robot_model.get_joint_names():
        control = IKInfluenceControl(name, 1.0)
        control.value_changed.connect(viewer._ik_influence_changed)
        viewer.ik_influence_controls[name] = control
        influence_layout.addWidget(control)
    influence_layout.addStretch()
    layout.addWidget(influence_group)
    layout.addStretch()
    return make_ik_scroll_area(content)
