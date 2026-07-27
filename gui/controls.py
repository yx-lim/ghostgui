"""
controls.py

Purpose:
    GUI panel for editing target reference-frame keyframes.

This replaces the older "robot control" sliders.

The GUI now edits:
    - selected robot frame: pelvis, left foot, right foot, torso
    - time
    - phase
    - target frame position
    - target frame roll/pitch/yaw
    - trajectory keyframe table
"""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QStackedWidget,
    QHBoxLayout,
    QCheckBox,
    QSizePolicy,
)

from core.trajectory import TargetFrame
from .widgets.trajectory_controls import (
    CurrentPageStack,
    InlineLabeledSlider,
    LabeledSlider,
    NoWheelComboBox,
)


class TrajectoryControlPanel(QGroupBox):
    """
    Left-side panel for creating trajectory keyframes.

    Signals:
        pose_changed:
            emitted when sliders change, so the viewer can update live.

        add_keyframe_clicked:
            emitted when user wants to add the current target frame.

        update_keyframe_clicked:
            emitted when user wants to overwrite selected keyframe.

        delete_keyframe_clicked:
            emitted when user wants to delete selected keyframe.

        clear_trajectory_clicked:
            emitted when user wants to delete every trajectory keyframe.

        generate_clicked:
            emitted when user wants to send trajectory to backend.
    """

    pose_changed = Signal(float, float, float, float, float, float)
    add_keyframe_clicked = Signal()
    update_keyframe_clicked = Signal()
    delete_keyframe_clicked = Signal()
    clear_trajectory_clicked = Signal()
    generate_clicked = Signal()
    keyframe_selected = Signal(int)
    frame_name_changed = Signal(str)
    trajectory_lines_changed = Signal(bool)
    keyframes_visibility_changed = Signal(bool)
    time_changed = Signal(float)
    model_changed = Signal(str)
    open_model_clicked = Signal()
    choose_mesh_folder_clicked = Signal()
    setup_import_requested = Signal(str)
    setup_export_requested = Signal(str)
    editing_mode_changed = Signal(str)

    def __init__(self, model_registry=None, model_key="g1", frame_names=None):
        super().__init__("Reference Frame Trajectory Editor")
        self._suppress_pose_changed = False
        self.model_registry = model_registry or {}
        self.model_key = model_key
        self.frame_names = list(frame_names or [
            "pelvis", "torso", "left_foot", "right_foot",
            "left_hand", "right_hand",
        ])
        self.build_ui()

    def build_ui(self):
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(root_layout)

        self.robot_panel, robot_layout = self._make_section_panel()
        self.target_panel, self.target_layout = self._make_section_panel()
        self.editing_mode_panel, self.editing_mode_layout = (
            self._make_section_panel()
        )
        self.editing_mode_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_ik_panel, self.preview_ik_layout = self._make_section_panel()
        self.trajectory_panel, self.trajectory_layout = self._make_section_panel()
        self.view_panel, self.view_layout = self._make_section_panel()

        if self.model_registry:
            robot_row = QHBoxLayout()
            robot_row.setContentsMargins(0, 0, 0, 0)
            robot_row.setSpacing(6)
            self.robot_label = QLabel("Robot")
            self.model_box = QComboBox()
            self.model_box.setObjectName("robotModelCombo")
            self.model_box.setToolTip("Choose the robot model and frame set to edit.")
            for key, info in self.model_registry.items():
                self.model_box.addItem(info.display_name, key)
            selected = self.model_box.findData(self.model_key)
            if selected >= 0:
                self.model_box.setCurrentIndex(selected)
            self.model_box.currentIndexChanged.connect(
                lambda index: self.model_changed.emit(self.model_box.itemData(index))
            )
            robot_row.addWidget(self.robot_label)
            robot_row.addWidget(self.model_box, stretch=1)
            robot_layout.addLayout(robot_row)

            import_row = QHBoxLayout()
            import_row.setContentsMargins(0, 0, 0, 0)
            import_row.setSpacing(6)
            self.import_action_label = QLabel("Import")
            self.import_action_box = NoWheelComboBox()
            self.import_action_box.setObjectName("importActionCombo")
            self.import_action_box.setToolTip("Import a robot model, qpos pose, or trajectory.")
            self.import_action_box.setPlaceholderText("Select...")
            self.import_action_box.addItem("Model", "model")
            self.import_action_box.addItem("Qpos", "qpos")
            self.import_action_box.addItem("Trajectory", "trajectory")
            self.import_action_box.setCurrentIndex(-1)
            self.import_action_box.activated.connect(
                self._emit_setup_import_action
            )
            import_row.addWidget(self.import_action_label)
            import_row.addWidget(self.import_action_box, stretch=1)
            robot_layout.addLayout(import_row)

            export_row = QHBoxLayout()
            export_row.setContentsMargins(0, 0, 0, 0)
            export_row.setSpacing(6)
            self.export_action_label = QLabel("Export")
            self.export_action_box = NoWheelComboBox()
            self.export_action_box.setObjectName("exportActionCombo")
            self.export_action_box.setToolTip(
                "Export the committed qpos pose or timed trajectory."
            )
            self.export_action_box.setPlaceholderText("Select...")
            self.export_action_box.addItem("Qpos", "qpos")
            self.export_action_box.addItem("Trajectory", "trajectory")
            self.export_action_box.setCurrentIndex(-1)
            self.export_action_box.activated.connect(
                self._emit_setup_export_action
            )
            export_row.addWidget(self.export_action_label)
            export_row.addWidget(self.export_action_box, stretch=1)
            robot_layout.addLayout(export_row)

            self.open_model_button = QPushButton("Import Model")
            self.open_model_button.setVisible(False)
            self.open_model_button.clicked.connect(self.open_model_clicked.emit)
            self.choose_mesh_folder_button = QPushButton("Meshes")
            self.choose_mesh_folder_button.setVisible(False)
            self.choose_mesh_folder_button.clicked.connect(
                self.choose_mesh_folder_clicked.emit
            )

        # --------------------------------------------------------
        # Select which robot frame this target refers to
        # --------------------------------------------------------
        self.target_layout.addWidget(QLabel("Target robot frame"))

        self.frame_box = QComboBox()
        self.frame_box.setObjectName("targetFrameCombo")
        self.frame_box.setToolTip("Choose the body, site, or logical frame to edit.")
        self.frame_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.frame_box.setMinimumContentsLength(10)
        self.frame_box.addItems(self.frame_names)
        # A hand is the most useful default for the 3D transform gizmo. The
        # user can still select pelvis/feet exactly as before.
        preferred = "left_hand" if "left_hand" in self.frame_names else self.frame_names[0]
        self.frame_box.setCurrentText(preferred)
        self.frame_box.currentTextChanged.connect(self.frame_name_changed.emit)
        self.target_layout.addWidget(self.frame_box)

        # --------------------------------------------------------
        # Editing mode: target-pose IK or direct joint angles
        # --------------------------------------------------------
        self.editing_mode_bar = QTabBar()
        self.editing_mode_bar.setObjectName("editingModeBar")
        self.editing_mode_bar.setDrawBase(False)
        self.editing_mode_bar.setExpanding(True)
        self.editing_mode_bar.setUsesScrollButtons(False)
        self.editing_mode_bar.setElideMode(Qt.TextElideMode.ElideRight)
        self.editing_mode_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.editing_mode_bar.addTab("End Effector")
        self.editing_mode_bar.addTab("Joint Angles")

        self.end_effector_page = QWidget()
        self.end_effector_page.setObjectName("endEffectorEditorPage")
        transform_layout = QVBoxLayout(self.end_effector_page)
        transform_layout.setContentsMargins(0, 4, 0, 0)
        transform_layout.setSpacing(4)
        self.transform_panel = self.end_effector_page

        self.joint_editor_stack = CurrentPageStack()
        self.joint_editor_stack.setObjectName("jointEditorStack")
        self.joint_editor_stack.setMinimumWidth(0)
        self.joint_editor_empty_page = QWidget()
        self.joint_editor_stack.addWidget(self.joint_editor_empty_page)

        self.editing_mode_stack = CurrentPageStack()
        self.editing_mode_stack.setObjectName("editingModeStack")
        self.editing_mode_stack.setMinimumWidth(0)
        self.editing_mode_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.editing_mode_stack.addWidget(self.end_effector_page)
        self.editing_mode_stack.addWidget(self.joint_editor_stack)
        self.editing_mode_bar.currentChanged.connect(
            self._on_editing_mode_changed
        )
        self.editing_mode_layout.addWidget(self.editing_mode_bar)
        self.editing_mode_layout.addWidget(self.editing_mode_stack)

        # --------------------------------------------------------
        # Motion phase
        # --------------------------------------------------------
        self.phase_label = QLabel("Motion phase")
        self.phase_label.setVisible(False)

        self.phase_box = QComboBox()
        self.phase_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.phase_box.setMinimumContentsLength(8)
        self.phase_box.addItems([
            "crouch",
            "launch",
            "flight",
            "landing",
        ])
        self.phase_box.setVisible(False)

        # --------------------------------------------------------
        # Sliders for target reference-frame pose
        # --------------------------------------------------------
        self.time_slider = LabeledSlider(
            "Time [s]",
            min_value=0,
            max_value=500,
            initial_value=0,
            scale=100,
        )
        self.time_slider.setObjectName("timeControl")
        self.time_slider.setToolTip("Choose the active time for editing and slicing.")

        self.x_slider = InlineLabeledSlider(
            "X [m]",
            min_value=-2.0,
            max_value=2.0,
            initial_value=0.0,
            single_step=0.001,
            decimals=3,
            suffix="m",
        )

        self.y_slider = InlineLabeledSlider(
            "Y [m]",
            min_value=-1.0,
            max_value=1.0,
            initial_value=0.0,
            single_step=0.001,
            decimals=3,
            suffix="m",
        )

        self.z_slider = InlineLabeledSlider(
            "Z [m]",
            min_value=0.0,
            max_value=2.0,
            initial_value=0.9,
            single_step=0.001,
            decimals=3,
            suffix="m",
        )

        self.roll_slider = InlineLabeledSlider(
            "Roll [°]",
            min_value=-3.14,
            max_value=3.14,
            initial_value=0.0,
            single_step=math.pi / 180.0,
            decimals=1,
            suffix="°",
            display_scale=180.0 / math.pi,
        )

        self.pitch_slider = InlineLabeledSlider(
            "Pitch [°]",
            min_value=-1.57,
            max_value=1.57,
            initial_value=0.0,
            single_step=math.pi / 180.0,
            decimals=1,
            suffix="°",
            display_scale=180.0 / math.pi,
        )

        self.yaw_slider = InlineLabeledSlider(
            "Yaw [°]",
            min_value=-3.14,
            max_value=3.14,
            initial_value=0.0,
            single_step=math.pi / 180.0,
            decimals=1,
            suffix="°",
            display_scale=180.0 / math.pi,
        )

        transform_layout.addWidget(self.x_slider)
        transform_layout.addWidget(self.y_slider)
        transform_layout.addWidget(self.z_slider)
        transform_layout.addWidget(self.roll_slider)
        transform_layout.addWidget(self.pitch_slider)
        transform_layout.addWidget(self.yaw_slider)

        # Whenever pose controls change, update the viewer target marker.
        self.x_slider.value_changed.connect(self.emit_pose_changed)
        self.y_slider.value_changed.connect(self.emit_pose_changed)
        self.z_slider.value_changed.connect(self.emit_pose_changed)
        self.roll_slider.value_changed.connect(self.emit_pose_changed)
        self.pitch_slider.value_changed.connect(self.emit_pose_changed)
        self.yaw_slider.value_changed.connect(self.emit_pose_changed)
        # Commit timeline selection once per interaction. Connecting the raw
        # valueChanged signal would create a qpos keyframe at every intermediate
        # slider tick while scrubbing from (for example) 0.0 to 0.2 seconds.
        self.time_slider.slider.sliderReleased.connect(
            lambda: self.emit_time_changed(self.time_slider.value())
        )
        self.time_slider.input.editingFinished.connect(
            lambda: self.emit_time_changed(self.time_slider.value())
        )

        # --------------------------------------------------------
        # Trajectory display options
        # --------------------------------------------------------
        self.show_keyframes_box = QCheckBox("Show keyframes")
        self.show_keyframes_box.setChecked(True)
        self.show_keyframes_box.toggled.connect(
            self.keyframes_visibility_changed.emit
        )
        self.view_layout.addWidget(self.show_keyframes_box)

        self.show_lines_box = QCheckBox("Show trajectory lines")
        self.show_lines_box.setChecked(True)
        self.show_lines_box.toggled.connect(self.trajectory_lines_changed.emit)
        self.view_layout.addWidget(self.show_lines_box)

        self.corner_smoothing_slider = InlineLabeledSlider(
            "Smoothing",
            min_value=0.0,
            max_value=1.0,
            initial_value=0.0,
            single_step=0.01,
            decimals=0,
            suffix="%",
            display_scale=100.0,
        )
        smoothing_label_width = self.corner_smoothing_slider.label.sizeHint().width()
        self.corner_smoothing_slider.label.setMinimumWidth(smoothing_label_width + 8)

        # --------------------------------------------------------
        # Keyframe buttons
        # --------------------------------------------------------
        self.add_button = QPushButton("Add Keyframe")
        self.update_button = QPushButton("Update")
        self.delete_button = QPushButton("Delete")
        self.clear_button = QPushButton("Clear Trajectory")

        self.generate_button = QPushButton("Generate / Simulate")
        self.generate_button.setObjectName("generateTrajectoryButton")
        self.generate_button.setToolTip(
            "Generate a sampled robot trajectory from saved keyframes."
        )

        self.add_button.clicked.connect(self.add_keyframe_clicked.emit)
        self.update_button.clicked.connect(self.update_keyframe_clicked.emit)
        self.delete_button.clicked.connect(self.delete_keyframe_clicked.emit)
        self.clear_button.clicked.connect(self.clear_trajectory_clicked.emit)
        self.generate_button.clicked.connect(self.generate_clicked.emit)

        # --------------------------------------------------------
        # Keyframe table
        # --------------------------------------------------------
        self.table = QTableWidget()
        self.table.setMinimumHeight(96)
        self.table.setMaximumHeight(180)
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "time",
            "phase",
            "frame",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        ])
        self.table.setColumnHidden(1, True)
        self.table.cellClicked.connect(self.on_table_cell_clicked)

        self.trajectory_layout.addWidget(QLabel("Trajectory keyframes"))
        self.trajectory_layout.addWidget(self.table)

        self.robot_context_stack = self._make_context_stack()
        self.target_context_stack = self._make_context_stack()
        self.trajectory_context_stack = self._make_context_stack()
        self.timeslice_context_stack = self._make_context_stack()
        self.preview_ik_context_stack = self._make_context_stack()
        self.display_context_stack = self._make_context_stack()
        robot_layout.addWidget(self.robot_context_stack)
        self.target_layout.addWidget(self.target_context_stack)
        robot_layout.addWidget(self.trajectory_context_stack)
        self.trajectory_layout.addWidget(self.timeslice_context_stack)
        self.preview_ik_layout.addWidget(self.preview_ik_context_stack)
        self.view_layout.addWidget(self.display_context_stack)

    def _make_section_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        return panel, layout

    def _make_context_stack(self):
        stack = QStackedWidget()
        stack.empty_widget = QWidget()
        stack.addWidget(stack.empty_widget)
        stack.setVisible(False)
        stack.setMinimumWidth(0)
        stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return stack

    def _set_context_widget(self, stack, widget):
        if widget is None:
            widget = stack.empty_widget
        if stack.indexOf(widget) < 0:
            stack.addWidget(widget)
        stack.setCurrentWidget(widget)
        stack.setVisible(widget is not stack.empty_widget)

    def _emit_setup_import_action(self, index):
        action = self.import_action_box.itemData(index)
        self.import_action_box.setCurrentIndex(-1)
        if action:
            self.setup_import_requested.emit(action)

    def _emit_setup_export_action(self, index):
        action = self.export_action_box.itemData(index)
        self.export_action_box.setCurrentIndex(-1)
        if action:
            self.setup_export_requested.emit(action)

    def set_target_context_widget(self, widget):
        self._set_context_widget(self.target_context_stack, widget)

    def set_robot_context_widget(self, widget):
        self._set_context_widget(self.robot_context_stack, widget)

    def set_trajectory_context_widget(self, widget):
        self._set_context_widget(self.trajectory_context_stack, widget)

    def set_timeslice_context_widget(self, widget):
        self._set_context_widget(self.timeslice_context_stack, widget)

    def set_display_context_widget(self, widget):
        self._set_context_widget(self.display_context_stack, widget)

    def set_preview_ik_context_widget(self, widget):
        self._set_context_widget(self.preview_ik_context_stack, widget)

    def set_joint_editor_widget(self, widget):
        if widget is None:
            widget = self.joint_editor_empty_page
        if self.joint_editor_stack.indexOf(widget) < 0:
            self.joint_editor_stack.addWidget(widget)
        self.joint_editor_stack.setCurrentWidget(widget)
        self.joint_editor_stack.updateGeometry()
        self.editing_mode_stack.updateGeometry()

    def editing_mode(self):
        return (
            "joint_angles"
            if self.editing_mode_bar.currentIndex() == 1
            else "end_effector"
        )

    def set_editing_mode(self, mode):
        index = 1 if mode == "joint_angles" else 0
        if self.editing_mode_bar.currentIndex() == index:
            self.editing_mode_stack.setCurrentIndex(index)
            return
        self.editing_mode_bar.setCurrentIndex(index)

    def _on_editing_mode_changed(self, index):
        if not 0 <= index < self.editing_mode_stack.count():
            return
        self.editing_mode_stack.setCurrentIndex(index)
        self.editing_mode_changed.emit(self.editing_mode())

    def target_context_widget(self):
        return self.target_context_stack.currentWidget()

    def robot_context_widget(self):
        return self.robot_context_stack.currentWidget()

    def trajectory_context_widget(self):
        return self.trajectory_context_stack.currentWidget()

    def timeslice_context_widget(self):
        return self.timeslice_context_stack.currentWidget()

    def display_context_widget(self):
        return self.display_context_stack.currentWidget()

    def preview_ik_context_widget(self):
        return self.preview_ik_context_stack.currentWidget()

    def workflow_sections(self):
        return [
            ("Target", self.target_panel, True),
            ("Editing Mode", self.editing_mode_panel, True),
            ("Planning", self.trajectory_panel, False),
        ]

    def inspector_sections(self):
        return [
            ("IK / Constraints", self.preview_ik_panel, False),
        ]

    def set_frame_names(self, frame_names, preferred=None):
        """Replace target choices without emitting an intermediate selection."""
        names = list(frame_names)
        if not names:
            return
        old = self.frame_box.currentText()
        self.frame_box.blockSignals(True)
        self.frame_box.clear()
        self.frame_box.addItems(names)
        choice = preferred if preferred in names else old if old in names else names[0]
        self.frame_box.setCurrentText(choice)
        self.frame_box.blockSignals(False)
        self.frame_name_changed.emit(choice)

    def add_model(self, key, display_name, select=True):
        if not hasattr(self, "model_box"):
            return
        index = self.model_box.findData(key)
        if index < 0:
            self.model_box.addItem(display_name, key)
            index = self.model_box.findData(key)
        if select and index >= 0:
            self.model_box.setCurrentIndex(index)

    def emit_pose_changed(self):
        if self._suppress_pose_changed:
            return

        self.pose_changed.emit(
            self.x_slider.value(),
            self.y_slider.value(),
            self.z_slider.value(),
            self.roll_slider.value(),
            self.pitch_slider.value(),
            self.yaw_slider.value(),
        )

    def emit_time_changed(self, value):
        if not self._suppress_pose_changed:
            self.time_changed.emit(float(value))

    def current_frame(self):
        """
        Convert GUI values into a TargetFrame object.
        """

        return TargetFrame(
            time=self.time_slider.value(),
            phase=self.phase_box.currentText(),
            frame_name=self.frame_box.currentText(),
            x=self.x_slider.value(),
            y=self.y_slider.value(),
            z=self.z_slider.value(),
            roll=self.roll_slider.value(),
            pitch=self.pitch_slider.value(),
            yaw=self.yaw_slider.value(),
        )

    def show_trajectory_lines(self):
        return self.show_lines_box.isChecked()

    def show_keyframes(self):
        return self.show_keyframes_box.isChecked()

    def corner_smoothing(self):
        return max(0.0, min(1.0, self.corner_smoothing_slider.value()))

    def set_from_frame(self, frame):
        """
        Load a keyframe into the editor.
        """

        self._suppress_pose_changed = True

        try:
            self.time_slider.set_value(frame.time)
            self.x_slider.set_value(frame.x)
            self.y_slider.set_value(frame.y)
            self.z_slider.set_value(frame.z)
            self.roll_slider.set_value(frame.roll)
            self.pitch_slider.set_value(frame.pitch)
            self.yaw_slider.set_value(frame.yaw)

            self.phase_box.setCurrentText(frame.phase)
            previous_block_state = self.frame_box.blockSignals(True)
            self.frame_box.setCurrentText(frame.frame_name)
            self.frame_box.blockSignals(previous_block_state)
        finally:
            self._suppress_pose_changed = False

        self.time_changed.emit(self.time_slider.value())
        self.emit_pose_changed()

    def set_position_from_viewer(self, x, z, emit_pose_changed=True):
        """
        Called when user drags target frame in the viewer.
        """

        self.set_position_values(x=x, z=z, emit_pose_changed=emit_pose_changed)

    def set_position_values(
        self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None,
        emit_pose_changed=True,
    ):
        """
        Set target position controls, optionally preserving untouched axes.
        """

        self._suppress_pose_changed = True

        try:
            if x is not None:
                self.x_slider.set_value(x)
            if y is not None:
                self.y_slider.set_value(y)
            if z is not None:
                self.z_slider.set_value(z)
            if roll is not None:
                self.roll_slider.set_value(roll)
            if pitch is not None:
                self.pitch_slider.set_value(pitch)
            if yaw is not None:
                self.yaw_slider.set_value(yaw)
        finally:
            self._suppress_pose_changed = False

        if emit_pose_changed:
            self.emit_pose_changed()

    def refresh_table(self, trajectory):
        """
        Display trajectory array in table.
        """

        self.table.setRowCount(len(trajectory.frames))

        for row, frame in enumerate(trajectory.frames):
            values = [
                f"{frame.time:.2f}",
                frame.phase,
                frame.frame_name,
                f"{frame.x:.2f}",
                f"{frame.y:.2f}",
                f"{frame.z:.2f}",
                f"{frame.roll:.2f}",
                f"{frame.pitch:.2f}",
                f"{frame.yaw:.2f}",
            ]

            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def selected_row(self):
        return self.table.currentRow()

    def on_table_cell_clicked(self, row, col):
        self.keyframe_selected.emit(row)
