"""MoveIt-style live robot editor wrapped around GhostGUI's OpenGL canvas."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from application.paths import (
    QPOS_CSV_DIR,
    TRAJECTORY_CSV_DIR,
)
from application.background_jobs import SerializedBackgroundJobs
from application.playback import PlaybackClock
from application.csv_io import (
    TrajectoryExport,
    read_qpos_csv,
    read_trajectory_csv,
    write_qpos_csv,
    write_trajectory_csv,
)
from core.models import (
    RobotStateTimeline,
    TrajectoryGhostRenderer,
    interpolate_qpos,
)
from core.ik import CollisionAwareIKSolver, CollisionChecker
from core.trajectory import quat_to_rpy, rpy_to_quat
from gui.file_selection import SynchronousFileSelectionStage
from gui.viewers import ik_panels
from gui.viewers.robot_canvas_3d import RobotCanvas3D
from core.ik import (
    FootLockTask,
    JointRegularizationTask,
    PostureTask,
    RootPoseTask,
)
from .widgets.compact import compact_combo as _compact_combo
from .widgets.compact import compact_spinbox as _compact_spinbox
from .widgets.joint_controls import JointControl
from .widgets.status import StatusValueLabel
from .widgets.timeline import TimesliceSlider


FRAME_BINDINGS = {
    "pelvis": ("body", "robot/pelvis"),
    "torso": ("body", "robot/torso_link"),
    "left_foot": ("site", "robot/left_foot"),
    "right_foot": ("site", "robot/right_foot"),
    "left_hand": ("site", "robot/left_palm"),
    "right_hand": ("site", "robot/right_palm"),
}
REVERSE_BINDINGS = {value: key for key, value in FRAME_BINDINGS.items()}


@dataclass(frozen=True)
class PreviewPathValidation:
    ok: bool
    message: str
    failed_index: int | None = None


class RobotViewer3D(QWidget):
    """Live FK/IK/trajectory viewer. It preserves the old canvas contract."""

    target_dragged = Signal(float, float)
    target_pose_dragged = Signal(float, float, float, float, float, float)
    target_pose_drag_finished = Signal(float, float, float, float, float, float)
    target_frame_changed = Signal(str)
    preview_cancelled = Signal()
    trajectory_csv_loaded = Signal(str)
    generate_requested = Signal()
    clear_trajectory_requested = Signal()
    timeslice_time_changed = Signal(float)
    timeslice_preview_time_changed = Signal(float)
    timeline_duration_changed = Signal(float)
    accept_timeslice_requested = Signal()
    delete_timeslice_requested = Signal()
    history_action_finished = Signal(str)
    playback_state_changed = Signal(bool)

    @property
    def _playback_last_tick(self):
        """Compatibility view of the extracted playback clock."""
        return self.playback_clock.last_tick

    @_playback_last_tick.setter
    def _playback_last_tick(self, value):
        self.playback_clock.last_tick = value

    def __init__(
        self,
        robot_model=None,
        error=None,
        background_jobs=None,
        file_selection_stage=None,
    ):
        super().__init__()
        self.robot_model = robot_model
        self.frame_bindings = (
            dict(robot_model.logical_frame_bindings)
            if robot_model is not None and hasattr(robot_model, "logical_frame_bindings")
            else dict(FRAME_BINDINGS)
        )
        self.reverse_bindings = {value: key for key, value in self.frame_bindings.items()}
        self.committed_state = robot_model.create_state() if robot_model else None
        self.preview_state = robot_model.create_state() if robot_model else None
        # Compatibility alias: external readers historically used robot_state.
        # It now always means the timeline-backed committed state.
        self.robot_state = self.committed_state
        self.playback_state = robot_model.create_state() if robot_model else None
        self.preview_active = False
        self.current_time = 0.0
        self.display_time = 0.0
        self.timeline_duration = 5.0
        self.state_timeline = (
            RobotStateTimeline(robot_model, initial_qpos=self.robot_state.get_qpos())
            if robot_model else None
        )
        self.ghost_renderer = TrajectoryGhostRenderer(robot_model) if robot_model else None
        self.collision_checker = CollisionChecker(robot_model) if robot_model else None
        self.collision_solver = (
            CollisionAwareIKSolver(
                robot_model,
                self.collision_checker,
                orientation_weight=(
                    0.0 if getattr(robot_model, "model_type", None) == "quadruped"
                    else 0.25
                ),
            )
            if robot_model else None
        )
        self.robot_trajectory = []
        self.robot_trajectory_times = []
        self._prompt_trajectory_import_dt_on_load = False
        self._background_trajectory_postprocess_requested = False
        self.csv_file_selection_stage = (
            file_selection_stage
            if file_selection_stage is not None
            else SynchronousFileSelectionStage(self)
        )
        self._owns_file_selection_stage = file_selection_stage is None
        self.background_jobs = (
            background_jobs
            if background_jobs is not None
            else SerializedBackgroundJobs(self)
        )
        self._owns_background_jobs = background_jobs is None
        self._shutdown = False
        self.ghost_trajectory = []
        self.ghost_source = None
        self.joint_controls = {}
        self.ik_influence_controls = {}
        self.ik_joint_weights = {
            name: 1.0 for name in (
                robot_model.get_joint_names() if robot_model else []
            )
        }
        self.preview_reference_qpos = None
        self.foot_lock_targets = {}
        self.root_lock_target = None
        self._last_ik_status = None
        self._syncing_target = False
        self.playback_clock = PlaybackClock()
        self._resume_playback_after_scrub = False
        self._pending_scrub_preview_time = None
        self.canvas = RobotCanvas3D()
        self.canvas.geometry_progress.connect(self._on_geometry_progress)
        self.canvas.target_dragged.connect(self.target_dragged.emit)
        self.canvas.target_transform_dragged.connect(self._on_transform_moved)
        self.canvas.transform_drag_finished.connect(
            self._on_transform_drag_finished
        )
        self.canvas.transform_drag_cancel_requested.connect(
            self._on_transform_cancel_requested
        )
        self.canvas.gizmo_mode_changed.connect(self._on_gizmo_mode_changed)
        self.canvas.body_double_clicked.connect(self._on_body_double_clicked)
        self.last_valid_target_position = None
        self.last_valid_target_quaternion = None
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._advance_playback)
        self.scrub_preview_timer = QTimer(self)
        self.scrub_preview_timer.setSingleShot(True)
        self.scrub_preview_timer.setInterval(16)
        self.scrub_preview_timer.timeout.connect(
            self._flush_pending_scrub_preview
        )
        self._build_ui(error)
        if self.robot_state:
            self.canvas.set_robot_states(
                self.committed_state, self.preview_state, self.ghost_renderer
            )
            self._set_target_to_selected_pose()

    def shutdown(self):
        """Stop timers, selectors, jobs, and GL resources idempotently."""
        if self._shutdown:
            return
        self._shutdown = True
        self.play_timer.stop()
        self.scrub_preview_timer.stop()
        self.playback_clock.stop()
        self._pending_scrub_preview_time = None
        if self._owns_file_selection_stage:
            self.csv_file_selection_stage.cancel()
        if self._owns_background_jobs:
            self.background_jobs.shutdown()
        self.canvas.shutdown()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def _build_ui(self, error):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar_controls = QWidget()

        self.robot_context_panel = QWidget()
        model_layout = QVBoxLayout(self.robot_context_panel)
        model_layout.setContentsMargins(6, 6, 6, 6)
        self.status_label = StatusValueLabel(error or "Robot model loaded; FK ready.")
        self.status_label.setWordWrap(True)
        self.timeline_state_label = StatusValueLabel("3D state time: 0.00 s")

        self.model_colors_box = QCheckBox("Use model colors")
        self.model_colors_box.setChecked(True)
        self.model_colors_box.toggled.connect(self.canvas.set_use_model_colors)
        display_panel = QWidget()
        self.display_layout = QVBoxLayout(display_panel)
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        self.display_layout.addWidget(self.model_colors_box)
        self.show_ghosts = QCheckBox("Show playback poses")
        self.show_ghosts.toggled.connect(self._sync_playback_pose_ghosts)
        self.display_layout.addWidget(self.show_ghosts)
        self.display_context_panel = display_panel
        if self.robot_model:
            texture_warnings = self.robot_model.get_visual_texture_warnings()
            if texture_warnings:
                warning = QLabel("; ".join(texture_warnings))
                warning.setWordWrap(True)
                self.display_layout.addWidget(warning)

        self.trajectory_context_panel = QWidget()
        trajectory_context_layout = QVBoxLayout(self.trajectory_context_panel)
        trajectory_context_layout.setContentsMargins(6, 6, 6, 6)
        self.load_qpos_button = QPushButton("Load")
        self.load_trajectory_button = QPushButton("Load")
        self.save_qpos_button = QPushButton("Save")
        self.save_trajectory_button = QPushButton("Save")
        for button in (
            self.load_qpos_button,
            self.load_trajectory_button,
            self.save_qpos_button,
            self.save_trajectory_button,
        ):
            button.setMinimumWidth(0)
            button.setMaximumWidth(104)
        self.trajectory_import_dt = QDoubleSpinBox()
        self.trajectory_import_dt.setDecimals(2)
        self.trajectory_import_dt.setRange(0.01, 10.0)
        self.trajectory_import_dt.setSingleStep(0.01)
        self.trajectory_import_dt.setValue(0.10)
        self.trajectory_import_dt.setSuffix(" s")
        self.trajectory_import_dt.setMaximumWidth(76)
        self.trajectory_import_dt.setVisible(False)
        self.load_qpos_button.clicked.connect(self.choose_qpos_csv)
        self.load_trajectory_button.clicked.connect(self.choose_trajectory_csv)
        self.save_qpos_button.clicked.connect(self.choose_qpos_save_path)
        self.save_trajectory_button.clicked.connect(
            self.choose_trajectory_save_path
        )
        self.trajectory_csv_group = QGroupBox("Trajectory CSV")
        trajectory_csv_layout = QHBoxLayout(self.trajectory_csv_group)
        trajectory_csv_layout.setContentsMargins(6, 6, 6, 6)
        trajectory_csv_layout.setSpacing(4)
        trajectory_csv_layout.addWidget(self.load_trajectory_button)
        trajectory_csv_layout.addWidget(self.save_trajectory_button)
        self.trajectory_csv_group.setVisible(False)
        self.qpos_csv_group = QGroupBox("Qpos CSV")
        qpos_csv_layout = QHBoxLayout(self.qpos_csv_group)
        qpos_csv_layout.setContentsMargins(6, 6, 6, 6)
        qpos_csv_layout.setSpacing(4)
        qpos_csv_layout.addWidget(self.load_qpos_button)
        qpos_csv_layout.addWidget(self.save_qpos_button)
        self.qpos_csv_group.setVisible(False)

        self.target_context_panel = QWidget()
        target_layout = QFormLayout(self.target_context_panel)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.target_box = QComboBox()
        _compact_combo(self.target_box, minimum_chars=12)
        if self.robot_model:
            for frame_name, (kind, name) in self.frame_bindings.items():
                try:
                    self.robot_state.resolve_object(name, kind)
                except KeyError:
                    continue
                self.target_box.addItem(frame_name.replace("_", " "), (kind, name))
            known = {self.target_box.itemData(i) for i in range(self.target_box.count())}
            for kind, names in (("site", self.robot_model.site_names),
                                ("body", self.robot_model.body_names)):
                for name in names:
                    if (kind, name) not in known and name != "world":
                        self.target_box.addItem(f"{kind}: {name}", (kind, name))
        self.target_box.currentIndexChanged.connect(self._target_selected)
        target_layout.addRow("Advanced target", self.target_box)
        self.root_pose_label = StatusValueLabel()
        self.root_pose_label.setWordWrap(True)

        self.preview_ik_context_panel = QWidget()
        preview_ik_layout = QVBoxLayout(self.preview_ik_context_panel)
        preview_ik_layout.setContentsMargins(4, 4, 4, 4)
        preview_ik_layout.setSpacing(4)

        self.timeslice_context_panel = QWidget()
        self.timeslice_context_layout = QFormLayout(self.timeslice_context_panel)
        self.timeslice_context_layout.setContentsMargins(6, 6, 6, 6)
        self.timeslice_context_layout.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.collision_substeps = QSpinBox()
        _compact_spinbox(self.collision_substeps)
        self.collision_substeps.setRange(1, 32)
        self.collision_substeps.setValue(8)
        self.collision_substeps.valueChanged.connect(
            self._set_collision_substeps
        )
        self.collision_substeps_label = QLabel("Collision substeps")
        self.timeslice_context_layout.addRow(
            self.collision_substeps_label, self.collision_substeps
        )
        self.playback_speed = QDoubleSpinBox()
        _compact_spinbox(self.playback_speed)
        self.playback_speed.setRange(0.10, 4.00)
        self.playback_speed.setDecimals(2)
        self.playback_speed.setSingleStep(0.25)
        self.playback_speed.setValue(1.00)
        self.playback_speed.setSuffix("×")
        self.playback_speed_label = QLabel("Playback speed")
        self.timeslice_context_layout.addRow(
            self.playback_speed_label, self.playback_speed
        )
        self.generate_button = QPushButton("Demo trajectory")
        self.generate_button.clicked.connect(self.generate_demo_trajectory)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.ghost_stride = QSpinBox()
        _compact_spinbox(self.ghost_stride)
        self.ghost_stride.setRange(1, 100)
        self.ghost_stride.setValue(8)
        self.ghost_stride.valueChanged.connect(self._rebuild_ghosts)
        self.ghost_alpha = QDoubleSpinBox()
        _compact_spinbox(self.ghost_alpha)
        self.ghost_alpha.setRange(0.02, 0.8)
        self.ghost_alpha.setSingleStep(0.05)
        self.ghost_alpha.setValue(0.16)
        self.ghost_alpha.valueChanged.connect(self._update_ghost_options)
        self.preview_alpha = QDoubleSpinBox()
        _compact_spinbox(self.preview_alpha)
        self.preview_alpha.setRange(0.1, 1.0)
        self.preview_alpha.setSingleStep(0.05)
        self.preview_alpha.setValue(0.65)
        self.preview_alpha.valueChanged.connect(self.canvas.set_preview_alpha)
        self.ghost_stride_label = QLabel("Playback spacing")
        self.ghost_alpha_label = QLabel("Playback opacity")
        self.preview_alpha_label = QLabel("Preview opacity")
        self.timeslice_context_layout.addRow(
            self.ghost_stride_label, self.ghost_stride
        )
        self.timeslice_context_layout.addRow(
            self.ghost_alpha_label, self.ghost_alpha
        )
        self.timeslice_context_layout.addRow(
            self.preview_alpha_label, self.preview_alpha
        )

        editor_tabs = QTabWidget()
        editor_tabs.setObjectName("ikEditorTabs")
        editor_tabs.setMinimumWidth(0)
        self.ik_editor_tabs = editor_tabs
        joint_group = QWidget()
        joint_group.setObjectName("ikEditorTabContent")
        joint_layout = QVBoxLayout(joint_group)
        joint_layout.setContentsMargins(6, 6, 6, 6)
        if self.robot_state:
            for name in self.robot_state.get_joint_names():
                control = JointControl(
                    name,
                    self.robot_state.get_joint_limits(name),
                    self.robot_state.get_joint_value(name),
                )
                control.value_changed.connect(self._joint_changed)
                self.joint_controls[name] = control
                joint_layout.addWidget(control)
        else:
            joint_layout.addWidget(QLabel("Joint controls unavailable."))
        joint_layout.addStretch()
        scroll = QScrollArea()
        scroll.setObjectName("jointEditorScroll")
        scroll.viewport().setObjectName("ikEditorViewport")
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(0)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(joint_group)
        self.joint_angles_page = scroll
        editor_tabs.addTab(self._build_ik_tasks_widget(), "Tasks")
        editor_tabs.addTab(self._build_joint_weights_widget(), "Weights")
        editor_tabs.addTab(self._build_solver_widget(), "Solver")
        editor_tabs.setCurrentIndex(0)
        preview_ik_layout.addWidget(editor_tabs)
        root.addWidget(self._build_canvas_workspace(), stretch=1)
        root.addWidget(self._build_timeslice_editor())

        enabled = self.robot_state is not None
        self.load_qpos_button.setEnabled(enabled)
        self.load_trajectory_button.setEnabled(enabled)
        self.trajectory_import_dt.setEnabled(enabled)
        self.save_qpos_button.setEnabled(enabled)
        self.save_trajectory_button.setEnabled(enabled)
        self.target_context_panel.setEnabled(enabled)
        self.timeslice_context_panel.setEnabled(enabled)
        self.preview_ik_context_panel.setEnabled(enabled)
        self.timeslice_editor.setEnabled(enabled)

    def _build_canvas_workspace(self):
        self.canvas_workspace = QWidget()
        layout = QGridLayout(self.canvas_workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 0, 0)
        return self.canvas_workspace

    def _build_timeslice_editor(self):
        self.timeslice_editor = QWidget()
        self.timeslice_layout = QVBoxLayout(self.timeslice_editor)
        self.timeslice_layout.setContentsMargins(8, 4, 8, 4)
        self.timeslice_layout.setSpacing(3)

        self.timeslice_label = QLabel("Time")

        self.timeslice_slider = TimesliceSlider(Qt.Orientation.Horizontal)
        self.timeslice_slider.setRange(0, 500)
        self.timeslice_slider.setValue(0)
        self.timeslice_slider.marker_activated.connect(
            self._emit_timeslice_marker_time
        )
        self.timeslice_slider.time_activated.connect(
            self._emit_timeslice_marker_time
        )
        self.timeslice_slider.sliderMoved.connect(
            self._preview_timeslice_slider_value
        )
        self.timeslice_slider.sliderPressed.connect(
            self._begin_timeslice_scrub
        )
        self.timeslice_slider.sliderReleased.connect(
            self._emit_timeslice_slider_time
        )

        self.timeslice_time_input = QDoubleSpinBox()
        _compact_spinbox(self.timeslice_time_input, width=72)
        self.timeslice_time_input.setRange(0.0, self.timeline_duration)
        self.timeslice_time_input.setDecimals(2)
        self.timeslice_time_input.setSingleStep(0.01)
        self.timeslice_time_input.setSuffix(" s")
        self.timeslice_time_input.editingFinished.connect(
            self._emit_timeslice_input_time
        )
        self.timeslice_frame_readout = QLabel("Frame —")
        self.timeslice_frame_readout.setMinimumWidth(82)
        self.timeslice_frame_readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.timeslice_frame_readout.setAccessibleName("Playback frame")

        self.delete_timeslice_button = QPushButton("Delete Keyframe")
        self.delete_timeslice_button.clicked.connect(self.delete_timeslice)
        self.timeslice_step_label = QLabel("Keyframe interval")
        self.timeslice_step_input = QDoubleSpinBox()
        _compact_spinbox(self.timeslice_step_input, width=72)
        self.timeslice_step_input.setRange(0.01, 5.0)
        self.timeslice_step_input.setDecimals(2)
        self.timeslice_step_input.setSingleStep(0.01)
        self.timeslice_step_input.setValue(0.10)
        self.timeslice_step_input.setSuffix(" s")

        self.timeslice_duration_label = QLabel("Max time")
        self.timeslice_duration_input = QDoubleSpinBox()
        _compact_spinbox(self.timeslice_duration_input, width=72)
        self.timeslice_duration_input.setRange(0.10, 120.0)
        self.timeslice_duration_input.setDecimals(2)
        self.timeslice_duration_input.setSingleStep(0.10)
        self.timeslice_duration_input.setValue(self.timeline_duration)
        self.timeslice_duration_input.setSuffix(" s")
        self.timeslice_duration_input.valueChanged.connect(
            self._on_timeslice_duration_changed
        )

        self.timeslice_time_row = QHBoxLayout()
        self.timeslice_time_row.setContentsMargins(0, 0, 0, 0)
        self.timeslice_time_row.setSpacing(8)
        self.timeslice_time_row.addWidget(self.timeslice_label)
        self.timeslice_time_row.addWidget(self.timeslice_slider, stretch=1)
        self.timeslice_time_row.addWidget(self.timeslice_time_input)
        self.timeslice_time_row.addWidget(self.timeslice_frame_readout)

        self.timeslice_context_layout.addRow(
            self.timeslice_step_label, self.timeslice_step_input
        )
        self.timeslice_context_layout.addRow(
            self.timeslice_duration_label, self.timeslice_duration_input
        )

        self.timeslice_timeline_group = QGroupBox("Timeline")
        self.timeslice_timeline_layout = QHBoxLayout(self.timeslice_timeline_group)
        self.timeslice_timeline_layout.setContentsMargins(6, 6, 6, 6)
        self.timeslice_timeline_layout.setSpacing(8)
        self.timeslice_scrubber_layout = QVBoxLayout()
        self.timeslice_scrubber_layout.setContentsMargins(0, 0, 0, 0)
        self.timeslice_scrubber_layout.setSpacing(3)
        self.timeslice_scrubber_layout.addLayout(self.timeslice_time_row)
        self.timeslice_action_row = QVBoxLayout()
        self.timeslice_action_row.setContentsMargins(0, 0, 0, 0)
        self.timeslice_action_row.setSpacing(4)
        self.timeslice_action_row.addWidget(self.delete_timeslice_button)
        self.timeslice_timeline_layout.addLayout(
            self.timeslice_scrubber_layout, stretch=1
        )
        self.timeslice_timeline_layout.addLayout(self.timeslice_action_row)

        self.timeslice_layout.addWidget(self.timeslice_timeline_group)
        return self.timeslice_editor

    def set_defined_timeslices(self, times):
        self.timeslice_slider.set_defined_times(times)

    def set_smoothing_widget(self, widget):
        if widget is None:
            return
        if (
            widget.parent() is self.timeslice_context_panel
            and getattr(self, "smoothing_widget", None) is widget
        ):
            return
        if widget.parent() is not self.timeslice_context_panel:
            widget.setParent(self.timeslice_context_panel)
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            widget.sizePolicy().verticalPolicy(),
        )
        self.smoothing_widget = widget
        self.timeslice_context_layout.insertRow(0, widget)

    def set_trajectory_display_widgets(self, keyframes_widget, lines_widget):
        widgets = [
            widget for widget in (keyframes_widget, lines_widget)
            if widget is not None
        ]
        if not widgets:
            return
        for widget in widgets:
            if widget.parent() is not self.display_context_panel:
                widget.setParent(self.display_context_panel)
        insert_index = 1
        if keyframes_widget is not None:
            self.display_layout.insertWidget(insert_index, keyframes_widget)
            insert_index += 1
        if lines_widget is not None:
            self.display_layout.insertWidget(insert_index, lines_widget)

    def set_trajectory_lines_widget(self, widget):
        self.set_trajectory_display_widgets(None, widget)

    def _set_timeslice_widgets(self, time):
        raw_time = int(round(float(time) * 100.0))
        raw_time = max(self.timeslice_slider.minimum(), raw_time)
        raw_time = min(self.timeslice_slider.maximum(), raw_time)
        was_blocked = self.timeslice_slider.blockSignals(True)
        self.timeslice_slider.setValue(raw_time)
        self.timeslice_slider.blockSignals(was_blocked)
        was_blocked = self.timeslice_time_input.blockSignals(True)
        self.timeslice_time_input.setValue(raw_time / 100.0)
        self.timeslice_time_input.blockSignals(was_blocked)

    def _preview_timeslice_slider_value(self, raw_value):
        time = raw_value / 100.0
        if not self.scrub_preview_timer.isActive():
            self.preview_trajectory_time(time, emit_time_signal=True)
            self.scrub_preview_timer.start()
            return

        self._pending_scrub_preview_time = time
        was_blocked = self.timeslice_time_input.blockSignals(True)
        self.timeslice_time_input.setValue(time)
        self.timeslice_time_input.blockSignals(was_blocked)

    def _flush_pending_scrub_preview(self):
        time = self._pending_scrub_preview_time
        self._pending_scrub_preview_time = None
        if time is None:
            return
        self.preview_trajectory_time(time, emit_time_signal=True)
        self.scrub_preview_timer.start()

    def _begin_timeslice_scrub(self):
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        self._resume_playback_after_scrub = self.play_timer.isActive()
        if self._resume_playback_after_scrub:
            self.pause_playback(commit_time=False)

    def _emit_timeslice_slider_time(self):
        time = self.timeslice_slider.value() / 100.0
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        self.preview_trajectory_time(time, emit_time_signal=True)
        self.timeslice_time_changed.emit(time)
        resume_playback = self._resume_playback_after_scrub
        self._resume_playback_after_scrub = False
        if resume_playback:
            self.start_playback()

    def _emit_timeslice_marker_time(self, time):
        was_playing = self.play_timer.isActive()
        if was_playing:
            self.pause_playback(commit_time=False)
        self._set_timeslice_widgets(time)
        self.timeslice_time_changed.emit(float(time))
        if was_playing:
            self.start_playback()

    def _emit_timeslice_input_time(self):
        time = self.timeslice_time_input.value()
        was_playing = self.play_timer.isActive()
        if was_playing:
            self.pause_playback(commit_time=False)
        self._set_timeslice_widgets(time)
        self.timeslice_time_changed.emit(time)
        if was_playing:
            self.start_playback()

    def timeslice_step(self):
        return float(self.timeslice_step_input.value())

    def _on_timeslice_duration_changed(self, duration):
        self.set_timeline_duration(duration)

    def set_timeline_duration(self, duration, emit_signal=True):
        duration = max(0.10, float(duration))
        self.timeline_duration = duration
        raw_max = int(round(duration * 100.0))

        was_blocked = self.timeslice_duration_input.blockSignals(True)
        self.timeslice_duration_input.setValue(duration)
        self.timeslice_duration_input.blockSignals(was_blocked)

        self.timeslice_slider.setMaximum(raw_max)
        self.timeslice_time_input.setMaximum(duration)
        self.timeslice_step_input.setMaximum(duration)
        if self.current_time > duration:
            if self.state_timeline:
                self.current_time = self.state_timeline.time_key(duration)
            else:
                self.current_time = duration
        self.display_time = min(self.display_time, duration)
        self._set_timeslice_widgets(self.display_time)
        self._update_frame_readout(self.display_time)
        if emit_signal:
            self.timeline_duration_changed.emit(duration)

    def next_timeslice_time(self, time=None):
        base_time = self.current_time if time is None else float(time)
        raw_time = int(round((base_time + self.timeslice_step()) * 100.0))
        raw_time = max(self.timeslice_slider.minimum(), raw_time)
        raw_time = min(self.timeslice_slider.maximum(), raw_time)
        return raw_time / 100.0

    def accept_timeslice(self):
        # Playback and live scrubbing intentionally move display_time without
        # creating an editable state. Commit Keyframe must first choose one authoritative
        # time so its qpos state and logical-target snapshot cannot be written
        # at two different positions on the timeline.
        slice_time = max(0.0, min(float(self.display_time), self.timeline_duration))
        if self.state_timeline is not None:
            slice_time = self.state_timeline.time_key(slice_time)

        if self.play_timer.isActive():
            self.pause_playback(commit_time=False)

        if self.preview_active:
            # Preserve the active preview while moving it to the visible time.
            # Calling set_current_time() here would discard that preview.
            self.current_time = slice_time
            self.display_time = slice_time
            self._set_timeslice_widgets(slice_time)
            self.timeslice_preview_time_changed.emit(slice_time)
            if not self.accept_preview(emit_pose_finished=False):
                return
        elif abs(slice_time - self.current_time) > 1e-9:
            # With no edit preview to preserve, load the visible trajectory pose
            # as the committed state before taking the logical-target snapshot.
            self.timeslice_time_changed.emit(slice_time)

        self.accept_timeslice_requested.emit()

    def delete_timeslice(self):
        self.delete_timeslice_requested.emit()

    def sidebar_context_widget(self):
        return self.sidebar_controls

    def robot_context_widget(self):
        return self.robot_context_panel

    def target_context_widget(self):
        return self.target_context_panel

    def trajectory_context_widget(self):
        return None

    def consume_trajectory_import_dt_prompt_request(self):
        requested = bool(
            getattr(self, "_prompt_trajectory_import_dt_on_load", False)
        )
        self._prompt_trajectory_import_dt_on_load = False
        return requested

    def consume_background_trajectory_postprocess_request(self):
        requested = bool(self._background_trajectory_postprocess_requested)
        self._background_trajectory_postprocess_requested = False
        return requested

    def timeslice_context_widget(self):
        return self.timeslice_context_panel

    def display_context_widget(self):
        return self.display_context_panel

    def preview_ik_context_widget(self):
        return self.preview_ik_context_panel

    def joint_editor_widget(self):
        return self.joint_angles_page

    def _make_ik_scroll_area(self, content):
        return ik_panels.make_ik_scroll_area(content)

    def _build_solver_widget(self):
        return ik_panels.build_solver_widget(self)

    def _build_ik_tasks_widget(self):
        return ik_panels.build_ik_tasks_widget(self)

    def _build_joint_weights_widget(self):
        return ik_panels.build_joint_weights_widget(self)

    def _on_geometry_progress(self, complete, total):
        if total <= 0:
            return
        if complete < total:
            self.status_label.setText(
                f"Preparing 3D geometry… {complete}/{total}"
            )
        else:
            self.status_label.setText("3D geometry ready.")

    def update_scene(
        self,
        trajectory,
        active_frame=None,
        show_trajectory_lines=True,
        trajectory_smoothing=0.0,
        show_keyframes=True,
    ):
        # The live gizmo owns its quaternion while editing. Ordinary status
        # refreshes must not reset an in-progress ring rotation from controls.
        self.canvas.update_scene(
            trajectory,
            None,
            show_trajectory_lines=show_trajectory_lines,
            trajectory_smoothing=trajectory_smoothing,
            show_keyframes=show_keyframes,
        )
        if active_frame is not None:
            binding = self.frame_bindings.get(active_frame.frame_name)
            if binding is not None:
                self.select_target(*binding, emit=False)

    def select_target(self, kind, name, emit=False):
        for index in range(self.target_box.count()):
            if self.target_box.itemData(index) == (kind, name):
                self._syncing_target = not emit
                self.target_box.setCurrentIndex(index)
                self._syncing_target = False
                self._set_canvas_selected_target(kind, name)
                return True
        return False

    def _selected_target(self):
        data = self.target_box.currentData()
        return tuple(data) if data else (None, None)

    def _target_selected(self):
        if not self.robot_state:
            return
        kind, name = self._selected_target()
        self._set_target_to_selected_pose()
        frame_name = self.reverse_bindings.get((kind, name))
        if frame_name and not self._syncing_target:
            self.target_frame_changed.emit(frame_name)

    def _set_target_to_selected_pose(self):
        kind, name = self._selected_target()
        if not name:
            return
        try:
            state = self.preview_state if self.preview_active else self.committed_state
            position, quaternion = state.get_body_pose(name, kind)
        except KeyError as exc:
            self.status_label.setText(str(exc))
            return
        self._set_canvas_selected_target(kind, name)
        self.canvas.set_target_pose(position, quaternion)
        self.last_valid_target_position = position.copy()
        self.last_valid_target_quaternion = quaternion.copy()
        self._update_root_pose_label()

    def _set_canvas_selected_target(self, kind, name):
        if not name or self.committed_state is None:
            self.canvas.set_selected_target()
            return
        try:
            resolved_kind, object_id = self.committed_state.resolve_object(name, kind)
        except KeyError:
            self.canvas.set_selected_target()
            return
        if resolved_kind == "site":
            owner_body_id = int(self.robot_model.mj_model.site_bodyid[object_id])
        else:
            owner_body_id = int(object_id)
        self.canvas.set_selected_target(resolved_kind, name, owner_body_id)

    def preview_target_pose(self, frame_name, position, quaternion):
        binding = self.frame_bindings.get(frame_name)
        if binding is None:
            self.status_label.setText(
                f"Frame {frame_name!r} has no editable 3D body/site target."
            )
            return False
        if not self.select_target(*binding, emit=False):
            self.status_label.setText(
                f"Frame {frame_name!r} is not selectable in the 3D viewer."
            )
            return False
        if self.last_valid_target_position is None:
            self._set_target_to_selected_pose()
        self._on_transform_moved(position, quaternion)
        return True

    def _joint_changed(self, name, value):
        self.begin_preview()
        self.preview_state.set_joint_value(name, value)
        self._set_target_to_selected_pose()
        collisions = self._update_preview_collisions()
        if self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self.target_pose_dragged.emit(
                *map(float, self.last_valid_target_position),
                roll,
                pitch,
                yaw,
            )
        if collisions:
            names = ", ".join(
                f"{item.geom1} ↔ {item.geom2}" for item in collisions[:2]
            )
            self.status_label.setText(
                f"Collision warning: {names}; "
                f"Preview FK: {name} = {value:+.3f} rad; "
                "adjust the pose before Commit Keyframe"
            )
        else:
            self.status_label.setText(
                f"Preview FK: {name} = {value:+.3f} rad; "
                "use Commit Keyframe to save"
            )

    def _update_preview_collisions(self, collisions=None):
        if collisions is None:
            collisions = (
                self.collision_checker.get_collisions(self.preview_state)
                if self.collision_checker and self.preview_state else []
            )
        collisions = list(collisions)
        self.canvas.set_preview_collisions(collisions)
        return collisions

    def _sync_joint_controls(self, state=None):
        if state is None:
            state = self.preview_state if self.preview_active else self.committed_state
        for name, control in self.joint_controls.items():
            control.set_value(state.get_joint_value(name))

    def _ik_influence_changed(self, name, value):
        self.ik_joint_weights[name] = float(value)
        state = "locked" if value <= 1e-9 else f"influence {value:.2f}"
        self.status_label.setText(f"IK joint {name}: {state}")

    def _set_all_ik_influences(self, selector):
        for name, control in self.ik_influence_controls.items():
            value = float(selector(name))
            self.ik_joint_weights[name] = value
            control.set_value(value)

    def apply_ik_preset(self):
        preset = self.ik_preset_box.currentText()
        if preset == "All joints normal":
            self._set_all_ik_influences(lambda name: 1.0)
        elif preset == "Root locked":
            # Floating roots are already excluded from limb IK. Keep actuated
            # joints normal and make that hard-lock explicit in status.
            self._set_all_ik_influences(lambda name: 1.0)
        elif preset == "Upper body only":
            tokens = ("waist", "shoulder", "elbow", "wrist", "arm")
            self._set_all_ik_influences(
                lambda name: 1.0 if any(token in name.lower() for token in tokens) else 0.0
            )
        elif preset in ("Legs only", "Quadruped legs only"):
            tokens = ("hip", "thigh", "knee", "calf", "ankle", "leg")
            self._set_all_ik_influences(
                lambda name: 1.0 if any(token in name.lower() for token in tokens) else 0.0
            )
        elif preset == "Selected limb only":
            kind, object_name = self._selected_target()
            logical = self.reverse_bindings.get((kind, object_name), "")
            lower = logical.lower()
            side = next(
                (token for token in ("left", "right", "fl", "fr", "rl", "rr")
                 if lower.startswith(token)),
                "",
            )
            limb_tokens = (
                ("shoulder", "elbow", "wrist", "arm")
                if "hand" in lower else
                ("hip", "thigh", "knee", "calf", "ankle", "leg")
            )
            self._set_all_ik_influences(lambda name: 1.0 if (
                (not side or name.lower().startswith(side))
                and any(token in name.lower() for token in limb_tokens)
            ) else 0.0)
        elif preset == "Feet planted":
            self._set_all_ik_influences(lambda name: 1.0)
            checkbox, spin = self.ik_task_controls["foot_lock"]
            checkbox.setChecked(True)
            spin.setValue(max(1.0, spin.value()))
        self.status_label.setText(f"Applied IK preset: {preset}")

    def _task_setting(self, name):
        checkbox, spin = self.ik_task_controls[name]
        return checkbox.isChecked(), float(spin.value())

    def _solver_settings(self):
        return {
            "damping": self.ik_damping.value(),
            "max_iterations": self.ik_max_iterations.value(),
            "step_size": self.ik_step_size.value(),
            "max_step": self.ik_max_step.value(),
            "position_tolerance": self.ik_position_tolerance.value(),
            "orientation_tolerance": self.ik_orientation_tolerance.value(),
        }

    def _capture_secondary_targets(self):
        self.preview_reference_qpos = self.committed_state.get_qpos()
        self.foot_lock_targets = {}
        for logical, (kind, name) in self.frame_bindings.items():
            if "foot" not in logical.lower():
                continue
            position, quaternion = self.committed_state.get_body_pose(name, kind)
            self.foot_lock_targets[logical] = (
                kind, name, position.copy(), quaternion.copy()
            )
        root_name = self.robot_model.root_body
        self.root_lock_target = (
            self.committed_state.get_body_pose(root_name, "body")
            if root_name else None
        )

    def _secondary_ik_tasks(self):
        tasks = []
        posture_enabled, posture_weight = self._task_setting("posture")
        if posture_enabled and posture_weight > 0.0:
            tasks.append(PostureTask(
                name="Posture preservation", weight=posture_weight,
                priority=3, enabled=True, required=False, tolerance=0.2,
                reference_qpos=self.preview_reference_qpos,
            ))
        regularization_enabled, regularization_weight = self._task_setting(
            "regularization"
        )
        if regularization_enabled and regularization_weight > 0.0:
            tasks.append(JointRegularizationTask(
                name="Joint regularization", weight=regularization_weight,
                priority=3, enabled=True, required=False, tolerance=0.3,
                reference_qpos=self.robot_model.home_qpos,
            ))
        selected = self.reverse_bindings.get(self._selected_target())
        foot_enabled, foot_weight = self._task_setting("foot_lock")
        if foot_enabled and foot_weight > 0.0:
            for logical, (kind, name, position, _) in self.foot_lock_targets.items():
                if logical == selected:
                    continue
                tasks.append(FootLockTask(
                    name=f"Lock {logical}", weight=foot_weight,
                    priority=1, enabled=True, required=False, tolerance=0.005,
                    object_name=name, kind=kind, target_position=position,
                ))
        root_enabled, root_weight = self._task_setting("root_orientation")
        if (
            root_enabled and root_weight > 0.0
            and self.root_lock_target is not None and self.robot_model.root_body
        ):
            _, quaternion = self.root_lock_target
            tasks.append(RootPoseTask(
                name="Root/base upright", weight=root_weight,
                priority=1, enabled=True, required=False, tolerance=0.05,
                object_name=self.robot_model.root_body, kind="body",
                target_quaternion=quaternion,
            ))
        return tasks

    def begin_preview(self):
        if not self.robot_state or self.preview_active:
            return
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self._capture_secondary_targets()
        self.preview_active = True
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(True)
        self._update_root_pose_label()

    def _on_transform_moved(self, position, quaternion):
        kind, name = self._selected_target()
        if not name:
            self.status_label.setText("Target pose moved; no body/site is selected.")
            return
        self.begin_preview()
        if self.last_valid_target_position is None:
            current_position, current_quaternion = self.preview_state.get_body_pose(
                name, kind
            )
            self.last_valid_target_position = current_position
            self.last_valid_target_quaternion = current_quaternion

        secondary_tasks = self._secondary_ik_tasks()
        tcp_position_enabled, tcp_position_weight = self._task_setting(
            "tcp_position"
        )
        tcp_orientation_enabled, tcp_orientation_weight = self._task_setting(
            "tcp_orientation"
        )
        selected_task_count = int(
            tcp_position_enabled and tcp_position_weight > 0.0
        ) + int(tcp_orientation_enabled and tcp_orientation_weight > 0.0)
        _, selected_object_id = self.preview_state.resolve_object(name, kind)
        is_free_root = (
            kind == "body"
            and self.robot_model.free_joint_for_body(selected_object_id) is not None
        )
        result = self.collision_solver.solve_drag(
            self.preview_state.get_qpos(),
            self.last_valid_target_position,
            self.last_valid_target_quaternion,
            position,
            quaternion,
            object_name=name,
            kind=kind,
            joint_weights=self.ik_joint_weights,
            secondary_tasks=secondary_tasks,
            solver_settings=self._solver_settings(),
            tcp_position_weight=(
                tcp_position_weight if tcp_position_enabled else 0.0
            ),
            tcp_orientation_weight=(
                tcp_orientation_weight if tcp_orientation_enabled else 0.0
            ),
        )
        if result.success:
            self.preview_state.set_qpos(result.qpos)
            self.last_valid_target_position = result.position.copy()
            self.last_valid_target_quaternion = result.quaternion.copy()

        # Collision is a preview warning rather than a drag constraint. IK
        # failures can still leave the handle at the last solvable substep.
        self.canvas.set_target_pose(
            self.last_valid_target_position,
            self.last_valid_target_quaternion,
        )
        self._update_preview_collisions(
            result.collisions
            if result.success or result.collisions
            else None
        )
        self._sync_joint_controls()
        logical_frame = self.reverse_bindings.get((kind, name), name)
        model_name = getattr(
            self.robot_model, "model_name", self.robot_model.model_path.stem
        )
        active_task_count = selected_task_count + (
            0 if is_free_root else len(secondary_tasks)
        )
        singularity_note = ""
        if result.near_singularity:
            singularity_note = (
                f"; near singularity sigma_min={result.min_singular_value:.2e}, "
                f"cond={result.condition_number:.1e}"
            )
        drag_status = self.canvas.gizmo.drag_status()
        drag_prefix = f"{drag_status}; " if drag_status else ""
        self._last_ik_status = (
            drag_prefix +
            f"{'TCP free translate; ' if self.canvas.gizmo.state.name == 'DRAG_TRANSLATE_FREE' else ''}"
            f"{result.status}; accepted={result.accepted_fraction:.0%}; "
            f"IK error={result.ik_error:.4f}; tasks={active_task_count}; "
            f"frame={logical_frame}; model={model_name}{singularity_note}; "
            "preview not committed"
        )
        self.status_label.setText(self._last_ik_status)
        roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
        self.target_pose_dragged.emit(
            *map(float, self.last_valid_target_position), roll, pitch, yaw
        )

    def _set_collision_substeps(self, count):
        if self.collision_solver:
            self.collision_solver.collision_drag_substeps = int(count)

    def _on_transform_drag_finished(self):
        if self.last_valid_target_position is not None:
            detail = self._last_ik_status or "Preview pose updated"
            self.status_label.setText(
                f"{detail}; ready to Plan, Accept, or Cancel"
            )

    def _on_transform_cancel_requested(self):
        if self.preview_active:
            self.cancel_preview()
        else:
            self.status_label.setText("Transform edit cancelled.")

    def _on_gizmo_mode_changed(self, mode):
        self.status_label.setText(f"Transform gizmo mode: {mode}.")

    def _joint_limit_violation(self, qpos):
        for joint in self.robot_model.joints.values():
            if joint.limits is None:
                continue
            value = float(qpos[joint.qpos_address])
            lo, hi = joint.limits
            if value < lo - 1e-9 or value > hi + 1e-9:
                return (
                    f"{joint.name} outside limits "
                    f"[{lo:.3f}, {hi:.3f}] at {value:.3f}"
                )
        return None

    def _build_validated_preview_path(self, start, goal, samples=40):
        planned = []
        candidate = self.robot_model.create_state()
        for index, alpha in enumerate(np.linspace(0.0, 1.0, int(samples))):
            qpos = self.state_timeline._interpolate(start, goal, alpha)
            qpos = np.asarray(qpos, dtype=float)
            if not np.all(np.isfinite(qpos)):
                return (
                    PreviewPathValidation(
                        False, f"non-finite qpos at path sample {index}", index
                    ),
                    [],
                )
            limit_error = self._joint_limit_violation(qpos)
            if limit_error:
                return (
                    PreviewPathValidation(
                        False, f"{limit_error} at path sample {index}", index
                    ),
                    [],
                )
            candidate.set_qpos(qpos)
            collisions = (
                self.collision_checker.get_collisions(candidate)
                if self.collision_checker else []
            )
            if collisions:
                names = ", ".join(
                    f"{item.geom1} <-> {item.geom2}"
                    for item in collisions[:2]
                )
                return (
                    PreviewPathValidation(
                        False,
                        f"collision at path sample {index}: {names}",
                        index,
                    ),
                    [],
                )
            planned.append(qpos)
        return PreviewPathValidation(True, "Preview path is valid."), planned

    def plan_preview(self):
        if not self.preview_active:
            self.status_label.setText("No preview changes to validate.")
            return
        start = self.committed_state.get_qpos()
        goal = self.preview_state.get_qpos()
        validation, planned = self._build_validated_preview_path(start, goal)
        if not validation.ok:
            self._clear_ghost_overlay(source="preview_path")
            self.status_label.setText(
                f"Cannot preview path: {validation.message}."
            )
            return
        self.ghost_trajectory = [qpos.copy() for qpos in planned]
        self.ghost_source = "preview_path"
        self._rebuild_ghosts()
        self.status_label.setText(
            "Preview path is valid; no keyframe was changed."
        )
        self.history_action_finished.emit("Preview path")

    def accept_preview(self, *, emit_pose_finished=True):
        if not self.preview_active:
            self.status_label.setText("No preview changes to commit.")
            return False
        preview_qpos = self.preview_state.get_qpos()
        if not np.all(np.isfinite(preview_qpos)):
            self.status_label.setText(
                "Cannot commit keyframe: preview pose contains non-finite qpos values."
            )
            return False
        collisions = (
            self.collision_checker.get_collisions(self.preview_state)
            if self.collision_checker else []
        )
        if collisions:
            self.canvas.set_preview_collisions(collisions)
            names = ", ".join(
                f"{item.geom1} ↔ {item.geom2}" for item in collisions[:2]
            )
            self.status_label.setText(
                f"Cannot commit keyframe: collision detected ({names})."
            )
            return False
        self.committed_state.set_qpos(preview_qpos)
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText(
            f"Committed keyframe at t={self.current_time:.2f} s"
        )
        if emit_pose_finished and self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self.target_pose_drag_finished.emit(
                *map(float, self.last_valid_target_position), roll, pitch, yaw
            )
        return True

    def cancel_preview(self):
        if not self.robot_state:
            return
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText("Preview discarded; committed state is unchanged.")
        self.preview_cancelled.emit()

    def _on_body_double_clicked(self, body_name):
        logical = (
            self.robot_model.logical_frame_for_body(body_name)
            if hasattr(self.robot_model, "logical_frame_for_body") else None
        )
        binding = self.frame_bindings.get(logical) if logical else None
        if binding is None or not self.select_target(*binding, emit=False):
            self.status_label.setText(
                f"Body {body_name!r} has no nearby editable trajectory frame."
            )
            return
        self.target_frame_changed.emit(logical)
        self.status_label.setText(
            f"Selected {logical} from double-clicked body {body_name}."
        )

    def get_current_time(self):
        return self.current_time

    def _trajectory_sample(self, time):
        """Return an interpolated playback qpos and its nearest frame index."""
        if not self.robot_trajectory:
            return None, None

        times = self.robot_trajectory_times
        if len(times) != len(self.robot_trajectory):
            times = [float(index) for index in range(len(self.robot_trajectory))]

        time = float(time)
        if len(times) == 1 or time <= times[0]:
            return self.robot_trajectory[0].copy(), 0
        if time >= times[-1]:
            last = len(self.robot_trajectory) - 1
            return self.robot_trajectory[last].copy(), last

        upper = bisect_right(times, time)
        lower = max(0, upper - 1)
        upper = min(len(times) - 1, upper)
        lower_time = times[lower]
        upper_time = times[upper]
        if upper_time <= lower_time:
            return self.robot_trajectory[upper].copy(), upper

        fraction = (time - lower_time) / (upper_time - lower_time)
        qpos = self.state_timeline._interpolate(
            self.robot_trajectory[lower],
            self.robot_trajectory[upper],
            fraction,
        )
        nearest = (
            lower
            if time - lower_time <= upper_time - time
            else upper
        )
        return qpos, nearest

    def _update_frame_readout(self, time=None, frame_index=None):
        if not self.robot_trajectory:
            self.timeslice_frame_readout.setText("Frame —")
            return
        if frame_index is None:
            _qpos, frame_index = self._trajectory_sample(
                self.display_time if time is None else time
            )
        self.timeslice_frame_readout.setText(
            f"Frame {int(frame_index) + 1} / {len(self.robot_trajectory)}"
        )

    def _set_canvas_target_from_state(self, state):
        kind, name = self._selected_target()
        if not name:
            return
        try:
            position, quaternion = state.get_body_pose(name, kind)
        except KeyError:
            return
        self.canvas.set_target_pose(position, quaternion)

    def _emit_selected_target_pose(self):
        if (
            self.last_valid_target_position is None
            or self.last_valid_target_quaternion is None
        ):
            return
        roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
        self.target_pose_dragged.emit(
            *map(float, self.last_valid_target_position),
            roll,
            pitch,
            yaw,
        )

    def _use_editor_canvas_states(self):
        if not self.robot_state:
            return
        self.canvas.set_robot_states(
            self.committed_state, self.preview_state, self.ghost_renderer
        )
        self.canvas.set_preview_visible(self.preview_active)

    def preview_trajectory_time(self, time, emit_time_signal=False):
        """Display a timeline pose without creating an editable qpos state."""
        if not self.robot_state:
            return None
        time = max(0.0, min(float(time), self.timeline_duration))
        self.display_time = time
        self._set_timeslice_widgets(time)

        qpos, frame_index = self._trajectory_sample(time)
        if qpos is None and self.state_timeline:
            qpos = self.state_timeline.sample_state(
                time,
                fallback_qpos=self.committed_state.get_qpos(),
            )

        if qpos is not None and self.playback_state is not None:
            self.playback_state.set_qpos(qpos)
            self.canvas.set_robot_states(
                self.playback_state, self.preview_state, self.ghost_renderer
            )
            self.canvas.set_preview_visible(False)
            self._sync_joint_controls(state=self.playback_state)
            self._set_canvas_target_from_state(self.playback_state)
            self.canvas.update()

        self._update_frame_readout(time, frame_index)
        self._update_timeline_label(time)
        if emit_time_signal:
            self.timeslice_preview_time_changed.emit(time)
        return None if qpos is None else qpos.copy()

    def get_current_keyframe(self):
        if not self.state_timeline:
            return None
        return self.state_timeline.get_state(self.current_time)

    def ensure_keyframe_at_current_time(self):
        if not self.state_timeline:
            return None
        return self.state_timeline.ensure_state(
            self.current_time,
            fallback_qpos=self.robot_state.get_qpos(),
        )

    def set_robot_state_for_current_time(self, qpos):
        if not self.robot_state:
            return
        self._use_editor_canvas_states()
        self.committed_state.set_qpos(qpos)
        self.preview_state.set_qpos(qpos)
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.canvas.update()

    def update_current_keyframe_from_robot_state(self, refresh_ghosts=True):
        if not self.state_timeline or not self.robot_state:
            return
        self.state_timeline.set_state(
            self.current_time, self.committed_state.get_qpos()
        )
        self._update_timeline_label()
        if refresh_ghosts:
            self._refresh_timeline_trajectory()

    def set_current_time(self, time):
        if not self.state_timeline:
            return
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        time = max(0.0, min(float(time), self.timeline_duration))
        self.current_time = self.state_timeline.time_key(time)
        self.display_time = self.current_time
        self._set_timeslice_widgets(self.current_time)
        qpos, _frame_index = self._trajectory_sample(self.current_time)
        if qpos is not None:
            # Releasing the unified scrubber commits exactly the pose that was
            # shown during live trajectory sampling.
            self.state_timeline.set_state(self.current_time, qpos)
        else:
            qpos = self.ensure_keyframe_at_current_time()
        self.set_robot_state_for_current_time(qpos)
        self._emit_selected_target_pose()
        self._update_frame_readout(self.current_time)
        self._refresh_timeline_trajectory()
        self._update_timeline_label()
        self.status_label.setText(
            f"Loaded editable 3D robot state at t={self.current_time:.2f} s"
        )

    def reset_robot_pose(self):
        """Reset only the active 3D timeline frame to the model home pose."""
        if not self.robot_state:
            return
        # Reset is an immediate one-shot action, never a playback mode. If the
        # timer is active, pause first so its next tick cannot overwrite the
        # just-reset editing frame, and cancel any in-progress gizmo gesture.
        was_playing = self.play_timer.isActive()
        self.pause_playback(commit_time=was_playing)
        self.canvas.cancel_transform_drag()
        self.committed_state.reset_to_default()
        self.preview_state.set_qpos(self.committed_state.get_qpos())
        self.preview_active = False
        self.canvas.set_preview_collisions([])
        self.canvas.set_preview_visible(False)
        self._clear_ghost_overlay(source="preview_path")
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        self.status_label.setText(
            f"Reset 3D pose once at t={self.current_time:.2f} s to model home qpos"
            + ("; playback paused" if was_playing else "")
        )
        if self.last_valid_target_position is not None:
            roll, pitch, yaw = quat_to_rpy(self.last_valid_target_quaternion)
            self._pending_history_action_description = "Reset 3D pose"
            self.target_pose_drag_finished.emit(
                *map(float, self.last_valid_target_position), roll, pitch, yaw
            )

    def choose_qpos_csv(self):
        self._open_csv_file_dialog(
            title="Load robot qpos",
            directory=QPOS_CSV_DIR,
            selected=self._load_selected_qpos_csv,
        )

    def choose_trajectory_csv(self, prompt_import_dt=False):
        self._open_csv_file_dialog(
            title="Load robot trajectory",
            directory=TRAJECTORY_CSV_DIR,
            selected=lambda path: self._load_selected_trajectory_csv(
                path, prompt_import_dt
            ),
        )

    def choose_qpos_save_path(self):
        self._open_csv_file_dialog(
            title="Save robot qpos",
            directory=QPOS_CSV_DIR,
            selected=self._save_selected_qpos_csv,
            save=True,
            filename="updated_qpos.csv",
        )

    def choose_trajectory_save_path(self):
        self._open_csv_file_dialog(
            title="Save robot trajectory",
            directory=TRAJECTORY_CSV_DIR,
            selected=self._save_selected_trajectory_csv,
            save=True,
            filename="robot_trajectory.csv",
        )

    def _open_csv_file_dialog(
        self,
        *,
        title,
        directory,
        selected,
        save=False,
        filename=None,
    ):
        if self.background_jobs.is_busy():
            self.status_label.setText(
                "Wait for the current import or export to finish."
            )
            return
        if self.csv_file_selection_stage.is_active():
            self.status_label.setText("A file selector is already open.")
            return

        self.csv_file_selection_stage.select_file(
            mode="save" if save else "open",
            title=title,
            directory=directory,
            name_filter="CSV files (*.csv);;All files (*)",
            filename=filename,
            selected=selected,
            failed=lambda message: self.status_label.setText(
                f"Could not open file selector: {message}"
            ),
        )

    def _load_selected_qpos_csv(self, path):
        expected = int(self.robot_model.mj_model.nq)
        self.status_label.setText(f"Loading qpos from {Path(path).name}...")
        self._submit_csv_job(
            "load qpos CSV",
            lambda: read_qpos_csv(path, expected),
            self._apply_loaded_qpos,
            lambda error: self.status_label.setText(
                f"Could not load qpos CSV: {error}"
            ),
        )

    def _load_selected_trajectory_csv(self, path, prompt_import_dt=False):
        expected = int(self.robot_model.mj_model.nq)
        self.status_label.setText(
            f"Loading trajectory from {Path(path).name}..."
        )
        self._submit_csv_job(
            "load trajectory CSV",
            lambda: read_trajectory_csv(path, expected),
            lambda loaded: self._apply_loaded_trajectory(
                loaded,
                prompt_import_dt=prompt_import_dt,
                background_postprocess=True,
            ),
            lambda error: self._trajectory_csv_load_failed(error),
        )

    def _save_selected_qpos_csv(self, path):
        qpos = self.committed_state.get_qpos().copy()
        preview_active = bool(self.preview_active)
        self.status_label.setText(f"Saving qpos to {Path(path).name}...")
        self._submit_csv_job(
            "save qpos CSV",
            lambda: write_qpos_csv(path, qpos),
            lambda saved_path: self._show_qpos_saved(
                saved_path, preview_active
            ),
            lambda error: self.status_label.setText(
                f"Could not save qpos CSV: {error}"
            ),
        )

    def _save_selected_trajectory_csv(self, path):
        try:
            export = self._trajectory_export_snapshot()
        except ValueError as error:
            self.status_label.setText(
                f"Could not save trajectory CSV: {error}"
            )
            return
        self.status_label.setText(
            f"Saving trajectory to {Path(path).name}..."
        )
        self._submit_csv_job(
            "save trajectory CSV",
            lambda: write_trajectory_csv(path, export),
            lambda saved_path: self._show_trajectory_saved(
                saved_path, export
            ),
            lambda error: self.status_label.setText(
                f"Could not save trajectory CSV: {error}"
            ),
        )

    def _submit_csv_job(self, name, work, succeeded, failed):
        if self.background_jobs.is_busy():
            self.status_label.setText(
                "Wait for the current import or export to finish."
            )
            return False
        submitted = self.background_jobs.submit(
            name,
            work,
            succeeded,
            failed,
        )
        if not submitted:
            self.status_label.setText(f"Could not start {name}.")
        return submitted

    def load_qpos_csv(self, csv_path):
        """Load one headerless MuJoCo qpos row into the active keyframe."""
        expected = int(self.robot_model.mj_model.nq)
        loaded = read_qpos_csv(csv_path, expected)
        self._apply_loaded_qpos(loaded)

    def _apply_loaded_qpos(self, loaded):
        self.pause_playback()
        self.canvas.cancel_transform_drag()
        self.set_robot_state_for_current_time(loaded.qpos)
        self.update_current_keyframe_from_robot_state(refresh_ghosts=True)
        self.status_label.setText(
            f"Loaded {loaded.qpos.size}-value qpos from {loaded.path.name} at "
            f"t={self.current_time:.2f} s"
        )
        self.history_action_finished.emit("Load qpos")

    def load_trajectory_csv(self, csv_path):
        """Load headerless time,qpos rows into playback and editable qpos states."""
        expected = int(self.robot_model.mj_model.nq)
        loaded = read_trajectory_csv(csv_path, expected)
        self._apply_loaded_trajectory(loaded)

    def _apply_loaded_trajectory(
        self,
        loaded,
        prompt_import_dt=None,
        background_postprocess=False,
    ):
        qposes = list(loaded.qposes)
        times = list(loaded.times)
        if prompt_import_dt is not None:
            self._prompt_trajectory_import_dt_on_load = bool(prompt_import_dt)
        self.pause_playback()
        self.canvas.cancel_transform_drag()
        self.set_robot_trajectory(qposes, times=times)
        if self.state_timeline and qposes:
            self.state_timeline.reset(times[0], qposes[0])
            for time, qpos in zip(times[1:], qposes[1:]):
                self.state_timeline.set_state(time, qpos)
            self.set_defined_timeslices(times)
            self.current_time = self.state_timeline.time_key(times[0])
            self.display_time = self.current_time
            self._set_timeslice_widgets(self.current_time)
            self._update_frame_readout(self.current_time)
            self._update_timeline_label()
        duration = times[-1] if times else 0.0
        self.status_label.setText(
            f"Loaded {len(qposes)} timed qpos trajectory states from "
            f"{loaded.path.name} ({duration:.3f} s)"
        )
        self._background_trajectory_postprocess_requested = bool(
            background_postprocess
        )
        self.trajectory_csv_loaded.emit(str(loaded.path))

    def _trajectory_csv_load_failed(self, error):
        self._prompt_trajectory_import_dt_on_load = False
        self.status_label.setText(f"Could not load trajectory CSV: {error}")

    def save_qpos_csv(self, csv_path):
        """Save the committed active keyframe as one headerless qpos row."""
        preview_active = bool(self.preview_active)
        path = write_qpos_csv(csv_path, self.committed_state.get_qpos())
        self._show_qpos_saved(path, preview_active)
        return path

    def _show_qpos_saved(self, path, preview_active):
        preview_note = (
            "; unaccepted preview was not saved" if preview_active else ""
        )
        self.status_label.setText(f"Saved committed qpos to {path}{preview_note}")

    def save_trajectory_csv(self, csv_path):
        """Save generated trajectory rows, falling back to editable timeline rows."""
        export = self._trajectory_export_snapshot()
        path = write_trajectory_csv(csv_path, export)
        self._show_trajectory_saved(path, export)
        return path

    def _trajectory_export_snapshot(self):
        if not self.state_timeline:
            raise ValueError("no robot timeline is available")

        expected = int(self.robot_model.mj_model.nq)
        source_name = "generated trajectory"

        if self.robot_trajectory:
            qposes = tuple(qpos.copy() for qpos in self.robot_trajectory)
            times = tuple(float(time) for time in self.robot_trajectory_times)
            if len(times) != len(qposes):
                times = tuple(float(index) for index in range(len(qposes)))
        else:
            self.state_timeline.set_state(
                self.current_time, self.committed_state.get_qpos()
            )
            times = tuple(float(time) for time in self.state_timeline.times())
            qposes = tuple(
                self.state_timeline.get_state(time_value)
                for time_value in times
            )
            source_name = "editable timeline"

        return TrajectoryExport(
            expected_qpos_count=expected,
            times=times,
            qposes=qposes,
            source_name=source_name,
            preview_active=bool(self.preview_active),
        )

    def _show_trajectory_saved(self, path, export):
        preview_note = (
            "; unaccepted preview was not saved"
            if export.preview_active
            else ""
        )
        self.status_label.setText(
            f"Saved {len(export.qposes)} timed qpos states from "
            f"{export.source_name} "
            f"to {path}{preview_note}"
        )

    def _refresh_timeline_trajectory(self):
        # Timeline keyframes no longer publish ghost poses. Scrubbing and
        # accepting states should update the committed robot, not silently swap
        # the overlay from preview/playback into keyframe ghosts.
        self._update_timeline_label()

    def _update_timeline_label(self, display_time=None):
        count = len(self.state_timeline.states) if self.state_timeline else 0
        time = self.current_time if display_time is None else float(display_time)
        self.timeline_state_label.setText(
            f"3D state time: {time:.2f} s ({count} keyframes)"
        )

    def _update_root_pose_label(self):
        if not self.robot_state:
            self.root_pose_label.setText("unavailable")
            return
        free_joints = list(self.robot_model.free_joints_by_body.values())
        if not free_joints:
            self.root_pose_label.setText("fixed root")
            return
        address = free_joints[0].qpos_address
        state = self.preview_state if self.preview_active else self.committed_state
        x, y, z = state.mj_data.qpos[address:address + 3]
        suffix = " (preview)" if self.preview_active else " (committed)"
        self.root_pose_label.setText(f"{x:+.3f}, {y:+.3f}, {z:+.3f} m{suffix}")

    def set_robot_trajectory(self, qposes, times=None, activate_first_frame=True):
        if not self.robot_state:
            return
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        valid = []
        valid_times = []
        times = list(times) if times is not None else None
        for index, qpos in enumerate(qposes):
            try:
                if len(qpos) == self.robot_state.mj_model.nq:
                    valid.append(qpos.copy())
                    if times is not None and index < len(times):
                        valid_times.append(float(times[index]))
            except (TypeError, AttributeError):
                continue
        if times is None or len(valid_times) != len(valid):
            valid_times = [float(index) for index in range(len(valid))]
        elif any(
            earlier > later
            for earlier, later in zip(valid_times, valid_times[1:])
        ):
            ordered = sorted(zip(valid_times, valid), key=lambda item: item[0])
            valid_times = [item[0] for item in ordered]
            valid = [item[1] for item in ordered]
        self.robot_trajectory = valid
        self.robot_trajectory_times = valid_times
        if valid_times:
            self.set_timeline_duration(max(self.timeline_duration, max(valid_times)))
        self._clear_ghost_overlay(source="preview_path")
        self._sync_playback_pose_ghosts()
        if valid and activate_first_frame:
            self.set_current_time(valid_times[0])
            self.status_label.setText(f"Loaded {len(valid)} robot trajectory states.")
        else:
            self._update_frame_readout(self.display_time)

    def clear_robot_trajectory(self):
        self.pause_playback()
        self.scrub_preview_timer.stop()
        self._pending_scrub_preview_time = None
        self.robot_trajectory = []
        self.robot_trajectory_times = []
        self.ghost_trajectory = []
        self.ghost_source = None
        self.display_time = self.current_time
        self._set_timeslice_widgets(self.display_time)
        self._update_frame_readout()
        self._use_editor_canvas_states()
        self._sync_joint_controls()
        self._set_target_to_selected_pose()
        if self.ghost_renderer:
            self.ghost_renderer.clear()
        self._update_ghost_options()

    def clear_editable_timeline(self, keep_current_pose=True, reset_time=None):
        if not self.state_timeline or not self.robot_state:
            return

        qpos = (
            self.committed_state.get_qpos()
            if keep_current_pose
            else self.robot_model.home_qpos
        )
        if reset_time is not None:
            self.current_time = self.state_timeline.time_key(reset_time)
            self.display_time = self.current_time
            self._set_timeslice_widgets(self.current_time)
        self.state_timeline.reset(self.current_time, qpos)
        self.set_robot_state_for_current_time(qpos)
        self._update_timeline_label()

    def load_backend_states(self, states):
        if not self.robot_state:
            return
        qposes = []
        times = []
        for configuration in states:
            times.append(float(getattr(configuration, "time", len(times))))
            if getattr(configuration, "qpos", None) is not None:
                qposes.append(configuration.qpos.copy())
                continue
            qpos = self.robot_state.get_qpos()
            if len(qpos) >= 7:
                qpos[:7] = [configuration.base_x, configuration.base_y, configuration.base_z,
                            configuration.base_qw, configuration.base_qx,
                            configuration.base_qy, configuration.base_qz]
            for name, value in zip(configuration.joint_names, configuration.joint_positions):
                joint = self.robot_model.joints.get(name)
                if joint:
                    qpos[joint.qpos_address] = value
            qposes.append(qpos)
        self.set_robot_trajectory(qposes, times=times)

    def generate_demo_trajectory(self):
        start = self.robot_state.get_qpos()
        target = start.copy()
        names = self.robot_state.get_joint_names()
        preferred = "left_shoulder_pitch_joint"
        name = preferred if preferred in names else names[0]
        joint = self.robot_model.joints[name]
        lo, hi = joint.limits or (-1.0, 1.0)
        target[joint.qpos_address] = max(lo, min(hi, start[joint.qpos_address] + 0.35))
        qposes = interpolate_qpos(start, target, 60)
        frame_period = self.play_timer.interval() / 1000.0
        times = [index * frame_period for index in range(len(qposes))]
        self.set_robot_trajectory(qposes, times=times)
        self.history_action_finished.emit("Demo trajectory")

    def _rebuild_ghosts(self):
        if self.ghost_renderer:
            self.ghost_renderer.update(
                self.ghost_trajectory, self.ghost_stride.value()
            )
        self._update_ghost_options()

    def _clear_ghost_overlay(self, source=None):
        if source is not None and self.ghost_source != source:
            return
        self.ghost_trajectory = []
        self.ghost_source = None
        if self.ghost_renderer:
            self.ghost_renderer.clear()
        self._update_ghost_options()

    def _sync_playback_pose_ghosts(self):
        if self.show_ghosts.isChecked() and self.robot_trajectory:
            self.ghost_trajectory = [qpos.copy() for qpos in self.robot_trajectory]
            self.ghost_source = "playback"
            self._rebuild_ghosts()
            return
        if self.ghost_source == "playback":
            self.ghost_trajectory = []
            self.ghost_source = None
            if self.ghost_renderer:
                self.ghost_renderer.clear()
        self._update_ghost_options()

    def _update_ghost_options(self):
        visible = bool(
            self.ghost_trajectory and (
                self.ghost_source == "preview_path"
                or (
                    self.ghost_source == "playback"
                    and self.show_ghosts.isChecked()
                )
            )
        )
        self.canvas.set_ghost_options(visible, self.ghost_alpha.value())

    def set_trajectory_frame(self, index):
        """Compatibility entry point that selects a trajectory frame by time."""
        if not self.robot_trajectory:
            return
        index = max(0, min(len(self.robot_trajectory) - 1, int(index)))
        self.set_current_time(self.robot_trajectory_times[index])

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.pause_playback(commit_time=True)
        else:
            self.start_playback()

    def start_playback(self):
        if not self.robot_trajectory:
            return
        start_time = self.robot_trajectory_times[0]
        end_time = self.robot_trajectory_times[-1]
        if end_time <= start_time:
            self.preview_trajectory_time(start_time, emit_time_signal=True)
            return
        if self.display_time < start_time or self.display_time >= end_time:
            self.preview_trajectory_time(start_time, emit_time_signal=True)
        self.playback_clock.start()
        self.play_timer.start()
        self._set_playback_button_text("Pause")
        self.playback_state_changed.emit(True)

    def pause_playback(self, commit_time=False):
        self.play_timer.stop()
        self.playback_clock.stop()
        self._set_playback_button_text("Play")
        self.playback_state_changed.emit(False)
        if commit_time and abs(self.display_time - self.current_time) > 1e-9:
            self.timeslice_time_changed.emit(self.display_time)

    def _set_playback_button_text(self, text):
        self.play_button.setText(text)

    def _advance_playback(self, elapsed=None):
        if not self.robot_trajectory:
            self.pause_playback()
            return
        start_time = self.robot_trajectory_times[0]
        end_time = self.robot_trajectory_times[-1]
        if end_time <= start_time:
            self.preview_trajectory_time(start_time, emit_time_signal=True)
            self.pause_playback(commit_time=True)
            return

        elapsed = self.playback_clock.elapsed(
            self.play_timer.interval() / 1000.0,
            supplied=elapsed,
        )
        next_time = self.playback_clock.advance(
            self.display_time,
            start_time,
            end_time,
            elapsed,
            self.playback_speed.value(),
        )
        self.preview_trajectory_time(next_time, emit_time_signal=True)

    def _advance_frame(self):
        """Backward-compatible alias for elapsed-time playback advancement."""
        self._advance_playback()
