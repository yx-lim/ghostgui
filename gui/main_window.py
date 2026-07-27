"""
main_window.py

Purpose:
    Main GUI window for reference-frame trajectory editing.

Updated project flow:
    1. User drags/edits target frame
    2. User adds keyframe
    3. GUI stores trajectory array
    4. User clicks Generate / Simulate
    5. Backend maps robot to each target frame
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from PySide6.QtCore import QEvent, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QStackedWidget,
    QSizePolicy,
    QInputDialog,
    QMessageBox,
    QToolBar,
)

from core.trajectory import (
    TargetFrame,
    quat_to_rpy,
    rpy_to_quat,
)
from application import model_sessions, timeslice_service, trajectory_generation
from application.editor_commands import (
    AddKeyframe,
    ClearTrajectory,
    DeleteKeyframe,
    DeleteTimeslice,
    ReplaceTrajectoryFrames,
    UpdateKeyframe,
    UpsertKeyframe,
    UpsertKeyframes,
)
from application.editor_controller import EditorController
from application.editor_events import DocumentDirtyChanged, EditorEventBus
from application.project_document import ProjectDocument
from application.visualization import VisualizationUpdate
from application.background_jobs import SerializedBackgroundJobs
from application.history import HistoryStack
from application.csv_io import write_trajectory_csv
from application.paths import mujoco_playback_cache_path
from application.project_manager import (
    GhostGUIProject,
    available_default_project_root_from_name,
    forget_recent_project,
    load_project_browser_previews,
    load_recent_projects,
    remember_recent_project,
    ghostgui_config_dir,
)
from .controls import TrajectoryControlPanel
from .file_selection import SynchronousFileSelectionStage
from .history import GuiHistorySnapshot
from .model_loading import ModelLoadThread
from .panels import EditorStatusPanel
from .render_progress import RenderProgressOverlay
from .robot_viewer_3d import RobotViewer3D
from .widgets.status import StatusEvent, status_event_from_text
from gui.viewers.mujoco_player import Mujoco3DViewerPanel
from application.backend_interface import BackendInterface
from core.models import MujocoReferenceFrames
from .app_sidebars import AppLeftSidebar, AppRightSidebar, SidebarSplitter
from .help import HelpCenterDialog
from .project_browser import ProjectBrowserDialog
from .theme import ensure_application_theme, theme_icon
from .tutorial import TutorialManager
from .visualization import SceneSnapshot, build_main_window_visualization
from core.models import MuJoCoRobotAdapter, ROBOT_MODELS
from application.model_importer import (
    default_model_library_root,
    discover_imported_models,
    import_robot_model,
)


LEFT_SIDEBAR_MIN_WIDTH = 200
LEFT_SIDEBAR_DEFAULT_WIDTH = 250
RIGHT_SIDEBAR_MIN_WIDTH = 200
RIGHT_SIDEBAR_DEFAULT_WIDTH = 270
SIDEBAR_MAX_WIDTH = 400
UI_SETTINGS_FILENAME = "ui.ini"
INITIAL_RENDER_PROGRESS_DELAY_MS = 500
PROJECT_AUTOSAVE_INTERVAL_MS = 30000
MAX_HISTORY_DEPTH = 100


class RobotGuiMainWindow(QMainWindow):
    @property
    def trajectory(self):
        return self.document.trajectory

    @trajectory.setter
    def trajectory(self, value):
        self.document.trajectory = value

    @property
    def active_index(self):
        return self.document.active_index

    @active_index.setter
    def active_index(self, value):
        self.document.active_index = int(value)

    def __init__(self, model_key="g1"):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            ensure_application_theme(app)

        self.setWindowTitle("GhostGui")
        self.ui_settings = QSettings(
            str(ghostgui_config_dir() / UI_SETTINGS_FILENAME),
            QSettings.Format.IniFormat,
        )
        try:
            saved_sidebar_width = int(
                self.ui_settings.value(
                    "left_sidebar/width",
                    LEFT_SIDEBAR_DEFAULT_WIDTH,
                )
            )
        except (TypeError, ValueError):
            saved_sidebar_width = LEFT_SIDEBAR_DEFAULT_WIDTH
        self._left_sidebar_last_width = max(
            LEFT_SIDEBAR_MIN_WIDTH,
            min(saved_sidebar_width, SIDEBAR_MAX_WIDTH),
        )
        self._restore_left_sidebar_collapsed = self.ui_settings.value(
            "left_sidebar/collapsed",
            False,
            type=bool,
        )
        self._left_sidebar_collapsed = False
        self._syncing_left_sidebar = False
        try:
            saved_right_sidebar_width = int(
                self.ui_settings.value(
                    "right_sidebar/width",
                    RIGHT_SIDEBAR_DEFAULT_WIDTH,
                )
            )
        except (TypeError, ValueError):
            saved_right_sidebar_width = RIGHT_SIDEBAR_DEFAULT_WIDTH
        self._right_sidebar_last_width = max(
            RIGHT_SIDEBAR_MIN_WIDTH,
            min(saved_right_sidebar_width, SIDEBAR_MAX_WIDTH),
        )
        self._restore_right_sidebar_collapsed = self.ui_settings.value(
            "right_sidebar/collapsed",
            False,
            type=bool,
        )
        self._right_sidebar_collapsed = False
        self._syncing_right_sidebar = False

        # --------------------------------------------------------
        # Core data
        # --------------------------------------------------------
        self.document = ProjectDocument(model_key=model_key)
        self.editor_events = EditorEventBus()
        self.editor_controller = EditorController(
            self.document,
            self.editor_events,
        )
        self._document_dirty_subscription = self.editor_events.subscribe(
            DocumentDirtyChanged,
            self._on_document_dirty_changed,
        )

        # One immutable MuJoCo model is shared by FK, IK, and rendering. Each
        # subsystem owns its own MjData so live UI and batch solves stay isolated.
        self.model_library_root = default_model_library_root()
        self.model_registry = dict(ROBOT_MODELS)
        for info in discover_imported_models(self.model_library_root).values():
            self.register_model_info(info)
        self.import_mesh_folder = None
        self.background_jobs = SerializedBackgroundJobs(self)
        self.file_selection_stage = SynchronousFileSelectionStage(self)
        # Compatibility name retained for existing callers and tests.
        self.model_file_selection_stage = self.file_selection_stage
        self.model_key = model_key
        self.robot_model_3d = None
        self.robot_model_error = None
        try:
            self.robot_model_3d = MuJoCoRobotAdapter(
                self.model_registry.get(model_key, model_key)
            )
            self.robot_model_error = self.robot_model_3d.load_warning
        except Exception as exc:
            self.robot_model_error = str(exc)
            if model_key != "g1":
                failed_model_key = model_key
                model_key = "g1"
                self.model_key = model_key
                try:
                    self.robot_model_3d = MuJoCoRobotAdapter(
                        self.model_registry.get(model_key, model_key)
                    )
                    fallback_warning = self.robot_model_3d.load_warning
                    self.robot_model_error = (
                        f"Could not load {failed_model_key}: {exc}\n"
                        "Loaded Unitree G1 instead."
                    )
                    if fallback_warning:
                        self.robot_model_error += f"\n{fallback_warning}"
                except Exception as fallback_exc:
                    self.robot_model_error = (
                        f"Could not load {failed_model_key}: {exc}\n"
                        f"Fallback g1 also failed: {fallback_exc}"
                    )
        if self.robot_model_3d is not None:
            self.setWindowTitle(
                f"GhostGui — {self.robot_model_3d.model_name}"
            )
        self.document.model_key = self.model_key

        shared_mj_model = (
            self.robot_model_3d.mj_model if self.robot_model_3d else None
        )
        self.backend_interface = BackendInterface(
            mj_model=shared_mj_model, adapter=self.robot_model_3d
        )
        self.model_reference = MujocoReferenceFrames(
            mj_model=shared_mj_model, adapter=self.robot_model_3d
        )

        # GUI widgets
        frame_names = (
            self.robot_model_3d.trajectory_frames
            if self.robot_model_3d else ["pelvis"]
        )
        self.controls = TrajectoryControlPanel(
            self.model_registry, model_key=model_key, frame_names=frame_names
        )
        self.viewer_3d = RobotViewer3D(
            robot_model=self.robot_model_3d,
            error=self.robot_model_error,
            background_jobs=self.background_jobs,
            file_selection_stage=self.file_selection_stage,
        )
        self.document.attach_qpos_timeline(self.viewer_3d.state_timeline)
        self.viewer_3d_mujoco = Mujoco3DViewerPanel(self.robot_model_3d)
        self.viewer_3d_mujoco.set_trajectory_regenerator(
            self.regenerate_mujoco_playback_cache
        )
        self.model_sessions = {
                model_key: model_sessions.RobotModelSession(
                adapter=self.robot_model_3d,
                backend=self.backend_interface,
                reference=self.model_reference,
                viewer_3d=self.viewer_3d,
                document=self.document,
                model_key=model_key,
            )
        }
        self.model_loaders = {}
        self._shutting_down = False
        self.current_project = None
        self._pending_project_restore = None
        self._pending_project_restore_autosave = False
        self._pending_project_restore_source = "direct"
        self.project_dirty = False
        self._suppress_project_dirty = False
        self.render_progress_overlay = None
        self.render_progress_viewer = None
        self.render_progress_restore_widget = None
        self.pending_initial_render_progress = None
        self.history = HistoryStack[GuiHistorySnapshot](MAX_HISTORY_DEPTH)
        # Compatibility views retained for integrations that inspect stack
        # availability directly.
        self.undo_stack = self.history.undo_entries
        self.redo_stack = self.history.redo_entries
        self._history_restoring = False
        self.viewer_tabs = self.build_viewer_tabs()
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.help_dialog = None
        self.build_menu_bar()
        self.app_toolbar = self.build_workflow_toolbar()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.app_toolbar)
        self.refresh_recent_projects()
        self.status_panel = self.build_status_panel()
        self.visualization_manager = build_main_window_visualization(self)
        self.left_sidebar_content = AppLeftSidebar(
            self.controls,
            include_view=False,
        )
        self.right_sidebar_content = AppRightSidebar(
            self.status_panel, self.controls.inspector_sections()
        )
        self.left_sidebar = self.left_sidebar_content
        self.right_sidebar = self.right_sidebar_content
        self.left_sidebar.setMinimumWidth(LEFT_SIDEBAR_MIN_WIDTH)
        self.left_sidebar.setMaximumWidth(SIDEBAR_MAX_WIDTH)
        self.left_sidebar.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        self.right_sidebar.setMinimumWidth(RIGHT_SIDEBAR_MIN_WIDTH)
        self.right_sidebar.setMaximumWidth(SIDEBAR_MAX_WIDTH)
        self.right_sidebar.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        self.connect_signals()
        self.sync_workflow_toolbar()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(PROJECT_AUTOSAVE_INTERVAL_MS)
        self.autosave_timer.timeout.connect(self.on_autosave_timer)
        self.autosave_timer.start()
        self.controls.corner_smoothing_slider.value_changed.connect(
            lambda _value: self.refresh_display()
        )
        self.controls.corner_smoothing_slider.value_changed.connect(
            lambda _value: self.mark_project_dirty("Smoothing")
        )
        self.set_current_frame_to_model_reference(
            self.controls.frame_box.currentText(),
            emit_pose_changed=False,
        )

        # --------------------------------------------------------
        # Layout
        # --------------------------------------------------------
        # Persistent splitter children resize in place. The 3D viewer is never
        # recreated, so resizing sidebars retains its OpenGL context.
        self.main_splitter = SidebarSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(14)
        self.main_splitter.addWidget(self.left_sidebar)
        self.main_splitter.addWidget(self.viewer_tabs)
        self.main_splitter.addWidget(self.right_sidebar)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([
            self._left_sidebar_last_width,
            900,
            self._right_sidebar_last_width,
        ])
        self.main_splitter.configure_left_sidebar_handle()
        self.main_splitter.configure_right_sidebar_handle()
        self.main_splitter.left_sidebar_toggle_requested.connect(
            self.toggle_left_sidebar
        )
        self.main_splitter.right_sidebar_toggle_requested.connect(
            self.toggle_right_sidebar
        )
        self.main_splitter.splitterMoved.connect(
            self.on_main_splitter_moved
        )
        self.setCentralWidget(self.main_splitter)
        if self._restore_left_sidebar_collapsed:
            restored_width = self._left_sidebar_last_width
            self.set_left_sidebar_visible(False)
            self._left_sidebar_last_width = restored_width
        else:
            self._sync_left_sidebar_controls()
        if self._restore_right_sidebar_collapsed:
            restored_width = self._right_sidebar_last_width
            self.set_right_sidebar_visible(False)
            self._right_sidebar_last_width = restored_width
        else:
            self._sync_right_sidebar_controls()
        self.render_progress_overlay = RenderProgressOverlay(self.viewer_3d_stack)
        self.tutorial_manager = TutorialManager(self)

        # Initial view
        if self.robot_model_3d is not None:
            self.pending_initial_render_progress = (
                f"Rendering {self.robot_model_3d.model_name}",
                "Preparing the 3D model for rendering...",
            )
        self.update_editor_context()
        self.refresh_display()
        self._refresh_history_baseline()

    def _on_document_dirty_changed(self, event):
        if event.document_id == self.document.document_id:
            self.set_project_dirty(event.dirty)

    def register_model_info(self, info):
        if info.key not in self.model_registry:
            self.model_registry[info.key] = info
            return info
        key = info.key
        index = 2
        while key in self.model_registry:
            key = f"{info.key}-{index}"
            index += 1
        info = replace(info, key=key)
        self.model_registry[info.key] = info
        return info

    # ============================================================
    # Application menus and workflow toolbar
    # ============================================================

    def build_menu_bar(self):
        menu_bar = self.menuBar()
        menu_bar.clear()
        menu_bar.setObjectName("appMenuBar")
        menu_bar.setNativeMenuBar(False)

        self.file_menu = menu_bar.addMenu("&File")
        self.new_project_action = QAction("&New Project…", self)
        self.new_project_action.setObjectName("newProjectAction")
        self.new_project_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_project_action.setToolTip("Create a GhostGUI project folder.")
        self.new_project_action.triggered.connect(self.on_new_project)
        self.file_menu.addAction(self.new_project_action)

        self.open_project_action = QAction("&Open Project…", self)
        self.open_project_action.setObjectName("openProjectAction")
        self.open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_project_action.setToolTip("Show GhostGUI project previews.")
        self.open_project_action.triggered.connect(self.on_open_project)
        self.file_menu.addAction(self.open_project_action)

        self.recent_projects_menu = self.file_menu.addMenu("Open &Recent")
        self.recent_projects_menu.setObjectName("recentProjectsMenu")
        self.recent_projects_menu.aboutToShow.connect(self.refresh_recent_projects)

        self.file_menu.addSeparator()
        self.save_project_action = QAction("&Save", self)
        self.save_project_action.setObjectName("saveProjectAction")
        self.save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_project_action.setToolTip("Save the current GhostGUI project.")
        self.save_project_action.triggered.connect(self.on_save_project)
        self.file_menu.addAction(self.save_project_action)

        self.file_menu.addSeparator()
        self.import_menu = self.file_menu.addMenu("&Import")
        self.import_menu.setObjectName("importMenu")
        self.import_actions = {}
        for label, action_key in (
            ("Robot Model…", "model"),
            ("Qpos…", "qpos"),
            ("Trajectory…", "trajectory"),
        ):
            action = QAction(label, self)
            action.setObjectName(f"import{action_key.title()}Action")
            action.setData(action_key)
            action.triggered.connect(
                lambda checked=False, key=action_key: (
                    self.on_setup_import_requested(key)
                )
            )
            self.import_menu.addAction(action)
            self.import_actions[action_key] = action

        self.export_menu = self.file_menu.addMenu("&Export")
        self.export_menu.setObjectName("exportMenu")
        self.export_actions = {}
        for label, action_key in (
            ("Qpos…", "qpos"),
            ("Trajectory…", "trajectory"),
        ):
            action = QAction(label, self)
            action.setObjectName(f"export{action_key.title()}Action")
            action.setData(action_key)
            action.triggered.connect(
                lambda checked=False, key=action_key: (
                    self.on_setup_export_requested(key)
                )
            )
            self.export_menu.addAction(action)
            self.export_actions[action_key] = action

        self.robot_menu = menu_bar.addMenu("&Robot")
        self.robot_menu.setObjectName("robotMenu")
        self.robot_menu.aboutToShow.connect(self.refresh_robot_menu)
        self.refresh_robot_menu()

        self.view_menu = menu_bar.addMenu("&View")
        self.view_menu.setObjectName("viewMenu")
        self.view_action_group = QActionGroup(self)
        self.view_action_group.setExclusive(True)
        self.view_actions = []
        for index in range(self.viewer_tabs.count()):
            action = QAction(self.viewer_tabs.tabText(index), self)
            action.setCheckable(True)
            action.setData(index)
            action.triggered.connect(
                lambda checked=False, tab_index=index: (
                    self.viewer_tabs.setCurrentIndex(tab_index)
                )
            )
            self.view_action_group.addAction(action)
            self.view_menu.addAction(action)
            self.view_actions.append(action)
        self.view_menu.addSeparator()

        self.left_sidebar_action = QAction("Left Sidebar", self)
        self.left_sidebar_action.setObjectName("leftSidebarAction")
        self.left_sidebar_action.setCheckable(True)
        self.left_sidebar_action.setChecked(True)
        self.left_sidebar_action.setToolTip(
            "Show or collapse the adjustable left editing sidebar."
        )
        self.left_sidebar_action.toggled.connect(
            self.set_left_sidebar_visible
        )
        self.view_menu.addAction(self.left_sidebar_action)

        self.right_sidebar_action = QAction("Right Sidebar", self)
        self.right_sidebar_action.setObjectName("rightSidebarAction")
        self.right_sidebar_action.setCheckable(True)
        self.right_sidebar_action.setChecked(True)
        self.right_sidebar_action.setToolTip(
            "Show or collapse the adjustable right inspector sidebar."
        )
        self.right_sidebar_action.toggled.connect(
            self.set_right_sidebar_visible
        )
        self.view_menu.addAction(self.right_sidebar_action)
        self.view_menu.addSeparator()

        self.model_colors_action = QAction("Use Model Colors", self)
        self.model_colors_action.setCheckable(True)
        self.model_colors_action.toggled.connect(
            lambda checked: self.viewer_3d.model_colors_box.setChecked(checked)
        )
        self.view_menu.addAction(self.model_colors_action)

        self.show_keyframes_action = QAction("Show Keyframes", self)
        self.show_keyframes_action.setCheckable(True)
        self.show_keyframes_action.toggled.connect(
            self.controls.show_keyframes_box.setChecked
        )
        self.controls.show_keyframes_box.toggled.connect(
            self.show_keyframes_action.setChecked
        )
        self.view_menu.addAction(self.show_keyframes_action)

        self.show_trajectory_lines_action = QAction("Show Trajectory Lines", self)
        self.show_trajectory_lines_action.setCheckable(True)
        self.show_trajectory_lines_action.toggled.connect(
            self.controls.show_lines_box.setChecked
        )
        self.controls.show_lines_box.toggled.connect(
            self.show_trajectory_lines_action.setChecked
        )
        self.view_menu.addAction(self.show_trajectory_lines_action)

        self.show_playback_poses_action = QAction("Show Playback Poses", self)
        self.show_playback_poses_action.setCheckable(True)
        self.show_playback_poses_action.toggled.connect(
            lambda checked: self.viewer_3d.show_ghosts.setChecked(checked)
        )
        self.view_menu.addAction(self.show_playback_poses_action)

        self.viewer_tabs.currentChanged.connect(self.sync_view_actions)
        self.sync_view_actions(self.viewer_tabs.currentIndex())
        self.sync_display_actions()

        self.help_menu = menu_bar.addMenu("&Help")
        self.help_menu.setObjectName("helpMenu")
        self.help_center_action = QAction("&Help Center…", self)
        self.help_center_action.setObjectName("helpCenterAction")
        self.help_center_action.setShortcut(QKeySequence(Qt.Key.Key_F1))
        self.help_center_action.triggered.connect(self.show_help_center)
        self.help_menu.addAction(self.help_center_action)
        self.start_tutorial_action = QAction(
            "Start First Motion &Tutorial", self
        )
        self.start_tutorial_action.setObjectName("startTutorialAction")
        self.start_tutorial_action.triggered.connect(
            self.start_first_motion_tutorial
        )
        self.help_menu.addAction(self.start_tutorial_action)

    def _toolbar_action(self, toolbar, text, icon_name, object_name, tooltip):
        action = QAction(theme_icon(icon_name, self), text, self)
        action.setObjectName(f"{object_name}Action")
        action.setToolTip(tooltip)
        toolbar.addAction(action)
        button = toolbar.widgetForAction(action)
        if button is not None:
            button.setObjectName(object_name)
        self._toolbar_icon_actions[action] = icon_name
        return action

    def build_workflow_toolbar(self):
        toolbar = QToolBar("Workflow", self)
        toolbar.setObjectName("workflowToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(16, 16))
        font = toolbar.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(6.0, font.pointSizeF() - 2.0))
        elif font.pixelSize() > 0:
            font.setPixelSize(max(8, font.pixelSize() - 2))
        toolbar.setFont(font)

        self._toolbar_icon_actions = {}
        self.preview_action = self._toolbar_action(
            toolbar,
            "Preview Path",
            "preview",
            "planPreviewButton",
            "Validate the path from the committed pose to the orange preview.",
        )
        self.preview_action.triggered.connect(
            lambda checked=False: self.viewer_3d.plan_preview()
        )
        self.slice_action = self._toolbar_action(
            toolbar,
            "Commit Keyframe",
            "slice",
            "sliceButton",
            "Commit the current pose as a keyframe and advance the timeline.",
        )
        self.slice_action.triggered.connect(
            lambda checked=False: self.viewer_3d.accept_timeslice()
        )
        self.generate_action = self._toolbar_action(
            toolbar,
            "Generate",
            "generate",
            "quickGenerateButton",
            "Generate a sampled trajectory from saved timeline states.",
        )
        self.generate_action.triggered.connect(
            lambda checked=False: self.viewer_3d.generate_requested.emit()
        )

        toolbar.addSeparator()
        self.playback_action = self._toolbar_action(
            toolbar,
            "Play",
            "play",
            "playbackToolbarButton",
            "Play or pause the active trajectory.",
        )
        self.playback_action.triggered.connect(
            lambda checked=False: self.viewer_3d.toggle_playback()
        )
        self.reset_action = self._toolbar_action(
            toolbar,
            "Reset",
            "reset",
            "resetToolbarButton",
            "Reset the active time to the model home pose.",
        )
        self.reset_action.triggered.connect(
            lambda checked=False: self.viewer_3d.reset_robot_pose()
        )
        self.clear_action = self._toolbar_action(
            toolbar,
            "Clear",
            "clear",
            "clearToolbarButton",
            "Clear the editable trajectory.",
        )
        self.clear_action.triggered.connect(
            lambda checked=False: self.viewer_3d.clear_trajectory_requested.emit()
        )

        toolbar.addSeparator()
        self.gizmo_action_group = QActionGroup(self)
        self.gizmo_action_group.setExclusive(True)
        self.move_action = self._toolbar_action(
            toolbar,
            "Move",
            "move",
            "moveToolButton",
            "Use the translation handles on the 3D transform gizmo (T).",
        )
        self.move_action.setCheckable(True)
        self.move_action.triggered.connect(
            lambda checked=False: self.set_gizmo_mode("translate")
        )
        self.gizmo_action_group.addAction(self.move_action)
        self.rotate_action = self._toolbar_action(
            toolbar,
            "Rotate",
            "rotate",
            "rotateToolButton",
            "Use the rotation rings on the 3D transform gizmo (R).",
        )
        self.rotate_action.setCheckable(True)
        self.rotate_action.triggered.connect(
            lambda checked=False: self.set_gizmo_mode("rotate")
        )
        self.gizmo_action_group.addAction(self.rotate_action)
        self.gizmo_visibility_action = self._toolbar_action(
            toolbar,
            "Gizmo",
            "gizmo",
            "gizmoVisibilityButton",
            "Show or hide the 3D transform gizmo.",
        )
        self.gizmo_visibility_action.setCheckable(True)
        self.gizmo_visibility_action.setChecked(True)
        self.gizmo_visibility_action.toggled.connect(
            self.sync_transform_gizmo_state
        )

        toolbar.addSeparator()
        self.undo_action = self._toolbar_action(
            toolbar,
            "Undo",
            "undo",
            "undoToolbarButton",
            "Undo the last edit (Ctrl+Z).",
        )
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.undo_action.triggered.connect(self.undo_last_action)
        self.redo_action = self._toolbar_action(
            toolbar,
            "Redo",
            "redo",
            "redoToolbarButton",
            "Redo the last undone edit (Ctrl+Shift+Z).",
        )
        self.redo_action.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        self.redo_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.redo_action.triggered.connect(self.redo_last_action)
        return toolbar

    def refresh_toolbar_icons(self):
        for action, icon_name in getattr(self, "_toolbar_icon_actions", {}).items():
            if (
                action is getattr(self, "playback_action", None)
                and self.viewer_3d.play_timer.isActive()
            ):
                icon_name = "pause"
            action.setIcon(theme_icon(icon_name, self))

    def sync_view_actions(self, active_index):
        for index, action in enumerate(getattr(self, "view_actions", [])):
            action.setChecked(index == active_index)
        self.sync_workflow_toolbar()

    def sync_display_actions(self):
        viewer = self.viewer_3d
        self.model_colors_action.setChecked(viewer.model_colors_box.isChecked())
        self.show_keyframes_action.setChecked(
            self.controls.show_keyframes_box.isChecked()
        )
        self.show_trajectory_lines_action.setChecked(
            self.controls.show_lines_box.isChecked()
        )
        self.show_playback_poses_action.setChecked(viewer.show_ghosts.isChecked())

    def _sync_left_sidebar_controls(self):
        if not hasattr(self, "main_splitter"):
            return
        self.main_splitter.set_left_sidebar_collapsed(
            self._left_sidebar_collapsed
        )
        action = getattr(self, "left_sidebar_action", None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(not self._left_sidebar_collapsed)
            action.blockSignals(False)

    def _sync_right_sidebar_controls(self):
        if not hasattr(self, "main_splitter"):
            return
        self.main_splitter.set_right_sidebar_collapsed(
            self._right_sidebar_collapsed
        )
        action = getattr(self, "right_sidebar_action", None)
        if action is not None:
            action.blockSignals(True)
            action.setChecked(not self._right_sidebar_collapsed)
            action.blockSignals(False)

    def set_left_sidebar_visible(self, visible):
        if not hasattr(self, "main_splitter"):
            return
        visible = bool(visible)
        if visible == (not self._left_sidebar_collapsed):
            self._sync_left_sidebar_controls()
            return

        sizes = self.main_splitter.sizes()
        self._syncing_left_sidebar = True
        try:
            if visible:
                self._left_sidebar_collapsed = False
                self.left_sidebar.setMinimumWidth(LEFT_SIDEBAR_MIN_WIDTH)
                target_width = max(
                    LEFT_SIDEBAR_MIN_WIDTH,
                    min(self._left_sidebar_last_width, SIDEBAR_MAX_WIDTH),
                )
                gained_width = max(0, target_width - sizes[0])
                sizes[0] = target_width
                sizes[1] = max(0, sizes[1] - gained_width)
                self.main_splitter.setCollapsible(0, False)
            else:
                if sizes[0] > 0:
                    self._left_sidebar_last_width = max(
                        LEFT_SIDEBAR_MIN_WIDTH,
                        min(sizes[0], SIDEBAR_MAX_WIDTH),
                    )
                released_width = sizes[0]
                self.left_sidebar.setMinimumWidth(0)
                self.main_splitter.setCollapsible(0, True)
                self._left_sidebar_collapsed = True
                sizes[0] = 0
                sizes[1] += released_width
            self.main_splitter.setSizes(sizes)
        finally:
            self._syncing_left_sidebar = False
        self._sync_left_sidebar_controls()

    def set_right_sidebar_visible(self, visible):
        if not hasattr(self, "main_splitter"):
            return
        visible = bool(visible)
        if visible == (not self._right_sidebar_collapsed):
            self._sync_right_sidebar_controls()
            return

        sizes = self.main_splitter.sizes()
        self._syncing_right_sidebar = True
        try:
            if visible:
                self._right_sidebar_collapsed = False
                self.right_sidebar.setMinimumWidth(RIGHT_SIDEBAR_MIN_WIDTH)
                target_width = max(
                    RIGHT_SIDEBAR_MIN_WIDTH,
                    min(self._right_sidebar_last_width, SIDEBAR_MAX_WIDTH),
                )
                gained_width = max(0, target_width - sizes[2])
                sizes[2] = target_width
                sizes[1] = max(0, sizes[1] - gained_width)
                self.main_splitter.setCollapsible(2, False)
            else:
                if sizes[2] > 0:
                    self._right_sidebar_last_width = max(
                        RIGHT_SIDEBAR_MIN_WIDTH,
                        min(sizes[2], SIDEBAR_MAX_WIDTH),
                    )
                released_width = sizes[2]
                self.right_sidebar.setMinimumWidth(0)
                self.main_splitter.setCollapsible(2, True)
                self._right_sidebar_collapsed = True
                sizes[2] = 0
                sizes[1] += released_width
            self.main_splitter.setSizes(sizes)
        finally:
            self._syncing_right_sidebar = False
        self._sync_right_sidebar_controls()

    def toggle_left_sidebar(self):
        self.set_left_sidebar_visible(self._left_sidebar_collapsed)

    def toggle_right_sidebar(self):
        self.set_right_sidebar_visible(self._right_sidebar_collapsed)

    def on_main_splitter_moved(self, _position, index):
        if self._syncing_left_sidebar or self._syncing_right_sidebar:
            return
        sizes = self.main_splitter.sizes()
        if index == 1:
            width = sizes[0]
            if width > SIDEBAR_MAX_WIDTH:
                excess = width - SIDEBAR_MAX_WIDTH
                self._syncing_left_sidebar = True
                try:
                    sizes[0] = SIDEBAR_MAX_WIDTH
                    sizes[1] += excess
                    self.main_splitter.setSizes(sizes)
                finally:
                    self._syncing_left_sidebar = False
                width = SIDEBAR_MAX_WIDTH
            if self._left_sidebar_collapsed and width > 0:
                self._left_sidebar_last_width = max(
                    LEFT_SIDEBAR_MIN_WIDTH,
                    min(width, SIDEBAR_MAX_WIDTH),
                )
                self.set_left_sidebar_visible(True)
                return
            if not self._left_sidebar_collapsed:
                self._left_sidebar_last_width = max(
                    LEFT_SIDEBAR_MIN_WIDTH,
                    min(width, SIDEBAR_MAX_WIDTH),
                )
            return
        if index == 2:
            width = sizes[2]
            if width > SIDEBAR_MAX_WIDTH:
                excess = width - SIDEBAR_MAX_WIDTH
                self._syncing_right_sidebar = True
                try:
                    sizes[2] = SIDEBAR_MAX_WIDTH
                    sizes[1] += excess
                    self.main_splitter.setSizes(sizes)
                finally:
                    self._syncing_right_sidebar = False
                width = SIDEBAR_MAX_WIDTH
            if self._right_sidebar_collapsed and width > 0:
                self._right_sidebar_last_width = max(
                    RIGHT_SIDEBAR_MIN_WIDTH,
                    min(width, SIDEBAR_MAX_WIDTH),
                )
                self.set_right_sidebar_visible(True)
                return
            if not self._right_sidebar_collapsed:
                self._right_sidebar_last_width = max(
                    RIGHT_SIDEBAR_MIN_WIDTH,
                    min(width, SIDEBAR_MAX_WIDTH),
                )

    def save_ui_settings(self):
        if hasattr(self, "main_splitter") and not self._left_sidebar_collapsed:
            width = self.main_splitter.sizes()[0]
            if width > 0:
                self._left_sidebar_last_width = max(
                    LEFT_SIDEBAR_MIN_WIDTH,
                    min(width, SIDEBAR_MAX_WIDTH),
                )
        if hasattr(self, "main_splitter") and not self._right_sidebar_collapsed:
            width = self.main_splitter.sizes()[2]
            if width > 0:
                self._right_sidebar_last_width = max(
                    RIGHT_SIDEBAR_MIN_WIDTH,
                    min(width, SIDEBAR_MAX_WIDTH),
                )
        ghostgui_config_dir().mkdir(parents=True, exist_ok=True)
        self.ui_settings.setValue(
            "left_sidebar/width",
            self._left_sidebar_last_width,
        )
        self.ui_settings.setValue(
            "left_sidebar/collapsed",
            self._left_sidebar_collapsed,
        )
        self.ui_settings.setValue(
            "right_sidebar/width",
            self._right_sidebar_last_width,
        )
        self.ui_settings.setValue(
            "right_sidebar/collapsed",
            self._right_sidebar_collapsed,
        )
        self.ui_settings.sync()

    def refresh_robot_menu(self):
        if not hasattr(self, "robot_menu"):
            return
        old_group = getattr(self, "robot_action_group", None)
        self.robot_menu.clear()
        self.robot_action_group = QActionGroup(self)
        self.robot_action_group.setExclusive(True)
        self.robot_actions = {}
        for key, info in self.model_registry.items():
            action = QAction(info.display_name, self.robot_action_group)
            action.setCheckable(True)
            action.setChecked(key == self.model_key)
            action.setData(key)
            action.triggered.connect(
                lambda checked=False, model_key=key: (
                    self.select_robot_from_menu(model_key)
                )
            )
            self.robot_menu.addAction(action)
            self.robot_actions[key] = action
        if old_group is not None:
            old_group.deleteLater()

    def select_robot_from_menu(self, model_key):
        index = self.controls.model_box.findData(model_key)
        if index >= 0:
            self.controls.model_box.setCurrentIndex(index)
        else:
            self.on_model_changed(model_key)

    def sync_robot_menu(self):
        for key, action in getattr(self, "robot_actions", {}).items():
            action.setChecked(key == self.model_key)

    def set_gizmo_mode(self, mode):
        tool_name = {
            "translate": "Move",
            "rotate": "Rotate",
        }.get(mode)
        manager = getattr(self, "visualization_manager", None)
        if manager is not None and tool_name is not None:
            manager.select_tool(tool_name)
            return
        self.viewer_3d.canvas.set_gizmo_mode(mode)

    def sync_transform_gizmo_state(self, _checked=None):
        if not hasattr(self, "gizmo_visibility_action"):
            return
        viewer = self.viewer_3d
        available = (
            viewer.robot_state is not None
            and self.viewer_tabs.currentWidget() is self.viewer_3d_stack
        )
        visible = available and self.gizmo_visibility_action.isChecked()
        viewer.canvas.set_transform_gizmo_visible(visible)
        viewer.canvas.set_transform_gizmo_interactive(visible)
        self.gizmo_visibility_action.setEnabled(available)
        self.move_action.setEnabled(visible)
        self.rotate_action.setEnabled(visible)

    def sync_workflow_toolbar(self):
        if not hasattr(self, "preview_action"):
            return
        viewer = self.viewer_3d
        has_robot = viewer.robot_state is not None
        for action in (
            self.preview_action,
            self.slice_action,
            self.generate_action,
            self.playback_action,
            self.reset_action,
            self.clear_action,
        ):
            action.setEnabled(has_robot)
        self.sync_transform_gizmo_state()
        mode = viewer.canvas.gizmo.mode
        self.move_action.setChecked(mode == "translate")
        self.rotate_action.setChecked(mode == "rotate")
        playing = viewer.play_timer.isActive()
        self.playback_action.setText("Pause" if playing else "Play")
        self.playback_action.setIcon(theme_icon("pause" if playing else "play", self))
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))

    # ============================================================
    # Build right status/debug panel
    # ============================================================

    def build_status_panel(self):
        panel = EditorStatusPanel(
            self.model_source_text(self.robot_model_3d)
        )
        for name, widget in panel.compatibility_widgets().items():
            setattr(self, name, widget)
        self._status_summary_signature = None
        self._status_repeat_count = 0
        self.apply_status_event(
            status_event_from_text(self.viewer_3d.status_label.text())
        )
        return panel

    def build_viewer_tabs(self):
        tabs = QTabWidget()
        self.viewer_3d_stack = QStackedWidget()
        self.viewer_3d_stack.addWidget(self.viewer_3d)
        tabs.addTab(self.viewer_3d_stack, "3D Pose")
        tabs.addTab(self.viewer_3d_mujoco, "Simulation")
        tabs.currentChanged.connect(self.update_editor_context)
        tabs.currentChanged.connect(lambda _index: self.mark_project_dirty("Active view"))
        tabs.setCurrentIndex(0)
        tabs.tabBar().hide()
        return tabs

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_render_progress_overlay_geometry()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.PaletteChange,
        ):
            QTimer.singleShot(0, self.refresh_toolbar_icons)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(
            INITIAL_RENDER_PROGRESS_DELAY_MS,
            self.prepare_pending_initial_render_progress,
        )

    def closeEvent(self, event):
        if not self.confirm_project_transition("close GhostGUI"):
            event.ignore()
            return
        self._shutting_down = True
        self.autosave_timer.stop()
        self.save_ui_settings()
        self.model_file_selection_stage.cancel()
        manager = getattr(self, "visualization_manager", None)
        if manager is not None:
            manager.shutdown()
        for loader in tuple(self.model_loaders.values()):
            loader.cancel_and_wait()
        self.model_loaders.clear()
        self.background_jobs.shutdown()
        for session in tuple(self.model_sessions.values()):
            session.close()
        self.viewer_3d_mujoco.shutdown()
        self._document_dirty_subscription.unsubscribe()
        self.editor_events.clear()
        super().closeEvent(event)

    def on_autosave_timer(self):
        try:
            self.autosave_current_project(show_status=False, reason="timer")
        except (OSError, ValueError) as exc:
            self.show_status_message(f"Project autosave failed: {exc}")

    def current_model_display_name(self):
        if self.robot_model_3d is not None:
            return self.robot_model_3d.model_name
        model_info = self.model_registry.get(self.model_key)
        return getattr(model_info, "display_name", self.model_key)

    def project_transition_reason(self, action):
        clean = "".join(
            character.lower() if character.isalnum() else "_"
            for character in str(action)
        )
        clean = "_".join(part for part in clean.split("_") if part)
        return f"before_{clean or 'transition'}"

    def update_project_chrome(self):
        title = "GhostGui"
        if self.current_project is not None:
            title = f"{title} - {self.current_project.project_name}"
        elif self.robot_model_3d is not None:
            title = f"{title} - {self.robot_model_3d.model_name}"
        if self.project_dirty:
            title = f"{title} *"
        self.setWindowTitle(title)

    def set_project_dirty(self, dirty=True):
        self.project_dirty = bool(dirty and self.current_project is not None)
        self.update_project_panel()

    def mark_project_dirty(self, reason=None):
        if self._suppress_project_dirty or self.current_project is None:
            return
        if not self.project_dirty:
            self.set_project_dirty(True)

    def handle_project_transition_choice(self, choice, action):
        if not self.project_dirty or self.current_project is None:
            return True

        reason = self.project_transition_reason(action)
        project = self.current_project
        if choice == "save":
            return self.save_current_project(reason=reason)
        if choice == "autosave":
            if not self.autosave_current_project(
                show_status=False,
                capture_snapshot=True,
                reason=reason,
            ):
                return False
            self.set_project_dirty(False)
            return True
        if choice == "discard":
            project.clear_autosave()
            project.append_session_event(
                "project_dirty_discarded",
                self.project_session_details(
                    {
                        "action": action,
                        "reason": reason,
                    }
                ),
            )
            self.set_project_dirty(False)
            return True
        if choice == "cancel":
            project.append_session_event(
                "project_transition_cancelled",
                self.project_session_details(
                    {
                        "action": action,
                        "reason": reason,
                    }
                ),
            )
        return False

    def confirm_project_transition(self, action):
        if not self.project_dirty or self.current_project is None:
            return True

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Unsaved Project Changes")
        dialog.setText(f"Save changes before you {action}?")
        dialog.setInformativeText(
            "Save updates the project files. Autosave stores recovery state "
            "for this project. Discard leaves the last saved project unchanged."
        )
        save_button = dialog.addButton(
            "Save",
            QMessageBox.ButtonRole.AcceptRole,
        )
        autosave_button = dialog.addButton(
            "Autosave",
            QMessageBox.ButtonRole.AcceptRole,
        )
        discard_button = dialog.addButton(
            "Discard",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(save_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()

        clicked = dialog.clickedButton()
        if clicked is save_button:
            choice = "save"
        elif clicked is autosave_button:
            choice = "autosave"
        elif clicked is discard_button:
            choice = "discard"
        else:
            choice = "cancel"
        return self.handle_project_transition_choice(choice, action)

    def project_session_details(self, extra=None):
        viewer = self.viewer_3d
        active_view = ""
        if 0 <= self.viewer_tabs.currentIndex() < self.viewer_tabs.count():
            active_view = self.viewer_tabs.tabText(self.viewer_tabs.currentIndex())
        details = {
            "model_key": self.model_key,
            "model_name": self.current_model_display_name(),
            "frame_count": len(self.trajectory.frames),
            "current_time": float(viewer.get_current_time()),
            "selected_frame": self.controls.frame_box.currentText(),
            "active_view": active_view,
        }
        if extra:
            details.update(extra)
        return details

    def update_project_panel(self):
        self.update_project_chrome()

    def recent_project_display_name(self, entry):
        project_path = Path(entry["path"])
        project_name = entry.get("project_name") or project_path.stem
        folder_name = project_path.name
        if folder_name.endswith(".ghostgui"):
            folder_name = folder_name[: -len(".ghostgui")]
        if folder_name and folder_name != project_name:
            return f"{project_name} ({folder_name})"
        return project_name

    def refresh_recent_projects(self):
        if not hasattr(self, "recent_projects_menu"):
            return
        entries = load_recent_projects()
        self.recent_projects_menu.clear()
        if not entries:
            empty_action = self.recent_projects_menu.addAction(
                "No recent projects"
            )
            empty_action.setEnabled(False)
            return
        for entry in entries:
            path = entry["path"]
            action = self.recent_projects_menu.addAction(
                self.recent_project_display_name(entry)
            )
            action.setData(path)
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, project_path=path: (
                    self.open_project_path(
                        project_path,
                        source="recent_projects",
                    )
                )
            )

    def remember_current_project(self):
        if self.current_project is None:
            return
        try:
            remember_recent_project(self.current_project)
            self.refresh_recent_projects()
        except OSError as exc:
            if hasattr(self, "status_text"):
                self.show_status_message(
                    f"Could not update recent projects: {exc}"
                )

    def build_project_browser_dialog(self):
        return ProjectBrowserDialog(load_project_browser_previews(), parent=self)

    def on_open_project(self):
        dialog = self.build_project_browser_dialog()
        if dialog.exec():
            if dialog.selected_project_path:
                self.open_project_path(
                    dialog.selected_project_path,
                    source="project_browser",
                )
            elif dialog.browse_requested:
                self.on_browse_project_folder()
        self.refresh_recent_projects()

    def on_project_browser(self):
        self.on_open_project()

    def on_new_project(self):
        if not self.confirm_project_transition("create a new project"):
            return
        default_name = (
            self.current_project.project_name
            if self.current_project is not None
            else f"{self.model_key}_project"
        )
        name, accepted = QInputDialog.getText(
            self,
            "New GhostGUI Project",
            "Project name",
            text=default_name,
        )
        if not accepted or not name.strip():
            return
        project_root = available_default_project_root_from_name(name)
        try:
            self.create_project_at(
                project_root,
                name.strip(),
                reset_workspace=True,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Create project failed", str(exc))

    def on_browse_project_folder(self):
        if self.background_jobs.is_busy():
            self.show_status_message(
                "Wait for the current import or export to finish."
            )
            return
        if self.file_selection_stage.is_active():
            self.show_status_message("A file selector is already open.")
            return
        self.model_file_selection_stage.select_file(
            mode="directory",
            title="Open GhostGUI Project",
            directory=Path.home(),
            name_filter="Folders (*)",
            selected=lambda path: self.open_project_path(
                path, source="folder_dialog"
            ),
            failed=lambda message: QMessageBox.warning(
                self, "Open project failed", message
            ),
        )

    def on_save_project(self):
        if self.current_project is None:
            default_name = f"{self.model_key}_project"
            name, accepted = QInputDialog.getText(
                self,
                "Save GhostGUI Project",
                "Project name",
                text=default_name,
            )
            if not accepted or not name.strip():
                return
            project_root = available_default_project_root_from_name(name)
            try:
                self.create_project_at(project_root, name.strip())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Save project failed", str(exc))
            return
        try:
            self.save_current_project()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save project failed", str(exc))

    def reset_workspace_for_new_project(self):
        was_suppressing_dirty = self._suppress_project_dirty
        self._suppress_project_dirty = True
        try:
            viewer = self.viewer_3d
            viewer.pause_playback()
            viewer.canvas.cancel_transform_drag()
            self.trajectory.clear()
            self.active_index = -1
            viewer.clear_robot_trajectory()
            self.viewer_3d_mujoco.clear_trajectory()
            self.backend_interface.clear_last_solution()
            viewer.clear_editable_timeline(keep_current_pose=False, reset_time=0.0)
            viewer.preview_active = False
            viewer.canvas.set_preview_visible(False)
            viewer.canvas.camera_distance = 5.0
            viewer.canvas.camera_yaw = 38.0
            viewer.canvas.camera_pitch = 24.0
            viewer.canvas.camera_center = np.asarray([0.0, 0.0, 0.75], dtype=float)
            viewer.canvas.update()
            self.set_editor_timeline_duration(5.0)
            self.controls.time_slider.set_value(0.0)
            self.controls.show_keyframes_box.setChecked(True)
            self.controls.show_lines_box.setChecked(True)
            self.controls.corner_smoothing_slider.set_value(0.0)
            viewer.show_ghosts.setChecked(False)
            if self.controls.table.currentRow() >= 0:
                self.controls.table.clearSelection()
            self.set_current_frame_to_model_reference(
                self.controls.frame_box.currentText(),
                emit_pose_changed=False,
            )
            self.refresh_display()
            self._refresh_history_baseline()
        finally:
            self._suppress_project_dirty = was_suppressing_dirty

    def create_project_at(self, project_root, project_name, reset_workspace=False):
        if reset_workspace:
            self.reset_workspace_for_new_project()
        project = GhostGUIProject.create(
            project_root,
            project_name,
            self.model_key,
            self.current_model_display_name(),
        )
        self.current_project = project
        self.save_current_project(reason="initial_create")
        return project

    def save_current_project(
        self,
        show_status=True,
        capture_snapshot=True,
        reason="manual",
    ):
        project = self.current_project
        if project is None:
            return False

        viewer = self.viewer_3d
        if viewer.state_timeline is not None and viewer.committed_state is not None:
            viewer.state_timeline.set_state(
                viewer.get_current_time(),
                viewer.committed_state.get_qpos(),
            )

        project.update_robot(self.model_key, self.current_model_display_name())
        snapshot_saved = False
        if capture_snapshot:
            snapshot_saved = bool(self.save_project_snapshot(project))
        project.save_bundle(
            self.trajectory,
            viewer.state_timeline,
            self.capture_project_workspace(),
        )
        project.clear_autosave()
        project.append_session_event(
            "project_saved",
            self.project_session_details(
                {
                    "reason": reason,
                    "snapshot_saved": snapshot_saved,
                }
            ),
        )
        self.editor_controller.mark_saved()
        self.set_project_dirty(False)
        self.remember_current_project()
        if show_status:
            message = f"Saved project: {project.root_dir}"
            self.show_status_message(message)
            self.statusBar().showMessage(message, 3000)
        return True

    def autosave_current_project(
        self,
        show_status=True,
        capture_snapshot=False,
        reason="manual",
    ):
        project = self.current_project
        if project is None:
            return False

        viewer = self.viewer_3d
        if viewer.state_timeline is None or viewer.committed_state is None:
            return False

        viewer.state_timeline.set_state(
            viewer.get_current_time(),
            viewer.committed_state.get_qpos(),
        )
        project.write_autosave(
            self.trajectory,
            viewer.state_timeline,
            self.capture_project_workspace(),
            self.model_key,
            self.current_model_display_name(),
        )
        snapshot_saved = False
        if capture_snapshot:
            snapshot_saved = bool(self.save_project_snapshot(project))
        project.append_session_event(
            "project_autosaved",
            self.project_session_details(
                {
                    "reason": reason,
                    "snapshot_saved": snapshot_saved,
                }
            ),
        )
        if show_status:
            message = f"Autosaved project recovery state: {project.root_dir}"
            self.statusBar().showMessage(message, 3000)
        return True

    def save_project_snapshot(self, project):
        try:
            path = project.paths.last_snapshot
            path.parent.mkdir(parents=True, exist_ok=True)
            pixmap = self.grab()
            if not pixmap.isNull():
                return pixmap.save(str(path))
        except Exception:
            return False
        return False

    def capture_project_workspace(self):
        viewer = self.viewer_3d
        canvas = viewer.canvas
        selected_kind, selected_name = viewer._selected_target()
        return {
            "schema_version": 1,
            "active_index": int(self.active_index),
            "active_view_index": int(self.viewer_tabs.currentIndex()),
            "active_view": self.viewer_tabs.tabText(
                self.viewer_tabs.currentIndex()
            ),
            "current_time": float(viewer.get_current_time()),
            "selected_frame": self.controls.frame_box.currentText(),
            "selected_row": int(self.controls.selected_row()),
            "timeline_duration": float(viewer.timeline_duration),
            "control_frame": self.controls.current_frame().to_dict(),
            "target_selection": {
                "kind": selected_kind,
                "name": selected_name,
            },
            "camera": {
                "yaw": float(canvas.camera_yaw),
                "pitch": float(canvas.camera_pitch),
                "distance": float(canvas.camera_distance),
                "center": [float(value) for value in canvas.camera_center],
            },
            "display": {
                "show_keyframes": bool(self.controls.show_keyframes()),
                "show_trajectory_lines": bool(self.controls.show_trajectory_lines()),
                "smoothing": float(self.controls.corner_smoothing()),
                "show_playback_ghosts": bool(viewer.show_ghosts.isChecked()),
                "trajectory_import_dt": float(viewer.trajectory_import_dt.value()),
            },
        }

    def open_project_path(self, path, source="direct"):
        try:
            project = GhostGUIProject.open(path)
        except (OSError, ValueError) as exc:
            try:
                forget_recent_project(path)
                self.refresh_recent_projects()
            except OSError:
                pass
            QMessageBox.warning(self, "Open project failed", str(exc))
            return False

        if not self.confirm_project_transition(f"open {project.project_name}"):
            return False

        restore_autosave = self.should_restore_project_autosave(project)
        model_key = self.project_restore_model_key(project, restore_autosave)

        if model_key != self.model_key:
            if model_key not in self.model_registry:
                message = f"Project robot model is not available: {model_key}"
                QMessageBox.warning(self, "Open project failed", message)
                return False
            self._pending_project_restore = project
            self._pending_project_restore_autosave = restore_autosave
            self._pending_project_restore_source = source
            self.on_model_changed(model_key)
            self.restore_pending_project_if_ready()
            return True

        return self.restore_project(
            project,
            autosave=restore_autosave,
            source=source,
        )

    def should_restore_project_autosave(self, project):
        if not project.is_autosave_newer():
            return False
        response = QMessageBox.question(
            self,
            "Restore project autosave?",
            (
                "A newer autosave exists for this project. "
                "Restore the autosaved workspace?\n\n"
                "Choose No to discard the autosave and open the last saved project."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if response == QMessageBox.StandardButton.Yes:
            return True
        project.clear_autosave()
        project.append_session_event(
            "project_autosave_discarded",
            {
                "model_key": project.model_key,
                "source": "open_project",
            },
        )
        return False

    def project_restore_model_key(self, project, autosave=False):
        if not autosave:
            return project.model_key
        try:
            return project.autosave_model_key()
        except (OSError, TypeError, ValueError):
            return project.model_key

    def restore_pending_project_if_ready(self):
        project = self._pending_project_restore
        autosave = self._pending_project_restore_autosave
        source = self._pending_project_restore_source
        if project is None:
            return False
        model_key = self.project_restore_model_key(project, autosave)
        if model_key != self.model_key:
            return False
        self._pending_project_restore = None
        self._pending_project_restore_autosave = False
        self._pending_project_restore_source = "direct"
        return self.restore_project(project, autosave=autosave, source=source)

    def restore_project(self, project, autosave=False, source="direct"):
        try:
            trajectory_data = project.read_trajectory_dict(autosave=autosave)
            workspace = project.read_workspace(autosave=autosave)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Open project failed", str(exc))
            return False
        if not isinstance(workspace, dict):
            QMessageBox.warning(
                self,
                "Open project failed",
                "Project workspace data must be a JSON object.",
            )
            return False

        viewer = self.viewer_3d
        viewer.pause_playback()
        viewer.canvas.cancel_transform_drag()
        viewer.clear_robot_trajectory()
        self.viewer_3d_mujoco.clear_trajectory()
        self.backend_interface.clear_last_solution()
        try:
            self.trajectory.load_project_dict(trajectory_data)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Open project failed", str(exc))
            return False

        if viewer.state_timeline is not None:
            qpos_path = (
                project.autosave_paths.qpos_timeline
                if autosave else project.paths.qpos_timeline
            )
            if qpos_path.exists():
                try:
                    project.load_qpos_timeline(
                        viewer.state_timeline,
                        autosave=autosave,
                    )
                except ValueError as exc:
                    QMessageBox.warning(self, "Open project failed", str(exc))
                    return False
            else:
                viewer.clear_editable_timeline(keep_current_pose=True, reset_time=0.0)

        timeline_duration = workspace.get("timeline_duration")
        if timeline_duration is not None:
            self.set_editor_timeline_duration(float(timeline_duration))

        frame_data = workspace.get("control_frame")
        if frame_data:
            try:
                self._restore_control_frame(TargetFrame.from_dict(dict(frame_data)))
            except (TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Open project failed", str(exc))
                return False

        was_suppressing_dirty = self._suppress_project_dirty
        self._suppress_project_dirty = True
        try:
            self.restore_project_workspace(workspace)
        finally:
            self._suppress_project_dirty = was_suppressing_dirty

        current_time = float(workspace.get("current_time", 0.0))
        viewer.set_current_time(current_time)
        self.controls._suppress_pose_changed = True
        try:
            self.controls.time_slider.set_value(current_time)
        finally:
            self.controls._suppress_pose_changed = False

        self.active_index = int(workspace.get("active_index", -1))
        self.refresh_display()
        selected_row = int(workspace.get("selected_row", -1))
        if 0 <= selected_row < self.controls.table.rowCount():
            self.controls.table.setCurrentCell(selected_row, 0)
        else:
            self.controls.table.clearSelection()

        self.current_project = project
        self.set_project_dirty(False)
        self.remember_current_project()
        project.append_session_event(
            "project_opened",
            self.project_session_details(
                {
                    "autosave": bool(autosave),
                    "source": source,
                }
            ),
        )
        self.history.clear()
        self._refresh_history_baseline()
        self.sync_workflow_toolbar()
        source = "autosaved project" if autosave else "project"
        message = f"Opened {source}: {project.root_dir}"
        self.show_status_message(message)
        self.statusBar().showMessage(message, 3000)
        return True

    def restore_project_workspace(self, workspace):
        display = workspace.get("display", {})
        if "show_keyframes" in display:
            self.controls.show_keyframes_box.setChecked(bool(display["show_keyframes"]))
        if "show_trajectory_lines" in display:
            self.controls.show_lines_box.setChecked(
                bool(display["show_trajectory_lines"])
            )
        if "smoothing" in display:
            self.controls.corner_smoothing_slider.set_value(float(display["smoothing"]))
        if "trajectory_import_dt" in display:
            self.viewer_3d.trajectory_import_dt.setValue(
                float(display["trajectory_import_dt"])
            )
        if "show_playback_ghosts" in display:
            self.viewer_3d.show_ghosts.setChecked(
                bool(display["show_playback_ghosts"])
            )

        target = workspace.get("target_selection", {})
        target_kind = target.get("kind")
        target_name = target.get("name")
        if target_kind and target_name:
            self.viewer_3d.select_target(target_kind, target_name, emit=False)
        else:
            selected_frame = workspace.get("selected_frame")
            binding = self.viewer_3d.frame_bindings.get(selected_frame)
            if binding is not None:
                self.viewer_3d.select_target(*binding, emit=False)

        camera = workspace.get("camera", {})
        canvas = self.viewer_3d.canvas
        if "yaw" in camera:
            canvas.camera_yaw = float(camera["yaw"])
        if "pitch" in camera:
            canvas.camera_pitch = float(camera["pitch"])
        if "distance" in camera:
            canvas.camera_distance = float(camera["distance"])
        center = camera.get("center")
        if isinstance(center, list) and len(center) == 3:
            canvas.camera_center = np.asarray(center, dtype=float)
        canvas.update()

        active_view_name = str(workspace.get("active_view") or "")
        restored_view_index = -1
        for index in range(self.viewer_tabs.count()):
            if self.viewer_tabs.tabText(index) == active_view_name:
                restored_view_index = index
                break
        if restored_view_index < 0:
            legacy_view_index = int(workspace.get("active_view_index", 0))
            # Before the 2D viewers were removed, Simulation was index 3.
            restored_view_index = 1 if legacy_view_index == 3 else 0
        self.viewer_tabs.setCurrentIndex(restored_view_index)

    def show_help_center(self):
        if self.help_dialog is None:
            self.help_dialog = HelpCenterDialog(self)
            self.help_dialog.start_tutorial_requested.connect(
                self.start_first_motion_tutorial
            )
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def start_first_motion_tutorial(self):
        if self.help_dialog is not None:
            self.help_dialog.hide()
        self.tutorial_manager.start_first_motion()

    def prepare_pending_initial_render_progress(self):
        if (
            not self.isVisible()
            or self.viewer_3d_stack.width() <= 1
            or self.viewer_3d_stack.height() <= 1
        ):
            QTimer.singleShot(50, self.prepare_pending_initial_render_progress)
            return
        if QApplication.platformName().lower() != "offscreen":
            self.raise_()
            self.activateWindow()
        QTimer.singleShot(0, self.show_pending_initial_render_progress)

    def update_render_progress_overlay_geometry(self):
        if self.render_progress_overlay is None:
            return
        self.render_progress_overlay.update_geometry()

    def present_render_progress_overlay(self):
        if self.render_progress_overlay is None:
            return
        self.render_progress_overlay.update_geometry()
        self.render_progress_overlay.show()
        self.render_progress_overlay.raise_()

    def show_pending_initial_render_progress(self):
        if self.pending_initial_render_progress is None:
            self.request_active_model_render()
            return
        title, detail = self.pending_initial_render_progress
        self.pending_initial_render_progress = None
        self.begin_render_progress(title, detail, viewer=self.viewer_3d)
        self.request_active_model_render()

    def begin_render_progress(self, title, detail="", viewer=None, progress=None):
        if self.render_progress_overlay is None:
            return
        self.render_progress_viewer = viewer
        self.render_progress_overlay.show_rendering(title, detail, progress)
        self.present_render_progress_overlay()
        QTimer.singleShot(0, self.present_render_progress_overlay)

    def update_render_progress(self, title, detail="", progress=None):
        if self.render_progress_overlay is None:
            return
        self.render_progress_overlay.show_rendering(title, detail, progress)
        self.present_render_progress_overlay()

    def finish_render_progress(self):
        if self.render_progress_overlay is not None:
            self.render_progress_overlay.finish()
        self.render_progress_viewer = None
        if self.render_progress_restore_widget is not None:
            restore_widget = self.render_progress_restore_widget
            self.render_progress_restore_widget = None
            if self.viewer_tabs.indexOf(restore_widget) >= 0:
                self.viewer_tabs.setCurrentWidget(restore_widget)
        else:
            self.render_progress_restore_widget = None

    def active_3d_geometry_ready(self):
        viewer = self.viewer_3d
        canvas = viewer.canvas
        if viewer.robot_state is None:
            return True
        if not canvas.isValid():
            return False
        if canvas._geometry_queue:
            return False
        return canvas._geometry_build_count > 0

    def request_active_model_render(self):
        if (
            self.render_progress_overlay is None
            or self.render_progress_overlay.isHidden()
        ):
            return
        self.render_progress_viewer = self.viewer_3d
        if self.active_3d_geometry_ready():
            QTimer.singleShot(0, self.finish_render_progress)
            return
        if not self.isVisible():
            return
        if self.viewer_tabs.currentWidget() is not self.viewer_3d_stack:
            self.render_progress_restore_widget = self.viewer_tabs.currentWidget()
            self.viewer_tabs.setCurrentWidget(self.viewer_3d_stack)
        self.viewer_3d.canvas.update()

    def on_viewer_geometry_progress(self, viewer, complete, total):
        if viewer is not self.viewer_3d:
            return
        if (
            self.render_progress_overlay is not None
            and not self.render_progress_overlay.isHidden()
            and self.render_progress_viewer not in (None, viewer)
        ):
            return
        if total <= 0:
            self.finish_render_progress()
            return

        progress = round((float(complete) / float(total)) * 100.0)
        model_name = (
            self.robot_model_3d.model_name
            if self.robot_model_3d is not None
            else "robot model"
        )
        self.update_render_progress(
            f"Rendering {model_name}",
            f"Building 3D geometry {complete}/{total}...",
            progress=progress,
        )
        self.render_progress_viewer = viewer
        if complete >= total:
            self.finish_render_progress()

    def model_source_text(self, adapter):
        path = getattr(adapter, "model_path", None)
        if path is None:
            return "Model source: unavailable"
        return f"Model source: {path}"

    def update_editor_context(self, index=None):
        active = self.viewer_tabs.currentWidget()
        self.controls.set_joint_editor_widget(
            self.viewer_3d.joint_editor_widget()
        )
        if active is self.viewer_3d_stack:
            self.controls.set_robot_context_widget(
                self.viewer_3d.robot_context_widget()
            )
            self.controls.set_target_context_widget(
                self.viewer_3d.target_context_widget()
            )
            self.controls.set_trajectory_context_widget(
                self.viewer_3d.trajectory_context_widget()
            )
            self.controls.set_timeslice_context_widget(
                self.viewer_3d.timeslice_context_widget()
            )
            self.controls.set_display_context_widget(
                self.viewer_3d.display_context_widget()
            )
            self.viewer_3d.set_trajectory_display_widgets(
                self.controls.show_keyframes_box,
                self.controls.show_lines_box,
            )
            self.controls.set_preview_ik_context_widget(
                self.viewer_3d.preview_ik_context_widget()
            )
            self.sync_viewer_status_panel()
        else:
            self.controls.set_robot_context_widget(None)
            self.controls.set_target_context_widget(None)
            self.controls.set_trajectory_context_widget(None)
            self.controls.set_timeslice_context_widget(None)
            self.controls.set_display_context_widget(None)
            self.controls.set_preview_ik_context_widget(None)
        self.sync_workflow_toolbar()

    # ============================================================
    # Signal connections
    # ============================================================

    def connect_signals(self):
        self.controls.model_changed.connect(self.on_model_changed)
        self.controls.open_model_clicked.connect(self.on_open_model_file)
        self.controls.choose_mesh_folder_clicked.connect(self.on_choose_mesh_folder)
        self.controls.setup_import_requested.connect(self.on_setup_import_requested)
        self.controls.setup_export_requested.connect(self.on_setup_export_requested)
        self.controls.editing_mode_changed.connect(self.on_editing_mode_changed)
        self.controls.pose_changed.connect(self.on_pose_changed)
        self.controls.add_keyframe_clicked.connect(self.on_add_keyframe)
        self.controls.update_keyframe_clicked.connect(self.on_update_keyframe)
        self.controls.delete_keyframe_clicked.connect(self.on_delete_keyframe)
        self.controls.clear_trajectory_clicked.connect(self.on_clear_trajectory)
        self.controls.generate_clicked.connect(self.on_generate_trajectory)
        self.controls.keyframe_selected.connect(self.on_keyframe_selected)
        self.controls.frame_name_changed.connect(self.on_frame_name_changed)
        self.controls.trajectory_lines_changed.connect(
            self.on_trajectory_lines_changed
        )
        self.controls.keyframes_visibility_changed.connect(
            self.on_trajectory_display_changed
        )
        self.controls.time_changed.connect(self.on_time_changed)

        self.connect_model_viewer_signals(self.viewer_3d)

    def connect_model_viewer_signals(self, viewer_3d):
        viewer_3d.target_dragged.connect(self.on_target_dragged)
        viewer_3d.target_pose_dragged.connect(self.on_target_pose_dragged)
        viewer_3d.target_pose_drag_finished.connect(
            self.on_target_pose_drag_finished
        )
        viewer_3d.target_frame_changed.connect(self.on_3d_target_frame_changed)
        viewer_3d.preview_cancelled.connect(self.on_preview_cancelled)
        viewer_3d.trajectory_csv_loaded.connect(self.on_trajectory_csv_loaded)
        viewer_3d.generate_requested.connect(self.on_generate_trajectory)
        viewer_3d.clear_trajectory_requested.connect(self.on_clear_trajectory)
        viewer_3d.canvas.geometry_progress.connect(
            lambda complete, total, viewer=viewer_3d: (
                self.on_viewer_geometry_progress(viewer, complete, total)
            )
        )
        viewer_3d.canvas.camera_changed.connect(
            lambda: self.mark_project_dirty("Camera")
        )
        viewer_3d.show_ghosts.toggled.connect(
            lambda _checked: self.mark_project_dirty("Display settings")
        )
        viewer_3d.show_ghosts.toggled.connect(
            lambda checked, viewer=viewer_3d: (
                self.on_viewer_playback_poses_changed(viewer, checked)
            )
        )
        viewer_3d.model_colors_box.toggled.connect(
            lambda checked, viewer=viewer_3d: (
                self.on_viewer_model_colors_changed(viewer, checked)
            )
        )
        viewer_3d.playback_state_changed.connect(
            lambda _playing, viewer=viewer_3d: (
                self.on_viewer_playback_state_changed(viewer)
            )
        )
        viewer_3d.canvas.gizmo_mode_changed.connect(
            lambda _mode, viewer=viewer_3d: (
                self.on_viewer_gizmo_mode_changed(viewer)
            )
        )
        viewer_3d.trajectory_import_dt.valueChanged.connect(
            lambda _value: self.mark_project_dirty("Import time step")
        )
        viewer_3d.timeslice_preview_time_changed.connect(
            self.on_viewer_timeslice_preview_time_changed
        )
        viewer_3d.timeslice_time_changed.connect(
            self.on_viewer_timeslice_time_changed
        )
        viewer_3d.timeline_duration_changed.connect(
            self.on_viewer_timeline_duration_changed
        )
        viewer_3d.accept_timeslice_requested.connect(
            self.on_accept_timeslice_requested
        )
        viewer_3d.delete_timeslice_requested.connect(
            self.on_delete_timeslice_requested
        )
        viewer_3d.history_action_finished.connect(
            self.on_viewer_history_action_finished
        )
        viewer_3d.status_label.text_changed.connect(
            lambda text, viewer=viewer_3d: self.on_viewer_status_changed(
                viewer, text
            )
        )
        viewer_3d.timeline_state_label.text_changed.connect(
            lambda text, viewer=viewer_3d: self.on_viewer_time_changed(
                viewer, text
            )
        )
        viewer_3d.root_pose_label.text_changed.connect(
            lambda text, viewer=viewer_3d: self.on_viewer_root_pose_changed(
                viewer, text
            )
        )
    def on_viewer_playback_poses_changed(self, viewer, checked):
        if viewer is self.viewer_3d:
            self.show_playback_poses_action.setChecked(bool(checked))

    def on_viewer_model_colors_changed(self, viewer, checked):
        if viewer is self.viewer_3d:
            self.model_colors_action.setChecked(bool(checked))

    def on_viewer_playback_state_changed(self, viewer):
        if viewer is self.viewer_3d:
            self.sync_workflow_toolbar()

    def on_viewer_gizmo_mode_changed(self, viewer):
        if viewer is self.viewer_3d:
            self.sync_workflow_toolbar()

    def on_editing_mode_changed(self, _mode):
        self.sync_workflow_toolbar()

    def capture_history_snapshot(self):
        viewer = self.viewer_3d
        timeline_states = ()
        if viewer.state_timeline is not None:
            timeline_states = tuple(
                (time, viewer.state_timeline.get_state(time))
                for time in viewer.state_timeline.times()
            )

        committed_qpos = (
            None if viewer.committed_state is None
            else viewer.committed_state.get_qpos()
        )
        preview_qpos = (
            None if viewer.preview_state is None
            else viewer.preview_state.get_qpos()
        )

        return GuiHistorySnapshot(
            trajectory_frames=tuple(
                frame.to_dict() for frame in self.trajectory.frames
            ),
            trajectory_track_names=tuple(self.trajectory.tracks.keys()),
            active_index=int(self.active_index),
            control_frame=self.controls.current_frame().to_dict(),
            selected_row=int(self.controls.selected_row()),
            current_time=float(viewer.get_current_time()),
            timeline_states=timeline_states,
            committed_qpos=committed_qpos,
            preview_qpos=preview_qpos,
            preview_active=bool(viewer.preview_active),
            robot_trajectory=tuple(
                qpos.copy() for qpos in viewer.robot_trajectory
            ),
            robot_trajectory_times=tuple(float(t) for t in viewer.robot_trajectory_times),
            ghost_trajectory=tuple(qpos.copy() for qpos in viewer.ghost_trajectory),
            ghost_source=viewer.ghost_source,
            show_ghosts=bool(viewer.show_ghosts.isChecked()),
            timeline_duration=float(viewer.timeline_duration),
        )

    def _restore_control_frame(self, frame):
        controls = self.controls
        controls._suppress_pose_changed = True
        try:
            controls.time_slider.set_value(frame.time)
            controls.x_slider.set_value(frame.x)
            controls.y_slider.set_value(frame.y)
            controls.z_slider.set_value(frame.z)
            controls.roll_slider.set_value(frame.roll)
            controls.pitch_slider.set_value(frame.pitch)
            controls.yaw_slider.set_value(frame.yaw)
            controls.phase_box.setCurrentText(frame.phase)
            blocked = controls.frame_box.blockSignals(True)
            controls.frame_box.setCurrentText(frame.frame_name)
            controls.frame_box.blockSignals(blocked)
        finally:
            controls._suppress_pose_changed = False

    def restore_history_snapshot(self, snapshot):
        viewer = self.viewer_3d
        self._history_restoring = True
        try:
            self.trajectory.tracks = {
                name: [] for name in snapshot.trajectory_track_names
            }
            for frame_data in snapshot.trajectory_frames:
                self.trajectory.add_frame(TargetFrame.from_dict(dict(frame_data)))
            self.active_index = snapshot.active_index

            if viewer.robot_state is not None:
                viewer.pause_playback()
                viewer.canvas.cancel_transform_drag()
                if snapshot.robot_trajectory:
                    viewer.set_robot_trajectory(
                        snapshot.robot_trajectory,
                        times=snapshot.robot_trajectory_times,
                        activate_first_frame=False,
                    )
                    self.viewer_3d_mujoco.clear_trajectory()
                    self.viewer_3d_mujoco.set_trajectory_metadata(
                        mujoco_playback_cache_path(),
                        snapshot.robot_trajectory_times,
                    )
                else:
                    viewer.clear_robot_trajectory()
                    self.viewer_3d_mujoco.clear_trajectory()
                    self.backend_interface.clear_last_solution()
                self.set_editor_timeline_duration(snapshot.timeline_duration)
                show_ghosts_blocked = viewer.show_ghosts.blockSignals(True)
                try:
                    viewer.show_ghosts.setChecked(snapshot.show_ghosts)
                finally:
                    viewer.show_ghosts.blockSignals(show_ghosts_blocked)
                self.show_playback_poses_action.setChecked(snapshot.show_ghosts)
                viewer.ghost_trajectory = [
                    qpos.copy() for qpos in snapshot.ghost_trajectory
                ]
                viewer.ghost_source = snapshot.ghost_source
                viewer._rebuild_ghosts()

                if viewer.state_timeline is not None:
                    viewer.state_timeline.states.clear()
                    for time, qpos in snapshot.timeline_states:
                        viewer.state_timeline.set_state(time, qpos)
                    if not snapshot.timeline_states and snapshot.committed_qpos is not None:
                        viewer.state_timeline.set_state(
                            snapshot.current_time, snapshot.committed_qpos
                        )
                    viewer.current_time = viewer.state_timeline.time_key(
                        snapshot.current_time
                    )
                    viewer.display_time = viewer.current_time
                    viewer._set_timeslice_widgets(viewer.current_time)
                    viewer._update_frame_readout(viewer.current_time)
                    viewer._update_timeline_label()

                if snapshot.committed_qpos is not None:
                    viewer.committed_state.set_qpos(snapshot.committed_qpos)
                if snapshot.preview_qpos is not None:
                    viewer.preview_state.set_qpos(snapshot.preview_qpos)
                viewer.preview_active = snapshot.preview_active
                viewer._use_editor_canvas_states()
                viewer.canvas.set_preview_visible(snapshot.preview_active)
                viewer._sync_joint_controls()
                viewer._set_target_to_selected_pose()

            control_frame = TargetFrame.from_dict(dict(snapshot.control_frame))
            self._restore_control_frame(control_frame)
            self.refresh_display()
            if 0 <= snapshot.selected_row < self.controls.table.rowCount():
                self.controls.table.setCurrentCell(snapshot.selected_row, 0)
            else:
                self.controls.table.clearSelection()
        finally:
            self._history_restoring = False

    def _refresh_history_baseline(self):
        if self._history_restoring:
            return
        self.history.set_baseline(self.capture_history_snapshot())

    def record_history_action(self, description):
        if self._history_restoring:
            return False
        before = self.history.baseline or self.capture_history_snapshot()
        after = self.capture_history_snapshot()
        self.history.record(
            description,
            before=before,
            after=after,
        )
        self.statusBar().showMessage(f"{description}; Ctrl+Z can undo.", 3000)
        self.mark_project_dirty(description)
        self.sync_workflow_toolbar()
        return True

    def undo_last_action(self):
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo.", 2000)
            return
        transition = self.history.undo(self.capture_history_snapshot())
        self.restore_history_snapshot(transition.target)
        self.statusBar().showMessage(
            f"Undid {transition.description}.", 3000
        )
        self.mark_project_dirty(f"Undo {transition.description}")
        self.sync_workflow_toolbar()

    def redo_last_action(self):
        if not self.redo_stack:
            self.statusBar().showMessage("Nothing to redo.", 2000)
            return
        transition = self.history.redo(self.capture_history_snapshot())
        self.restore_history_snapshot(transition.target)
        self.statusBar().showMessage(
            f"Redid {transition.description}.", 3000
        )
        self.mark_project_dirty(f"Redo {transition.description}")
        self.sync_workflow_toolbar()

    def on_viewer_history_action_finished(self, description):
        self.record_history_action(description)

    def on_viewer_status_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            status = self.format_viewer_status(text)
            self.status_frame_label.setText(status["frame"])
            self.status_ik_label.setText(status["ik"])
            self.status_move_label.setText(status["move"])
            self.apply_status_event(status["event"])

    def format_viewer_status(self, text):
        event = status_event_from_text(text)
        parts = [part.strip() for part in str(text).split(";") if part.strip()]
        if not self.is_verbose_ik_status(parts):
            return {
                "event": event,
                "state": event.title,
                "frame": "-",
                "ik": "-",
                "move": "-",
                "detail": event.details,
            }

        frame = self.status_field(parts, "frame") or "selected frame"
        frame = frame.replace("_", " ")
        accepted = self.status_field(parts, "accepted")
        ik_error = self.status_field(parts, "IK error")
        lower_text = str(text).lower()
        state = "Preview"

        if "collision warning" in lower_text:
            state = "Warning: collision"
        elif "collision blocked" in lower_text:
            state = "Blocked: collision"
        elif "ik blocked" in lower_text:
            state = "Blocked: IK"

        return {
            "event": event,
            "state": state,
            "frame": frame,
            "ik": f"{ik_error} m" if ik_error else "-",
            "move": accepted or "-",
            "detail": event.details,
        }

    def apply_status_event(self, event):
        """Display one compact event and replace the previous diagnostics."""
        if not isinstance(event, StatusEvent):
            event = status_event_from_text(event)

        if event.signature == self._status_summary_signature:
            self._status_repeat_count += 1
        else:
            self._status_summary_signature = event.signature
            self._status_repeat_count = 1
            self.status_icon_label.setText(event.icon)
            self.status_icon_label.setAccessibleName(
                f"{event.severity.capitalize()} status"
            )
            self.viewer_status_label.setText(event.title)
            self.status_message_label.setText(event.message)
            self.status_message_label.setVisible(bool(event.message))

            for widget in (self.status_icon_label, self.viewer_status_label):
                widget.setProperty("severity", event.severity)
                widget.style().unpolish(widget)
                widget.style().polish(widget)

        # Details always describe only the latest operation. This deliberately
        # replaces rapid live updates instead of growing an event transcript.
        self.status_text.setPlainText(event.details)

    def show_status_message(self, text):
        self.apply_status_event(status_event_from_text(text))

    def is_verbose_ik_status(self, parts):
        fields = {"accepted", "IK error", "frame"}
        found = {
            part.split("=", 1)[0].strip()
            for part in parts
            if "=" in part
        }
        return fields.issubset(found)

    def status_field(self, parts, name):
        prefix = f"{name}="
        for part in parts:
            if part.startswith(prefix):
                return part[len(prefix):].strip()
        return None

    def format_status_detail(self, text):
        return status_event_from_text(text).details

    def on_viewer_time_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_time_label.setText(text.replace("3D state time:", "Time:"))

    def on_viewer_root_pose_changed(self, viewer, text):
        if viewer is self.viewer_3d:
            self.viewer_root_pose_label.setText(text)

    def sync_viewer_status_panel(self):
        self.on_viewer_status_changed(
            self.viewer_3d, self.viewer_3d.status_label.text()
        )
        self.on_viewer_time_changed(
            self.viewer_3d, self.viewer_3d.timeline_state_label.text()
        )
        self.on_viewer_root_pose_changed(
            self.viewer_3d, self.viewer_3d.root_pose_label.text()
        )

    def on_model_changed(self, model_key):
        """Swap model-owned widgets while retaining the surrounding app."""
        if model_key == self.model_key:
            return
        if (
            self.background_jobs.is_busy()
            or self.file_selection_stage.is_active()
        ):
            self.show_status_message(
                "Wait for the current file operation to finish before "
                "changing robot models."
            )
            index = self.controls.model_box.findData(self.model_key)
            self.controls.model_box.blockSignals(True)
            self.controls.model_box.setCurrentIndex(index)
            self.controls.model_box.blockSignals(False)
            self.sync_robot_menu()
            return
        model_info = self.model_registry.get(model_key)
        if model_info is None:
            self.show_status_message(f"Unknown robot model: {model_key}")
            return
        cached = self.model_sessions.get(model_key)
        if cached is not None:
            self.activate_model_session(model_key, cached)
            return
        if model_key in self.model_loaders:
            return
        self.robot_menu.setEnabled(False)
        self.begin_render_progress(
            f"Loading {model_info.display_name}",
            "Loading robot model data...",
        )
        self.controls.model_box.setEnabled(False)
        self.statusBar().showMessage(f"Loading {model_info.display_name}…")
        loader = ModelLoadThread(model_key, model_info, self)
        loader.loaded.connect(self.on_model_loaded)
        loader.failed.connect(self.on_model_load_failed)
        loader.finished.connect(loader.deleteLater)
        self.model_loaders[model_key] = loader
        loader.start()

    def on_model_loaded(self, model_key, adapter):
        self.model_loaders.pop(model_key, None)
        if self._shutting_down:
            return
        backend = BackendInterface(mj_model=adapter.mj_model, adapter=adapter)
        reference = MujocoReferenceFrames(adapter=adapter)
        viewer_3d = RobotViewer3D(
            adapter,
            adapter.load_warning,
            background_jobs=self.background_jobs,
            file_selection_stage=self.file_selection_stage,
        )
        self.connect_model_viewer_signals(viewer_3d)
        self.viewer_3d_stack.addWidget(viewer_3d)
        session = model_sessions.RobotModelSession(
            adapter,
            backend,
            reference,
            viewer_3d,
            document=ProjectDocument(model_key=model_key),
            model_key=model_key,
        )
        self.model_sessions[model_key] = session
        self.finish_model_loading_ui()
        self.activate_model_session(model_key, session)

    def on_model_load_failed(self, model_key, error):
        self.model_loaders.pop(model_key, None)
        if self._shutting_down:
            return
        if (
            self._pending_project_restore is not None
            and self.project_restore_model_key(
                self._pending_project_restore,
                self._pending_project_restore_autosave,
            ) == model_key
        ):
            self._pending_project_restore = None
            self._pending_project_restore_autosave = False
            self._pending_project_restore_source = "direct"
        self.finish_model_loading_ui()
        self.finish_render_progress()
        self.show_status_message(f"Could not load {model_key}: {error}")
        index = self.controls.model_box.findData(self.model_key)
        self.controls.model_box.blockSignals(True)
        self.controls.model_box.setCurrentIndex(index)
        self.controls.model_box.blockSignals(False)

    def on_open_model_file(self):
        if self.background_jobs.is_busy():
            self.show_status_message(
                "Wait for the current import or export to finish."
            )
            return
        if self.file_selection_stage.is_active():
            self.show_status_message("A file selector is already open.")
            return
        self.model_file_selection_stage.select_file(
            mode="open",
            title="Open robot model",
            directory=Path.home(),
            name_filter="Robot model files (*.urdf *.xml)",
            selected=self.import_model_file,
            failed=lambda message: self.show_status_message(
                f"Could not open robot model selector: {message}"
            ),
        )

    def on_setup_import_requested(self, action):
        if action == "model":
            self.on_open_model_file()
        elif action == "qpos":
            self.viewer_3d.choose_qpos_csv()
        elif action == "trajectory":
            self.viewer_3d.choose_trajectory_csv(prompt_import_dt=True)

    def on_setup_export_requested(self, action):
        if action == "qpos":
            self.viewer_3d.choose_qpos_save_path()
        elif action == "trajectory":
            self.viewer_3d.choose_trajectory_save_path()

    def on_choose_mesh_folder(self):
        if self.background_jobs.is_busy():
            self.show_status_message(
                "Wait for the current import or export to finish."
            )
            return
        if self.file_selection_stage.is_active():
            self.show_status_message("A file selector is already open.")
            return
        self.model_file_selection_stage.select_file(
            mode="directory",
            title="Choose Mesh Folder (.stl)",
            directory=Path.home(),
            name_filter="Folders (*)",
            selected=self._set_import_mesh_folder,
            failed=lambda message: self.show_status_message(
                f"Could not open mesh-folder selector: {message}"
            ),
        )

    def _set_import_mesh_folder(self, path):
        self.import_mesh_folder = Path(path).expanduser().resolve()
        self.show_status_message(f"Mesh folder: {self.import_mesh_folder}")

    def import_model_file(self, path):
        mesh_roots = (
            (self.import_mesh_folder,) if self.import_mesh_folder else ()
        )
        self._queue_model_import(path, mesh_roots, allow_mesh_prompt=True)

    def _queue_model_import(
        self,
        path,
        mesh_roots,
        *,
        allow_mesh_prompt,
    ):
        if self.background_jobs.is_busy():
            self.show_status_message(
                "Wait for the current import or export to finish."
            )
            return

        source_path = Path(path).expanduser().resolve()
        library_root = Path(self.model_library_root).expanduser().resolve()
        resolved_mesh_roots = tuple(
            Path(root).expanduser().resolve() for root in mesh_roots
        )
        self.begin_render_progress(
            "Importing robot model",
            f"Copying and preparing {source_path.name}...",
        )

        submitted = self.background_jobs.submit(
            "import robot model",
            lambda: import_robot_model(
                source_path,
                library_root,
                mesh_roots=resolved_mesh_roots,
            ),
            self._on_model_import_completed,
            lambda error: self._on_model_import_failed(
                source_path,
                error,
                allow_mesh_prompt=allow_mesh_prompt,
            ),
        )
        if not submitted:
            self.finish_render_progress()
            self.show_status_message("Robot model import was not started.")

    def _on_model_import_completed(self, info):
        self.finish_render_progress()
        info = self.register_model_info(info)
        self.show_status_message(
            f"Imported {info.display_name} to {info.model_path}"
        )
        self.controls.add_model(info.key, info.display_name, select=True)
        self.refresh_robot_menu()

    def _on_model_import_failed(
        self,
        source_path,
        error,
        *,
        allow_mesh_prompt,
    ):
        self.finish_render_progress()
        if isinstance(error, RuntimeError) and allow_mesh_prompt:
            self.show_status_message(
                "The model needs an external mesh folder. Choose it to retry."
            )
            self.model_file_selection_stage.select_file(
                mode="directory",
                title="Choose Mesh Folder (.stl)",
                directory=Path.home(),
                name_filter="Folders (*)",
                selected=lambda folder: self._retry_model_import_with_meshes(
                    source_path, folder
                ),
                failed=lambda message: self._show_model_import_error(
                    f"Could not open mesh-folder selector: {message}"
                ),
                cancelled=lambda: self.show_status_message(
                    "Robot model import cancelled."
                ),
            )
            return
        self._show_model_import_error(f"Could not import model: {error}")

    def _retry_model_import_with_meshes(self, source_path, folder):
        self._set_import_mesh_folder(folder)
        self._queue_model_import(
            source_path,
            (self.import_mesh_folder,),
            allow_mesh_prompt=False,
        )

    def _show_model_import_error(self, message):
        self.show_status_message(message)
        QMessageBox.warning(self, "Import model failed", message)

    def finish_model_loading_ui(self):
        self.controls.model_box.setEnabled(True)
        self.robot_menu.setEnabled(True)
        self.sync_robot_menu()
        self.statusBar().clearMessage()

    def activate_model_session(self, model_key, session):
        previous_model_key = self.model_key
        restoring_project = (
            self._pending_project_restore is not None
            and self.project_restore_model_key(
                self._pending_project_restore,
                self._pending_project_restore_autosave,
            ) == model_key
        )
        model_sessions.remember_current_session(
            self.model_sessions, self.model_key, self.trajectory, self.active_index
        )
        for name, value in model_sessions.activated_session_state(
            model_key, session
        ).items():
            setattr(self, name, value)
        self.editor_controller.activate_document(self.document)
        self.viewer_3d_stack.setCurrentWidget(self.viewer_3d)
        self.viewer_3d_mujoco.set_model_adapter(session.adapter)
        self.viewer_3d_mujoco.clear_trajectory()
        if self.viewer_3d.robot_trajectory:
            self.viewer_3d_mujoco.set_trajectory_metadata(
                mujoco_playback_cache_path(),
                self.viewer_3d.robot_trajectory_times,
            )
        self.viewer_3d.set_smoothing_widget(self.controls.corner_smoothing_slider)
        self.set_editor_timeline_duration(self.viewer_3d.timeline_duration)
        self.model_source_label.setText(self.model_source_text(session.adapter))
        self.begin_render_progress(
            f"Rendering {session.adapter.model_name}",
            "Preparing the 3D model geometry...",
            viewer=self.viewer_3d,
        )
        self.controls.set_frame_names(session.adapter.trajectory_frames)
        self.update_project_chrome()
        self.update_editor_context()
        self.refresh_display()
        self.request_active_model_render()
        self.history.clear()
        self._refresh_history_baseline()
        self.sync_robot_menu()
        self.sync_display_actions()
        self.sync_workflow_toolbar()
        if previous_model_key != model_key and not restoring_project:
            self.mark_project_dirty("Change robot model")
        self.restore_pending_project_if_ready()

    def on_trajectory_csv_loaded(self, csv_path):
        self.viewer_3d_mujoco.set_trajectory_metadata(
            csv_path,
            self.viewer_3d.robot_trajectory_times,
        )
        if self.viewer_3d.consume_trajectory_import_dt_prompt_request():
            self.prompt_trajectory_import_dt()
        import_dt = self.viewer_3d.trajectory_import_dt.value()
        if self.viewer_3d.consume_background_trajectory_postprocess_request():
            qposes = tuple(
                qpos.copy() for qpos in self.viewer_3d.robot_trajectory
            )
            times = tuple(self.viewer_3d.robot_trajectory_times)
            phase = self.controls.phase_box.currentText()
            frame_names = tuple(self.editable_logical_frame_names())
            frame_bindings = dict(self.viewer_3d.frame_bindings)
            robot_model = self.viewer_3d.robot_model
            self.show_status_message(
                f"Loaded trajectory CSV: {csv_path}\n"
                "Computing editable target-frame keyframes..."
            )
            submitted = self.background_jobs.submit(
                "compute imported trajectory target frames",
                lambda: timeslice_service.build_loaded_trajectory_target_frames(
                    robot_model,
                    times,
                    qposes,
                    interval=import_dt,
                    phase=phase,
                    frame_names=frame_names,
                    frame_bindings=frame_bindings,
                ),
                lambda result: self._finish_trajectory_csv_import(
                    csv_path,
                    result,
                    import_dt,
                ),
                lambda error: self.show_status_message(
                    f"Loaded trajectory CSV, but could not compute editable "
                    f"target frames: {error}"
                ),
            )
            if submitted:
                return

        count = self.import_loaded_robot_trajectory_as_keyframes()
        self._show_trajectory_csv_import_completed(
            csv_path,
            count,
            import_dt,
        )

    def _finish_trajectory_csv_import(self, csv_path, result, import_dt):
        count = self._apply_loaded_trajectory_targets(result)
        self._show_trajectory_csv_import_completed(
            csv_path,
            count,
            import_dt,
        )

    def _show_trajectory_csv_import_completed(
        self,
        csv_path,
        count,
        import_dt,
    ):
        self.show_status_message(
            f"Loaded trajectory CSV: {csv_path}\n"
            f"Imported {count} editable target-frame keyframes from FK "
            f"at {import_dt:.2f} s intervals."
        )
        self.record_history_action("Load trajectory")

    def prompt_trajectory_import_dt(self):
        current = self.viewer_3d.trajectory_import_dt.value()
        value, accepted = QInputDialog.getDouble(
            self,
            "Import trajectory time step",
            "Editable keyframe interval [s]",
            current,
            0.01,
            10.0,
            2,
        )
        if accepted:
            self.viewer_3d.trajectory_import_dt.setValue(value)
            self.mark_project_dirty("Import time step")

    # ============================================================
    # GUI interaction callbacks
    # ============================================================

    def on_pose_changed(self, x, y, z, roll, pitch, yaw):
        """
        Called when sliders change.

        If a keyframe is selected, we only preview the target frame.
        The committed keyframe is overwritten only when the preview is accepted.
        """

        frame_name = self.controls.frame_box.currentText()
        self.viewer_3d.preview_target_pose(
            frame_name,
            (x, y, z),
            rpy_to_quat(roll, pitch, yaw),
        )
        self.refresh_display()

    def on_time_changed(self, time):
        """Load or create the editable qpos keyframe for this GUI time."""
        self.viewer_3d.set_current_time(time)
        self.editor_controller.set_current_time(time)
        frame_name = self.controls.frame_box.currentText()
        target = self.trajectory.targets_at_time(time).get(frame_name)
        if target is not None:
            self.controls.set_position_values(
                x=target.x,
                y=target.y,
                z=target.z,
                roll=target.roll,
                pitch=target.pitch,
                yaw=target.yaw,
                emit_pose_changed=False,
            )
        self.refresh_display()
        self._refresh_history_baseline()

    def on_viewer_timeslice_time_changed(self, time):
        """Keep the sidebar time editor in sync with the viewer-bottom scrubber."""
        self.controls.time_slider.set_value(time)
        self.on_time_changed(time)

    def on_viewer_timeslice_preview_time_changed(self, time):
        """Mirror live playback time without creating an editable qpos state."""
        self.controls.time_slider.set_value(time)

    def on_viewer_timeline_duration_changed(self, duration):
        self.set_sidebar_timeline_duration(duration)
        self.mark_project_dirty("Timeline duration")

    def set_sidebar_timeline_duration(self, duration):
        self.controls.time_slider.set_range(0, int(round(float(duration) * 100.0)))

    def set_editor_timeline_duration(self, duration):
        self.viewer_3d.set_timeline_duration(duration, emit_signal=False)
        self.set_sidebar_timeline_duration(duration)
        self.document.set_timeline_duration(duration)

    def on_accept_timeslice_requested(self):
        if (
            self.viewer_3d.preview_active
            and not self.viewer_3d.accept_preview(emit_pose_finished=False)
        ):
            return

        count = self.define_timeslice_from_committed_pose()
        time = self.viewer_3d.get_current_time()
        if count <= 0:
            message = f"No logical target frames were available at t={time:.2f} s."
            self.viewer_3d.status_label.setText(message)
            return

        self.refresh_display()
        next_time = self.viewer_3d.next_timeslice_time(time)
        if abs(next_time - time) > 1e-9:
            self.on_viewer_timeslice_time_changed(next_time)
            advance_note = f" advanced to t={next_time:.2f} s."
        else:
            advance_note = " already at the timeline end."
        message = (
            f"Committed keyframe at t={time:.2f} s; captured {count} logical targets "
            f"from the committed solved pose;{advance_note}"
        )
        self.viewer_3d.status_label.setText(message)
        self.record_history_action("Commit keyframe")

    def on_delete_timeslice_requested(self):
        time = self.viewer_3d.get_current_time()
        deleted_targets = self.delete_timeslice_at_time(time)
        deleted_qpos = False
        if self.viewer_3d.state_timeline is not None:
            deleted_qpos = self.viewer_3d.state_timeline.delete_state(time)

        if deleted_targets <= 0 and not deleted_qpos:
            message = f"No keyframe found at t={time:.2f} s."
            self.viewer_3d.status_label.setText(message)
            return

        self.active_index = -1
        next_time = time
        if deleted_qpos and self.viewer_3d.state_timeline is not None:
            remaining_times = self.viewer_3d.state_timeline.times()
            if remaining_times:
                next_time = min(
                    remaining_times, key=lambda candidate: abs(candidate - time)
                )
        self.controls.time_slider.set_value(next_time)
        self.viewer_3d.set_current_time(next_time)
        self.refresh_display()
        parts = []
        if deleted_targets > 0:
            parts.append(f"{deleted_targets} target frames")
        if deleted_qpos:
            parts.append("robot qpos")
        detail = " and ".join(parts)
        message = f"Deleted keyframe at t={time:.2f} s ({detail})."
        self.viewer_3d.status_label.setText(message)
        self.record_history_action("Delete keyframe")

    def define_timeslice_from_committed_pose(self):
        """Snapshot every editable logical target from the committed MuJoCo pose."""
        frames = timeslice_service.capture_timeslice_from_committed_pose(
            self.viewer_3d.committed_state,
            time=self.viewer_3d.get_current_time(),
            phase=self.controls.phase_box.currentText(),
            frame_names=self.editable_logical_frame_names(),
            frame_bindings=self.viewer_3d.frame_bindings,
        )
        selected_frame_name = self.controls.frame_box.currentText()
        result = self.editor_controller.execute(
            UpsertKeyframes(
                frames,
                selected_frame_name=selected_frame_name,
                operation="commit_timeslice",
            )
        )
        selected_frame = next(
            (
                frame
                for frame in frames
                if frame.frame_name == selected_frame_name
            ),
            None,
        )
        if selected_frame is not None:
            self.controls.set_position_values(
                x=selected_frame.x,
                y=selected_frame.y,
                z=selected_frame.z,
                roll=selected_frame.roll,
                pitch=selected_frame.pitch,
                yaw=selected_frame.yaw,
                emit_pose_changed=False,
            )
        return result.affected_count

    def editable_logical_frame_names(self):
        return timeslice_service.editable_logical_frame_names(
            self.robot_model_3d,
            self.controls.frame_names,
            self.viewer_3d.frame_bindings,
        )

    def import_loaded_robot_trajectory_as_keyframes(self):
        """Convert loaded qpos playback rows into editable logical target frames."""
        qposes = list(getattr(self.viewer_3d, "robot_trajectory", []))
        times = list(getattr(self.viewer_3d, "robot_trajectory_times", []))
        if not qposes:
            return 0

        if self.viewer_3d.robot_model is None:
            return 0

        result = timeslice_service.build_loaded_trajectory_target_frames(
            self.viewer_3d.robot_model,
            times,
            qposes,
            interval=self.viewer_3d.trajectory_import_dt.value(),
            phase=self.controls.phase_box.currentText(),
            frame_names=self.editable_logical_frame_names(),
            frame_bindings=self.viewer_3d.frame_bindings,
        )
        return self._apply_loaded_trajectory_targets(result)

    def _apply_loaded_trajectory_targets(self, result):
        selected_frame_name = self.controls.frame_box.currentText()
        current_time = self.viewer_3d.get_current_time()
        self.editor_controller.execute(
            ReplaceTrajectoryFrames(
                result.frames,
                selected_frame_name=selected_frame_name,
                selected_time=current_time,
            )
        )
        selected_frame = next(
            (
                frame
                for frame in result.frames
                if frame.frame_name == selected_frame_name
                and abs(frame.time - current_time) <= 1e-6
            ),
            None,
        )

        qposes = self.viewer_3d.robot_trajectory
        times = self.viewer_3d.robot_trajectory_times
        if qposes and times:
            self.viewer_3d.committed_state.set_qpos(qposes[0])
            self.viewer_3d.preview_state.set_qpos(qposes[0])
            self.viewer_3d.preview_active = False
            self.viewer_3d.canvas.set_preview_visible(False)
            self.controls.time_slider.set_value(float(times[0]))

        if selected_frame is not None:
            self.controls.set_position_values(
                x=selected_frame.x,
                y=selected_frame.y,
                z=selected_frame.z,
                roll=selected_frame.roll,
                pitch=selected_frame.pitch,
                yaw=selected_frame.yaw,
                emit_pose_changed=False,
            )
        self.viewer_3d.set_defined_timeslices(result.imported_times)
        self.refresh_display()
        return len(result.frames)

    def selected_loaded_trajectory_import_samples(self, times, qposes):
        return timeslice_service.selected_loaded_trajectory_import_samples(
            times, qposes, self.viewer_3d.trajectory_import_dt.value()
        )

    def delete_timeslice_at_time(self, time, tolerance=1e-6):
        result = self.editor_controller.execute(
            DeleteTimeslice(time, tolerance)
        )
        return result.affected_count

    def on_target_dragged(self, x, z):
        """
        Called when user drags the red reference frame in the viewer.

        This updates the sliders, so the GUI stays consistent.
        """

        self.controls.set_position_from_viewer(x, z)

    def on_target_pose_dragged(
        self, x, y, z, roll=None, pitch=None, yaw=None
    ):
        """Sync controls without repainting every viewer on every mouse event."""
        # The live canvas already updated its transforms; refresh the remaining
        # controls and status once on mouse release.
        self.controls.set_position_values(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            emit_pose_changed=False,
        )

    def on_target_pose_drag_finished(
        self, x, y, z, roll=None, pitch=None, yaw=None
    ):
        self.controls.set_position_values(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            emit_pose_changed=False,
        )
        # A completed 3D edit is an intentional keyframe edit. Upsert the
        # selected logical target at the active time while RobotViewer3D stores
        # the corresponding accepted qpos in its time-keyed state timeline.
        frame = self.controls.current_frame()
        self.editor_controller.execute(UpsertKeyframe(frame))
        self.refresh_display()
        description = (
            getattr(self.viewer_3d, "_pending_history_action_description", None)
            or "Accept pose"
        )
        self.viewer_3d._pending_history_action_description = None
        self.record_history_action(description)

    def on_3d_target_frame_changed(self, frame_name):
        """Map common 3D body/site selections back to the 2D frame concept."""
        self.controls.frame_box.blockSignals(True)
        self.controls.frame_box.setCurrentText(frame_name)
        self.controls.frame_box.blockSignals(False)
        binding = self.robot_model_3d.resolve_logical_frame(frame_name)
        if binding is not None:
            kind, name = binding
            state = (
                self.viewer_3d.preview_state
                if self.viewer_3d.preview_active
                else self.viewer_3d.committed_state
            )
            position, quaternion = state.get_body_pose(name, kind)
            roll, pitch, yaw = quat_to_rpy(quaternion)
            self.controls.set_position_values(
                x=float(position[0]), y=float(position[1]), z=float(position[2]),
                roll=roll, pitch=pitch, yaw=yaw,
                emit_pose_changed=False,
            )
        self.refresh_display()

    def on_preview_cancelled(self):
        kind, name = self.viewer_3d._selected_target()
        if not name:
            return
        position, quaternion = self.viewer_3d.committed_state.get_body_pose(
            name, kind
        )
        roll, pitch, yaw = quat_to_rpy(quaternion)
        self.controls.set_position_values(
            x=float(position[0]), y=float(position[1]), z=float(position[2]),
            roll=roll, pitch=pitch, yaw=yaw,
            emit_pose_changed=False,
        )
        self.refresh_display()

    def on_trajectory_lines_changed(self, checked):
        self.on_trajectory_display_changed(checked)

    def on_trajectory_display_changed(self, checked):
        self.refresh_display()
        self.mark_project_dirty("Display settings")

    def on_add_keyframe(self):
        """
        Add the currently edited target frame to the trajectory array.
        """

        frame = self.controls.current_frame()
        self.editor_controller.execute(AddKeyframe(frame))

        self.refresh_display()
        self.record_history_action("Add keyframe")

    def on_update_keyframe(self):
        """
        Replace selected keyframe with current editor values.
        """

        row = self.controls.selected_row()

        if row < 0:
            self.show_status_message("No keyframe selected to update.")
            return

        frame = self.controls.current_frame()
        self.editor_controller.execute(UpdateKeyframe(row, frame))
        self.refresh_display()
        self.record_history_action("Update keyframe")

    def on_delete_keyframe(self):
        """
        Delete selected keyframe.
        """

        row = self.controls.selected_row()

        if row < 0:
            self.show_status_message("No keyframe selected to delete.")
            return

        self.editor_controller.execute(DeleteKeyframe(row))

        self.refresh_display()
        self.record_history_action("Delete keyframe")

    def on_clear_trajectory(self):
        keyframe_count = len(self.trajectory.frames)
        if keyframe_count == 0:
            self.show_status_message("Trajectory is already empty.")
            return

        response = QMessageBox.question(
            self,
            "Clear trajectory",
            f"Delete all {keyframe_count} trajectory keyframes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            self.show_status_message("Clear trajectory cancelled.")
            return

        self.editor_controller.execute(ClearTrajectory())
        self.viewer_3d.clear_robot_trajectory()
        self.viewer_3d_mujoco.clear_trajectory()
        self.backend_interface.clear_last_solution()
        self.viewer_3d.clear_editable_timeline(
            keep_current_pose=True,
            reset_time=0.0,
        )
        self.controls.time_slider.set_value(0.0)
        self.viewer_3d.set_defined_timeslices([])
        self.refresh_display()
        message = (
            f"Cleared {keyframe_count} trajectory keyframes; "
            "current robot pose was left unchanged."
        )
        self.viewer_3d.status_label.setText(message)
        self.record_history_action("Clear trajectory")

    def on_keyframe_selected(self, row):
        """
        Load selected keyframe into the editor.
        """

        if row < 0 or row >= len(self.trajectory.frames):
            return

        self.active_index = row
        frame = self.trajectory.frames[row]

        self.controls.set_from_frame(frame)
        self.refresh_display()
        self._refresh_history_baseline()

    def on_frame_name_changed(self, frame_name):
        """
        Called when user changes the selected target robot frame.

        Example:
            pelvis -> left_foot

        The target controls should jump to the current real MuJoCo body/site
        position using the actual robot model.
        """

        if self.set_current_frame_to_model_reference(
            frame_name,
            emit_pose_changed=False,
        ):
            self.refresh_display()
        else:
            self.show_status_message(
                f"No model reference pose is available for {frame_name}."
            )
        self._refresh_history_baseline()

    def set_current_frame_to_model_reference(
        self,
        frame_name,
        emit_pose_changed=True,
    ):
        pose = self.model_reference.pose_for_frame(frame_name)
        if pose is None and self.viewer_3d.committed_state is not None:
            binding = self.viewer_3d.frame_bindings.get(frame_name)
            if binding is not None:
                kind, object_name = binding
                try:
                    pose = self.viewer_3d.committed_state.get_body_pose(
                        object_name,
                        kind,
                    )
                except KeyError:
                    pose = None

        if pose is None:
            return False

        position, quaternion = pose
        x, y, z = position
        roll, pitch, yaw = quat_to_rpy(quaternion)
        self.controls.set_position_values(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            emit_pose_changed=emit_pose_changed,
        )
        return True

    def on_generate_trajectory(self):
        if len(self.trajectory.frames) == 0:
            self.show_status_message("Trajectory is empty. Add keyframes first.")
            return

        result = trajectory_generation.generate_trajectory_status(
            self.trajectory,
            self.backend_interface,
            smoothing=self.controls.corner_smoothing(),
        )
        self.viewer_3d.load_backend_states(result.result_states)
        self.viewer_3d_mujoco.set_trajectory_csv(result.csv_path)
        self.show_status_message(result.status_text)
        self.record_history_action("Generate trajectory")

    def regenerate_mujoco_playback_cache(self):
        """Recreate the disposable viewer CSV from the current solved states."""
        playback_path = mujoco_playback_cache_path()
        if self.viewer_3d.robot_trajectory:
            export = self.viewer_3d._trajectory_export_snapshot()
            return write_trajectory_csv(playback_path, export)
        if self.backend_interface.has_last_solution():
            self.backend_interface.export_last_solution_csv(playback_path)
            return playback_path.resolve()
        raise RuntimeError("Generate or import a trajectory first.")

    # ============================================================
    # Display update
    # ============================================================

    def refresh_display(self):
        """
        Refresh viewer, table, and status text.
        """

        scene = SceneSnapshot(
            trajectory=self.trajectory,
            active_frame=self.controls.current_frame(),
            show_trajectory_lines=self.controls.show_trajectory_lines(),
            trajectory_smoothing=self.controls.corner_smoothing(),
            show_keyframes=self.controls.show_keyframes(),
            defined_timeslices=tuple(
                sorted({frame.time for frame in self.trajectory.frames})
            ),
        )
        self.visualization_manager.update(
            VisualizationUpdate(
                scene=scene,
                revision=self.document.revision,
            )
        )
        self.controls.refresh_table(self.trajectory)

    def _render_robot_scene(self, update):
        """Display adapter retaining the existing RobotViewer3D API."""
        scene = update.scene
        self.viewer_3d.update_scene(
            trajectory=scene.trajectory,
            active_frame=scene.active_frame,
            show_trajectory_lines=scene.show_trajectory_lines,
            trajectory_smoothing=scene.trajectory_smoothing,
            show_keyframes=scene.show_keyframes,
        )

    def _render_timeline_markers(self, update):
        """Display adapter owning the committed-timeslice markers."""
        scene = update.scene
        self.viewer_3d.set_defined_timeslices(
            scene.defined_timeslices
        )

    def _refresh_status_panel(self, _update):
        """Panel adapter for backend and live viewer diagnostics."""
        self.backend_label.setText(
            f"Backend: {self.backend_interface.backend_name()}"
        )
        self.sync_viewer_status_panel()
