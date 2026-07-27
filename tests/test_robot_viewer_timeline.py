import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QCloseEvent, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolBar,
    QToolButton,
    QWidget,
)

from core.ik import Collision
from application.backend_interface import PythonRobotConfiguration
from application.project_manager import (
    GhostGUIProject,
    ghostgui_projects_dir,
    load_recent_projects,
)
from application.paths import mujoco_playback_cache_path
from gui.main_window import (
    INITIAL_RENDER_PROGRESS_DELAY_MS,
    LEFT_SIDEBAR_MIN_WIDTH,
    RIGHT_SIDEBAR_MIN_WIDTH,
    SIDEBAR_MAX_WIDTH,
    RobotGuiMainWindow,
)
from gui.viewers.transform_gizmo import GizmoInteractionState
from core.trajectory import quat_to_rpy, rpy_to_quat
from scripts.view_g1_mujoco import RAW_QPOS_KEY, load_trajectory_csv


class RobotViewerTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config_dir = tempfile.TemporaryDirectory()
        self.config_patch = patch.dict(
            os.environ,
            {
                "GHOSTGUI_CONFIG_DIR": str(
                    Path(self.config_dir.name) / "config"
                ),
                "GHOSTGUI_PROJECTS_DIR": str(
                    Path(self.config_dir.name) / "projects"
                ),
                "GHOSTGUI_CACHE_DIR": str(
                    Path(self.config_dir.name) / "cache"
                ),
            },
        )
        self.config_patch.start()
        self.window = RobotGuiMainWindow()
        self.viewer = self.window.viewer_3d

    def tearDown(self):
        self.window.close()
        self.config_patch.stop()
        self.config_dir.cleanup()

    def test_time_control_loads_distinct_editable_state(self):
        at_zero = self.viewer.get_current_keyframe()
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.assertEqual(self.viewer.get_current_time(), 0.2)
        self.assertIsNotNone(self.viewer.get_current_keyframe())

        name = self.viewer.robot_state.get_joint_names()[-1]
        old_value = self.viewer.robot_state.get_joint_value(name)
        self.viewer.robot_state.set_joint_value(name, old_value + 0.05)
        self.viewer.update_current_keyframe_from_robot_state()

        self.viewer.set_current_time(0.0)
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), at_zero)
        self.viewer.set_current_time(0.2)
        self.assertAlmostEqual(
            self.viewer.robot_state.get_joint_value(name), old_value + 0.05
        )

    def test_reset_changes_only_active_time(self):
        at_zero = self.viewer.get_current_keyframe()
        self.viewer.set_current_time(0.2)
        changed = self.viewer.robot_state.get_qpos()
        changed[-1] += 0.1
        self.viewer.robot_state.set_qpos(changed)
        self.viewer.update_current_keyframe_from_robot_state()
        self.viewer.reset_robot_pose()
        np.testing.assert_allclose(
            self.viewer.get_current_keyframe(), self.viewer.robot_model.home_qpos
        )
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.0), at_zero
        )

    def test_reset_is_one_shot_and_does_not_replace_playback(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        first[-1] += 0.05
        second[-1] += 0.10
        self.viewer.set_robot_trajectory([first, second])
        playback_before = [qpos.copy() for qpos in self.viewer.robot_trajectory]

        self.viewer.set_current_time(0.2)
        self.viewer.play_timer.start()
        self.viewer.canvas.gizmo.state = GizmoInteractionState.DRAG_TRANSLATE_FREE
        self.viewer.reset_robot_pose()

        self.assertFalse(self.viewer.play_timer.isActive())
        self.assertEqual(
            self.viewer.canvas.gizmo.state, GizmoInteractionState.NONE
        )
        for actual, expected in zip(
            self.viewer.robot_trajectory, playback_before
        ):
            np.testing.assert_allclose(actual, expected)

        self.viewer._advance_playback(elapsed=0.8)
        np.testing.assert_allclose(
            self.viewer.playback_state.get_qpos(), second
        )
        np.testing.assert_allclose(
            self.viewer.robot_state.get_qpos(),
            self.viewer.robot_model.home_qpos,
        )
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.2),
            self.viewer.robot_model.home_qpos,
        )

    def test_undo_redo_shortcuts_restore_keyframe_actions(self):
        self.window.show()
        self.window.setFocus()
        QApplication.processEvents()

        self.window.on_add_keyframe()
        self.assertEqual(len(self.window.trajectory.frames), 1)

        QTest.keyClick(
            self.window,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.assertEqual(len(self.window.trajectory.frames), 0)
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Undid Add keyframe.",
        )

        QTest.keyClick(
            self.window,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier,
        )
        self.assertEqual(len(self.window.trajectory.frames), 1)
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Redid Add keyframe.",
        )

    def test_undo_redo_restores_viewer_owned_reset_pose(self):
        changed = self.viewer.robot_model.home_qpos.copy()
        changed[-1] += 0.05
        self.viewer.set_robot_state_for_current_time(changed)
        self.viewer.update_current_keyframe_from_robot_state()
        self.window._refresh_history_baseline()

        self.viewer.reset_robot_pose()

        np.testing.assert_allclose(
            self.viewer.committed_state.get_qpos(),
            self.viewer.robot_model.home_qpos,
        )

        self.window.undo_last_action()
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), changed)
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Undid Reset 3D pose.",
        )
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(self.viewer.get_current_time()),
            changed,
        )

        self.window.redo_last_action()
        np.testing.assert_allclose(
            self.viewer.committed_state.get_qpos(),
            self.viewer.robot_model.home_qpos,
        )
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Redid Reset 3D pose.",
        )

    def test_pelvis_drag_updates_preview_only_until_accept(self):
        self.viewer.select_target("body", "robot/pelvis", emit=False)
        self.viewer._set_target_to_selected_pose()
        start = self.viewer.last_valid_target_position.copy()
        quaternion = self.viewer.last_valid_target_quaternion.copy()
        self.viewer._on_transform_moved(
            start + np.array([0.01, 0.0, 0.0]), quaternion
        )
        root_address = next(
            iter(self.viewer.robot_model.free_joints_by_body.values())
        ).qpos_address
        self.assertAlmostEqual(self.viewer.committed_state.get_qpos()[root_address], start[0])
        self.assertAlmostEqual(
            self.viewer.get_current_keyframe()[root_address], start[0]
        )
        self.assertAlmostEqual(
            self.viewer.preview_state.get_qpos()[root_address], start[0] + 0.01
        )
        self.assertTrue(self.viewer.preview_active)
        self.assertTrue(self.viewer.canvas.preview_visible)
        self.viewer.accept_preview()
        self.assertAlmostEqual(
            self.viewer.committed_state.get_qpos()[root_address], start[0] + 0.01
        )
        self.assertAlmostEqual(
            self.viewer.get_current_keyframe()[root_address], start[0] + 0.01
        )
        self.assertFalse(self.viewer.preview_active)

    def test_target_pose_sliders_update_3d_preview_only_until_accept(self):
        self.viewer.select_target("body", "robot/pelvis", emit=False)
        self.viewer._set_target_to_selected_pose()
        self.window.controls.frame_box.setCurrentText("pelvis")
        start_position = self.viewer.last_valid_target_position.copy()
        start_quaternion = self.viewer.last_valid_target_quaternion.copy()
        root_address = next(
            iter(self.viewer.robot_model.free_joints_by_body.values())
        ).qpos_address
        before = self.viewer.committed_state.get_qpos()

        self.window.controls.set_position_values(
            x=float(start_position[0] + 0.01),
            y=float(start_position[1]),
            z=float(start_position[2]),
            roll=0.02,
            pitch=-0.01,
            yaw=0.03,
        )

        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.get_current_keyframe(), before)
        self.assertGreater(
            self.viewer.preview_state.get_qpos()[root_address],
            before[root_address],
        )
        self.assertFalse(
            np.allclose(
                self.viewer.preview_state.get_qpos()[root_address + 3:root_address + 7],
                start_quaternion,
            )
        )
        self.assertTrue(self.viewer.preview_active)
        self.assertTrue(self.viewer.canvas.preview_visible)

    def test_cancel_discards_preview_without_touching_committed(self):
        before = self.viewer.committed_state.get_qpos()
        name = self.viewer.preview_state.get_joint_names()[-1]
        old_value = self.viewer.preview_state.get_joint_value(name)
        self.viewer._joint_changed(name, old_value + 0.05)
        self.assertTrue(self.viewer.preview_active)
        self.assertFalse(np.allclose(self.viewer.preview_state.get_qpos(), before))
        self.viewer.cancel_preview()
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.preview_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.get_current_keyframe(), before)

    def test_plan_creates_ghost_path_without_committing(self):
        before = self.viewer.committed_state.get_qpos()
        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        preview = self.viewer.preview_state.get_qpos()
        self.viewer.plan_preview()
        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(len(self.viewer.ghost_trajectory), 40)
        self.assertEqual(self.viewer.ghost_source, "preview_path")
        np.testing.assert_allclose(self.viewer.ghost_trajectory[0], before)
        np.testing.assert_allclose(self.viewer.ghost_trajectory[-1], preview)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        np.testing.assert_allclose(self.viewer.get_current_keyframe(), before)
        self.assertTrue(self.viewer.preview_active)
        self.assertTrue(self.viewer.canvas.show_ghosts)
        self.assertFalse(self.viewer.show_ghosts.isChecked())

    def test_accept_preview_clears_preview_path_ghost(self):
        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        self.viewer.plan_preview()
        self.assertEqual(self.viewer.ghost_source, "preview_path")

        self.viewer.accept_preview()

        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIsNone(self.viewer.ghost_source)
        self.assertFalse(self.viewer.canvas.show_ghosts)

    def test_playback_load_clears_preview_path_ghost_when_pose_overlay_is_off(self):
        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        self.viewer.plan_preview()
        self.assertEqual(self.viewer.ghost_source, "preview_path")

        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05
        self.viewer.set_robot_trajectory([first, second])

        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIsNone(self.viewer.ghost_source)
        self.assertFalse(self.viewer.canvas.show_ghosts)

    def test_playback_pose_ghosts_are_explicit(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05

        self.viewer.set_robot_trajectory([first, second])

        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIsNone(self.viewer.ghost_source)
        self.assertFalse(self.viewer.canvas.show_ghosts)

        self.viewer.show_ghosts.setChecked(True)

        self.assertEqual(len(self.viewer.ghost_trajectory), 2)
        self.assertEqual(self.viewer.ghost_source, "playback")
        np.testing.assert_allclose(self.viewer.ghost_trajectory[0], first)
        np.testing.assert_allclose(self.viewer.ghost_trajectory[1], second)
        self.assertTrue(self.viewer.canvas.show_ghosts)

        self.viewer.show_ghosts.setChecked(False)

        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIsNone(self.viewer.ghost_source)
        self.assertFalse(self.viewer.canvas.show_ghosts)

    def test_timeline_keyframes_do_not_publish_ghost_overlays(self):
        joint_name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            joint_name,
            self.viewer.preview_state.get_joint_value(joint_name) + 0.05,
        )
        self.viewer.accept_preview()

        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIsNone(self.viewer.ghost_source)
        self.assertFalse(self.viewer.canvas.show_ghosts)

    def test_plan_preview_rejects_midpoint_collision_without_publishing(self):
        class MidpointCollisionChecker:
            def __init__(self, address, lower, upper):
                self.address = address
                self.lower = lower
                self.upper = upper

            def get_collisions(self, state):
                value = state.get_qpos()[self.address]
                if self.lower <= value <= self.upper:
                    return [
                        Collision(
                            "geom_a", "geom_b", "body_a", "body_b",
                            -0.01, "self",
                        )
                    ]
                return []

        joint = next(
            item for item in self.viewer.robot_model.joints.values()
            if item.limits is not None
        )
        start_value = self.viewer.preview_state.get_joint_value(joint.name)
        lo, hi = joint.limits
        delta = 0.1 if start_value + 0.1 < hi else -0.1
        goal_value = start_value + delta
        lower = min(start_value, goal_value) + abs(delta) * 0.45
        upper = min(start_value, goal_value) + abs(delta) * 0.55

        self.viewer._joint_changed(joint.name, goal_value)
        self.viewer.collision_checker = MidpointCollisionChecker(
            joint.qpos_address, lower, upper
        )
        self.viewer.plan_preview()

        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIn("Cannot preview path", self.viewer.status_label.text())
        self.assertIn("collision at path sample", self.viewer.status_label.text())

    def test_plan_preview_rejects_raw_joint_limit_violation_before_clamp(self):
        joint = next(
            item for item in self.viewer.robot_model.joints.values()
            if item.limits is not None
        )
        _, hi = joint.limits
        self.viewer.begin_preview()
        qpos = self.viewer.preview_state.get_qpos()
        qpos[joint.qpos_address] = hi + 0.5
        self.viewer.preview_state.mj_data.qpos[:] = qpos
        self.viewer.collision_checker = None

        self.viewer.plan_preview()

        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertIn("Cannot preview path", self.viewer.status_label.text())
        self.assertIn("outside limits", self.viewer.status_label.text())

    def test_accept_preview_is_timeline_local(self):
        at_zero = self.viewer.committed_state.get_qpos()
        self.viewer.set_current_time(0.2)
        name = self.viewer.preview_state.get_joint_names()[-1]
        value = self.viewer.preview_state.get_joint_value(name) + 0.05
        self.viewer._joint_changed(name, value)
        self.viewer.accept_preview()
        accepted = self.viewer.get_current_keyframe()
        self.viewer.set_current_time(0.0)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), at_zero)
        self.viewer.set_current_time(0.2)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), accepted)

    def test_joint_slider_edits_preview_not_committed(self):
        before = self.viewer.committed_state.get_qpos()
        name = self.viewer.preview_state.get_joint_names()[0]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.03
        )
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        self.assertFalse(np.allclose(self.viewer.preview_state.get_qpos(), before))

    def test_joint_collision_highlights_preview_link_and_still_blocks_accept(self):
        model = self.viewer.robot_model.mj_model
        geom_id = next(
            candidate
            for candidate in self.viewer.canvas.render_geom_ids(model)
            if int(model.geom_bodyid[candidate]) > 0
        )
        body_id = int(model.geom_bodyid[geom_id])
        collision = Collision(
            "colliding_geom",
            "environment",
            "colliding_body",
            "world",
            -0.01,
            "environment",
            geom1_id=geom_id,
            body1_id=body_id,
            body2_id=0,
        )

        class FixedCollisionChecker:
            def get_collisions(self, _state):
                return [collision]

        self.viewer.collision_checker = FixedCollisionChecker()
        name = self.viewer.preview_state.get_joint_names()[0]
        value = self.viewer.preview_state.get_joint_value(name) + 0.01

        self.viewer._joint_changed(name, value)

        self.assertTrue(self.viewer.preview_active)
        self.assertIn("Collision warning", self.viewer.status_label.text())
        self.assertIn(geom_id, self.viewer.canvas.preview_collision_geom_ids)
        self.assertIn(body_id, self.viewer.canvas.preview_collision_body_ids)
        self.assertTrue(
            self.viewer.canvas._geom_is_preview_collision(model, geom_id)
        )

        self.assertFalse(self.viewer.accept_preview())
        self.assertTrue(self.viewer.preview_active)
        self.assertIn("Cannot commit keyframe", self.viewer.status_label.text())

        class NoCollisionChecker:
            def get_collisions(self, _state):
                return []

        self.viewer.collision_checker = NoCollisionChecker()
        self.viewer._joint_changed(name, value)
        self.assertEqual(self.viewer.canvas.preview_collision_geom_ids, set())
        self.assertEqual(self.viewer.canvas.preview_collision_body_ids, set())

        self.viewer.cancel_preview()
        self.assertEqual(self.viewer.canvas.preview_collision_geom_ids, set())
        self.assertEqual(self.viewer.canvas.preview_collision_body_ids, set())

    def test_timeline_change_discards_unaccepted_preview(self):
        name = self.viewer.preview_state.get_joint_names()[0]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.03
        )
        self.assertTrue(self.viewer.preview_active)
        self.viewer.set_current_time(0.2)
        self.assertFalse(self.viewer.preview_active)
        self.assertFalse(self.viewer.canvas.preview_visible)
        np.testing.assert_allclose(
            self.viewer.preview_state.get_qpos(),
            self.viewer.committed_state.get_qpos(),
        )

    def test_logical_pelvis_selection_maps_to_mujoco_pelvis(self):
        self.window.controls.frame_box.setCurrentText("pelvis")
        self.assertEqual(
            self.viewer._selected_target(), ("body", "robot/pelvis")
        )

    def test_completed_edits_upsert_logical_targets_by_time(self):
        self.window.on_target_pose_drag_finished(0.10, 0.20, 0.80)
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.window.on_target_pose_drag_finished(0.30, 0.20, 0.85)

        frame_name = self.window.controls.frame_box.currentText()
        track = self.window.trajectory.tracks[frame_name]
        self.assertEqual([frame.time for frame in track], [0.0, 0.2])
        self.assertAlmostEqual(track[0].x, 0.10)
        self.assertAlmostEqual(track[1].x, 0.30)

        self.window.controls.time_slider.set_value(0.0)
        self.window.controls.emit_time_changed(0.0)
        self.assertAlmostEqual(self.window.controls.x_slider.value(), 0.10)

    def test_viewer_timeslice_scrubber_loads_editable_time(self):
        self.viewer.timeslice_slider.setValue(20)
        self.viewer._emit_timeslice_slider_time()

        self.assertEqual(self.viewer.get_current_time(), 0.2)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.2)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.2)

    def test_timeslice_scrubber_samples_live_without_creating_intermediate_states(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.20
        self.viewer.set_robot_trajectory(
            [first, second],
            times=[0.0, 1.0],
        )
        initial_times = self.viewer.state_timeline.times()

        self.viewer._preview_timeslice_slider_value(50)

        self.assertEqual(self.viewer.get_current_time(), 0.0)
        self.assertEqual(self.viewer.state_timeline.times(), initial_times)
        self.assertAlmostEqual(self.viewer.display_time, 0.5)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.5)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.5)
        self.assertAlmostEqual(
            self.viewer.playback_state.get_qpos()[-1],
            first[-1] + 0.10,
        )
        self.assertEqual(
            self.viewer.timeslice_frame_readout.text(),
            "Frame 1 / 2",
        )

        self.viewer._emit_timeslice_slider_time()

        self.assertEqual(self.viewer.get_current_time(), 0.5)
        self.assertIn(0.5, self.viewer.state_timeline.times())
        self.assertAlmostEqual(
            self.viewer.committed_state.get_qpos()[-1],
            first[-1] + 0.10,
        )
        kind, name = self.viewer._selected_target()
        position, quaternion = self.viewer.committed_state.get_body_pose(
            name, kind
        )
        roll, pitch, yaw = quat_to_rpy(quaternion)
        np.testing.assert_allclose(
            [
                self.window.controls.x_slider.value(),
                self.window.controls.y_slider.value(),
                self.window.controls.z_slider.value(),
            ],
            position,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            [
                self.window.controls.roll_slider.value(),
                self.window.controls.pitch_slider.value(),
                self.window.controls.yaw_slider.value(),
            ],
            [roll, pitch, yaw],
            atol=np.deg2rad(0.11),
        )

    def test_playback_advances_elapsed_time_and_interpolates_irregular_frames(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.24
        self.viewer.set_robot_trajectory(
            [first, second],
            times=[0.2, 0.8],
        )

        self.viewer.preview_trajectory_time(0.2)
        self.viewer._advance_playback(elapsed=0.15)

        self.assertAlmostEqual(self.viewer.display_time, 0.35)
        self.assertAlmostEqual(
            self.viewer.playback_state.get_qpos()[-1],
            first[-1] + 0.06,
        )
        self.assertEqual(self.viewer.timeslice_slider.value(), 35)
        self.assertEqual(
            self.viewer.timeslice_frame_readout.text(),
            "Frame 1 / 2",
        )

    def test_playback_speed_scales_elapsed_time_without_changing_timer_rate(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.24
        self.viewer.set_robot_trajectory(
            [first, second],
            times=[0.2, 0.8],
        )
        self.viewer.preview_trajectory_time(0.2)
        self.viewer.playback_speed.setValue(2.0)

        self.viewer._advance_playback(elapsed=0.15)

        self.assertAlmostEqual(self.viewer.display_time, 0.5)
        self.assertAlmostEqual(
            self.viewer.playback_state.get_qpos()[-1],
            first[-1] + 0.12,
        )
        self.assertEqual(self.viewer.play_timer.interval(), 33)

    def test_live_scrub_render_updates_are_coalesced(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.20
        self.viewer.set_robot_trajectory(
            [first, second],
            times=[0.0, 1.0],
        )

        self.viewer._preview_timeslice_slider_value(25)
        first_rendered_value = self.viewer.playback_state.get_qpos()[-1]
        self.viewer._preview_timeslice_slider_value(75)

        self.assertAlmostEqual(
            self.viewer.playback_state.get_qpos()[-1],
            first_rendered_value,
        )
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.75)

        self.viewer._flush_pending_scrub_preview()

        self.assertAlmostEqual(self.viewer.display_time, 0.75)
        self.assertAlmostEqual(
            self.viewer.playback_state.get_qpos()[-1],
            first[-1] + 0.15,
        )

    def test_timeslice_wheel_scroll_updates_active_time(self):
        self.assertEqual(self.viewer.get_current_time(), 0.0)

        event = QWheelEvent(
            QPointF(100, 15),
            QPointF(100, 15),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.viewer.timeslice_slider.wheelEvent(event)

        self.assertEqual(self.viewer.timeslice_slider.value(), 2)
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.02)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.02)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.02)

    def test_timeslice_bar_uses_single_compact_time_display(self):
        self.assertEqual(self.viewer.timeslice_label.text(), "Time")
        self.assertEqual(self.viewer.timeslice_time_input.suffix(), " s")
        self.assertEqual(self.viewer.timeslice_step_label.text(), "Keyframe interval")
        self.assertEqual(self.viewer.timeslice_step_input.suffix(), " s")
        self.assertAlmostEqual(self.viewer.timeslice_step_input.value(), 0.10)
        self.assertEqual(self.viewer.timeslice_duration_label.text(), "Max time")
        self.assertEqual(self.viewer.timeslice_duration_input.suffix(), " s")
        self.assertAlmostEqual(self.viewer.timeslice_duration_input.value(), 5.0)
        self.assertEqual(self.viewer.timeslice_frame_readout.text(), "Frame —")
        self.assertFalse(hasattr(self.viewer, "frame_slider"))
        self.assertFalse(hasattr(self.viewer, "timeslice_time_label"))
        self.assertFalse(hasattr(self.viewer, "accept_timeslice_button"))
        self.assertEqual(
            self.viewer.delete_timeslice_button.text(), "Delete Keyframe"
        )

    def test_timeline_duration_updates_bottom_and_sidebar_time_ranges(self):
        self.viewer.timeslice_duration_input.setValue(8.0)

        self.assertAlmostEqual(self.viewer.timeline_duration, 8.0)
        self.assertEqual(self.viewer.timeslice_slider.maximum(), 800)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.maximum(), 8.0)
        self.assertEqual(self.window.controls.time_slider.slider.maximum(), 800)
        self.assertAlmostEqual(self.window.controls.time_slider.input.maximum(), 8.0)

        self.window.controls.time_slider.set_value(7.5)
        self.window.controls.emit_time_changed(7.5)

        self.assertAlmostEqual(self.viewer.get_current_time(), 7.5)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 7.5)

    def test_accept_slice_captures_all_logical_targets_from_committed_pose(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        preview_qpos = self.viewer.preview_state.get_qpos()
        expected_state = self.viewer.robot_model.create_state()
        expected_state.set_qpos(preview_qpos)

        self.viewer.accept_timeslice()

        self.assertFalse(self.viewer.preview_active)
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.3)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.3)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.3)
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.2), preview_qpos
        )
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.2})
        state = expected_state
        expected_names = []
        for frame_name, (kind, object_name) in self.viewer.frame_bindings.items():
            try:
                state.get_body_pose(object_name, kind)
            except KeyError:
                continue
            expected_names.append(frame_name)

        actual_names = {
            frame.frame_name
            for frame in self.window.trajectory.frames
            if abs(frame.time - 0.2) <= 1e-6
        }
        self.assertEqual(actual_names, set(expected_names))

        for frame_name in expected_names:
            with self.subTest(frame_name=frame_name):
                frame = next(
                    frame for frame in self.window.trajectory.tracks[frame_name]
                    if abs(frame.time - 0.2) <= 1e-6
                )
                kind, object_name = self.viewer.frame_bindings[frame_name]
                position, quaternion = state.get_body_pose(object_name, kind)
                roll, pitch, yaw = quat_to_rpy(quaternion)
                np.testing.assert_allclose(
                    [frame.x, frame.y, frame.z], position, atol=1e-9
                )
                np.testing.assert_allclose(
                    [frame.roll, frame.pitch, frame.yaw],
                    [roll, pitch, yaw],
                    atol=1e-9,
                )

    def test_accept_slice_uses_visible_playback_time_without_duplicate_keyframe(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.10
        self.viewer.set_robot_trajectory(
            [first, second],
            times=[0.0, 1.0],
        )
        self.viewer.start_playback()
        self.viewer.preview_trajectory_time(0.6, emit_time_signal=True)

        name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            name, self.viewer.preview_state.get_joint_value(name) + 0.05
        )
        preview_qpos = self.viewer.preview_state.get_qpos()

        self.assertAlmostEqual(self.viewer.get_current_time(), 0.0)
        self.assertAlmostEqual(self.viewer.display_time, 0.6)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.6)

        self.viewer.accept_timeslice()

        self.assertFalse(self.viewer.play_timer.isActive())
        self.assertFalse(self.viewer.preview_active)
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.6})
        self.assertEqual(
            {round(frame.time, 6) for frame in self.window.trajectory.frames},
            {0.6},
        )
        np.testing.assert_allclose(
            self.viewer.state_timeline.get_state(0.6),
            preview_qpos,
        )

    def test_accept_slice_auto_advance_uses_step_control(self):
        self.viewer.timeslice_step_input.setValue(0.25)
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)

        self.viewer.accept_timeslice()

        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.2})
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.45)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.45)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.45)
        self.assertIn("advanced to t=0.45 s", self.viewer.status_label.text())

    def test_accept_slice_auto_advance_clamps_at_timeline_end(self):
        self.viewer.timeslice_step_input.setValue(1.0)
        self.window.controls.time_slider.set_value(4.8)
        self.window.controls.emit_time_changed(4.8)

        self.viewer.accept_timeslice()

        self.assertEqual(self.viewer.timeslice_slider.defined_times, {4.8})
        self.assertAlmostEqual(self.viewer.get_current_time(), 5.0)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 5.0)
        self.assertIn("advanced to t=5.00 s", self.viewer.status_label.text())

    def test_accept_slice_auto_advance_clamps_at_custom_timeline_end(self):
        self.viewer.timeslice_duration_input.setValue(8.0)
        self.viewer.timeslice_step_input.setValue(1.0)
        self.window.controls.time_slider.set_value(7.8)
        self.window.controls.emit_time_changed(7.8)

        self.viewer.accept_timeslice()

        self.assertEqual(self.viewer.timeslice_slider.defined_times, {7.8})
        self.assertAlmostEqual(self.viewer.get_current_time(), 8.0)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 8.0)
        self.assertIn("advanced to t=8.00 s", self.viewer.status_label.text())

    def test_defined_timeslice_marker_snaps_back_to_slice_time(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer._joint_changed(
            self.viewer.preview_state.get_joint_names()[-1],
            self.viewer.preview_state.get_joint_value(
                self.viewer.preview_state.get_joint_names()[-1]
            ) + 0.05,
        )
        self.viewer.accept_timeslice()
        self.viewer.set_current_time(0.0)

        snapped = self.viewer.timeslice_slider.snap_to_nearest_defined_time(0.21)

        self.assertTrue(snapped)
        self.assertEqual(self.viewer.get_current_time(), 0.2)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.2)

    def test_clicking_undefined_timeslice_groove_changes_time(self):
        self.viewer.timeslice_slider.resize(500, 30)
        self.viewer.timeslice_slider.set_defined_times([0.2])
        pixel = self.viewer.timeslice_slider._time_to_pixel(0.37)

        self.viewer.timeslice_slider.activate_time_at_pixel(pixel)

        self.assertEqual(self.viewer.get_current_time(), 0.37)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.37)

    def test_pressing_current_defined_slice_handle_allows_normal_drag(self):
        self.viewer.timeslice_slider.resize(500, 30)
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer.timeslice_slider.set_defined_times([0.2])
        activated = []
        self.viewer.timeslice_slider.marker_activated.connect(activated.append)
        handle_center = self.viewer.timeslice_slider._handle_rect().center()
        position = QPointF(handle_center)
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            position,
            position,
            position,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

        self.viewer.timeslice_slider.mousePressEvent(event)

        self.assertEqual(activated, [])
        self.assertEqual(self.viewer.get_current_time(), 0.2)

    def test_timeslice_markers_use_requested_dimensions(self):
        slider = self.viewer.timeslice_slider
        slider.resize(500, 40)
        x = slider._time_to_pixel(0.2)

        normal_marker = slider._marker_rect(x, current=False)
        current_marker = slider._marker_rect(x, current=True)

        self.assertEqual(
            (normal_marker.width(), normal_marker.height()),
            (2, 6),
        )
        self.assertEqual(
            (current_marker.width(), current_marker.height()),
            (2, 10),
        )

    def test_delete_slice_removes_defined_marker_and_logical_targets(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer._joint_changed(
            self.viewer.preview_state.get_joint_names()[-1],
            self.viewer.preview_state.get_joint_value(
                self.viewer.preview_state.get_joint_names()[-1]
            ) + 0.05,
        )
        self.viewer.accept_timeslice()
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.2})

        self.viewer.set_current_time(0.2)
        self.viewer.delete_timeslice()

        self.assertNotIn(0.2, self.viewer.timeslice_slider.defined_times)
        self.assertFalse(
            any(abs(frame.time - 0.2) <= 1e-6 for frame in self.window.trajectory.frames)
        )
        self.assertIsNone(self.viewer.state_timeline.get_state(0.2))

    def test_clear_trajectory_requires_confirmation(self):
        self.window.on_target_pose_drag_finished(0.10, 0.20, 0.80)
        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.No
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(len(self.window.trajectory.frames), 1)

    def test_clear_trajectory_removes_keyframes_markers_and_playback(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        self.viewer._joint_changed(
            self.viewer.preview_state.get_joint_names()[-1],
            self.viewer.preview_state.get_joint_value(
                self.viewer.preview_state.get_joint_names()[-1]
            ) + 0.05,
        )
        self.viewer.accept_timeslice()
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05
        self.viewer.set_robot_trajectory([first, second])
        self.viewer.show_ghosts.setChecked(True)
        pose_before = self.viewer.committed_state.get_qpos()
        self.assertTrue(self.window.trajectory.frames)
        self.assertTrue(self.viewer.robot_trajectory)
        self.assertTrue(self.viewer.timeslice_slider.defined_times)

        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.Yes
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(self.window.trajectory.frames, [])
        self.assertEqual(self.window.active_index, -1)
        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.robot_trajectory_times, [])
        self.assertEqual(self.viewer.ghost_trajectory, [])
        self.assertEqual(self.viewer.timeslice_slider.defined_times, set())
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.0)
        self.assertEqual(self.viewer.timeslice_slider.value(), 0)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.0)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.0)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), pose_before)

    def test_clear_trajectory_drops_stale_editable_qpos_times(self):
        self.window.controls.time_slider.set_value(0.2)
        self.window.controls.emit_time_changed(0.2)
        joint_name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            joint_name,
            self.viewer.preview_state.get_joint_value(joint_name) + 0.05,
        )
        self.viewer.accept_timeslice()
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.3)
        stale_qpos = self.viewer.state_timeline.get_state(0.2)
        self.assertFalse(np.allclose(stale_qpos, self.viewer.robot_model.home_qpos))

        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.Yes
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(self.viewer.state_timeline.times(), [0.0])
        self.assertAlmostEqual(self.viewer.get_current_time(), 0.0)
        self.assertEqual(self.viewer.timeslice_slider.value(), 0)
        self.assertAlmostEqual(self.viewer.timeslice_time_input.value(), 0.0)
        self.assertAlmostEqual(self.window.controls.time_slider.value(), 0.0)

        self.viewer.reset_robot_pose()

        np.testing.assert_allclose(
            self.viewer.committed_state.get_qpos(),
            self.viewer.robot_model.home_qpos,
        )
        self.assertFalse(np.allclose(self.viewer.committed_state.get_qpos(), stale_qpos))

    def test_workflow_toolbar_mirrors_common_controls(self):
        toolbar = self.window.findChild(QToolBar, "workflowToolbar")
        self.assertIs(toolbar, self.window.app_toolbar)
        self.assertFalse(hasattr(self.viewer, "quick_actions_panel"))
        self.assertEqual(
            toolbar.toolButtonStyle(),
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        if self.window.font().pointSizeF() > 0:
            self.assertAlmostEqual(
                toolbar.font().pointSizeF(),
                max(6.0, self.window.font().pointSizeF() - 2.0),
            )

        actions = [action for action in toolbar.actions() if not action.isSeparator()]
        self.assertEqual(
            [action.text() for action in actions],
            [
                "Preview Path",
                "Commit Keyframe",
                "Generate",
                "Play",
                "Reset",
                "Clear",
                "Move",
                "Rotate",
                "Gizmo",
                "Undo",
                "Redo",
            ],
        )
        for action in actions:
            self.assertFalse(action.icon().isNull(), action.text())

        generated = []
        self.viewer.generate_requested.connect(lambda: generated.append(True))
        self.window.generate_action.trigger()
        self.assertEqual(generated, [True])

        cleared = []
        self.viewer.clear_trajectory_requested.connect(lambda: cleared.append(True))
        self.window.clear_action.trigger()
        self.assertEqual(cleared, [True])

        self.window.show_playback_poses_action.setChecked(True)
        self.assertTrue(self.viewer.show_ghosts.isChecked())
        self.viewer.show_ghosts.setChecked(False)
        self.assertFalse(self.window.show_playback_poses_action.isChecked())

        self.window.rotate_action.trigger()
        self.assertEqual(self.viewer.canvas.gizmo.mode, "rotate")
        self.assertTrue(self.window.rotate_action.isChecked())
        self.viewer.canvas.set_gizmo_mode("translate")
        self.assertTrue(self.window.move_action.isChecked())
        self.assertFalse(self.window.rotate_action.isChecked())
        self.assertTrue(self.window.gizmo_visibility_action.isChecked())
        self.assertTrue(self.viewer.canvas.transform_gizmo_visible)
        self.window.gizmo_visibility_action.trigger()
        self.assertFalse(self.window.gizmo_visibility_action.isChecked())
        self.assertFalse(self.viewer.canvas.transform_gizmo_visible)
        self.assertFalse(self.viewer.canvas.transform_gizmo_interactive)
        self.assertFalse(self.window.move_action.isEnabled())
        self.assertFalse(self.window.rotate_action.isEnabled())
        self.window.gizmo_visibility_action.trigger()
        self.assertTrue(self.viewer.canvas.transform_gizmo_visible)
        self.assertTrue(self.viewer.canvas.transform_gizmo_interactive)

        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05
        self.viewer.set_robot_trajectory([first, second])
        self.window.playback_action.trigger()
        self.assertTrue(self.viewer.play_timer.isActive())
        self.assertEqual(self.window.playback_action.text(), "Pause")
        self.assertEqual(self.viewer.play_button.text(), "Pause")
        self.window.playback_action.trigger()
        self.assertFalse(self.viewer.play_timer.isActive())
        self.assertEqual(self.window.playback_action.text(), "Play")
        self.assertEqual(self.viewer.play_button.text(), "Play")

        before = self.viewer.committed_state.get_qpos()
        joint_name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            joint_name,
            self.viewer.preview_state.get_joint_value(joint_name) + 0.05,
        )
        preview = self.viewer.preview_state.get_qpos()
        self.window.preview_action.trigger()
        self.assertEqual(len(self.viewer.robot_trajectory), 2)
        np.testing.assert_allclose(self.viewer.robot_trajectory[0], first)
        np.testing.assert_allclose(self.viewer.robot_trajectory[-1], second)
        self.assertEqual(len(self.viewer.ghost_trajectory), 40)
        self.assertEqual(self.viewer.ghost_source, "preview_path")
        np.testing.assert_allclose(self.viewer.ghost_trajectory[0], before)
        np.testing.assert_allclose(self.viewer.ghost_trajectory[-1], preview)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), before)
        self.assertTrue(self.viewer.preview_active)
        self.assertTrue(self.viewer.canvas.show_ghosts)

    def test_editing_mode_switches_sidebar_editor_and_gizmo(self):
        controls = self.window.controls
        self.assertEqual(
            [
                controls.editing_mode_bar.tabText(index)
                for index in range(controls.editing_mode_bar.count())
            ],
            ["End Effector", "Joint Angles"],
        )
        self.assertTrue(controls.editing_mode_bar.expanding())
        self.assertEqual(controls.editing_mode(), "end_effector")
        self.assertTrue(self.viewer.canvas.transform_gizmo_visible)
        self.assertTrue(self.window.gizmo_visibility_action.isEnabled())
        self.assertIs(
            controls.editing_mode_stack.currentWidget(),
            controls.end_effector_page,
        )
        for slider in (
            controls.x_slider,
            controls.y_slider,
            controls.z_slider,
            controls.roll_slider,
            controls.pitch_slider,
            controls.yaw_slider,
        ):
            self.assertTrue(controls.end_effector_page.isAncestorOf(slider))

        joint_page = self.viewer.joint_editor_widget()
        self.assertIs(controls.joint_editor_stack.currentWidget(), joint_page)
        self.assertTrue(controls.joint_editor_stack.isAncestorOf(joint_page))
        self.assertEqual(joint_page.frameShape(), QFrame.Shape.NoFrame)
        self.assertGreater(joint_page.maximumWidth(), 244)

        controls.set_editing_mode("joint_angles")
        self.app.processEvents()
        self.assertEqual(controls.editing_mode(), "joint_angles")
        self.assertIs(
            controls.editing_mode_stack.currentWidget(),
            controls.joint_editor_stack,
        )
        self.assertTrue(self.viewer.canvas.transform_gizmo_visible)
        self.assertTrue(self.viewer.canvas.transform_gizmo_interactive)
        self.assertTrue(self.window.gizmo_visibility_action.isEnabled())
        self.assertTrue(self.window.gizmo_visibility_action.isChecked())
        self.assertTrue(self.window.move_action.isEnabled())
        self.assertTrue(self.window.rotate_action.isEnabled())
        self.window.rotate_action.trigger()
        self.assertEqual(self.viewer.canvas.gizmo.mode, "rotate")

        self.window.gizmo_visibility_action.trigger()
        self.assertFalse(self.viewer.canvas.transform_gizmo_visible)
        self.assertFalse(self.viewer.canvas.transform_gizmo_interactive)
        self.assertFalse(self.window.move_action.isEnabled())
        self.assertFalse(self.window.rotate_action.isEnabled())
        self.window.gizmo_visibility_action.trigger()
        self.assertTrue(self.viewer.canvas.transform_gizmo_visible)
        self.assertTrue(self.viewer.canvas.transform_gizmo_interactive)

        joint_name = self.viewer.preview_state.get_joint_names()[-1]
        self.viewer._joint_changed(
            joint_name,
            self.viewer.preview_state.get_joint_value(joint_name),
        )
        expected_rpy = quat_to_rpy(self.viewer.last_valid_target_quaternion)
        actual_pose = (
            controls.x_slider.value(),
            controls.y_slider.value(),
            controls.z_slider.value(),
            controls.roll_slider.value(),
            controls.pitch_slider.value(),
            controls.yaw_slider.value(),
        )
        np.testing.assert_allclose(
            actual_pose,
            (*self.viewer.last_valid_target_position, *expected_rpy),
            atol=1e-3,
        )

        controls.set_editing_mode("end_effector")
        self.app.processEvents()
        self.assertTrue(self.viewer.canvas.transform_gizmo_visible)
        self.assertTrue(self.viewer.canvas.transform_gizmo_interactive)
        self.assertTrue(self.window.gizmo_visibility_action.isEnabled())
        self.assertTrue(self.window.move_action.isEnabled())
        self.assertTrue(self.window.rotate_action.isEnabled())

    def test_render_progress_overlay_waits_for_active_3d_geometry_progress(self):
        self.window.begin_render_progress(
            "Rendering test model",
            "Preparing geometry...",
            viewer=self.viewer,
        )

        overlay = self.window.render_progress_overlay
        self.assertFalse(overlay.isHidden())
        self.assertIs(overlay.parentWidget(), self.window.viewer_3d_stack)
        self.assertEqual(overlay.geometry(), self.window.viewer_3d_stack.rect())
        self.assertLessEqual(
            abs(overlay.card.geometry().center().x() - overlay.rect().center().x()),
            1,
        )
        self.assertLessEqual(
            abs(overlay.card.geometry().center().y() - overlay.rect().center().y()),
            1,
        )
        self.assertFalse(overlay._allow_close)

        self.window.on_viewer_geometry_progress(self.viewer, 2, 8)

        self.assertFalse(overlay.isHidden())
        self.assertEqual(overlay.progress_bar.value(), 25)
        self.assertIn(
            "2/8",
            overlay.detail_label.text(),
        )

        self.window.on_viewer_geometry_progress(self.viewer, 8, 8)

        self.assertTrue(overlay.isHidden())
        self.assertTrue(overlay._allow_close)

    def test_initial_render_progress_overlay_is_deferred_until_window_is_shown(self):
        self.assertTrue(self.window.render_progress_overlay.isHidden())
        self.assertIsNotNone(self.window.pending_initial_render_progress)

        self.window.show()
        for _ in range(3):
            self.app.processEvents()

        self.assertIsNotNone(self.window.pending_initial_render_progress)
        self.assertTrue(self.window.render_progress_overlay.isHidden())

        QTest.qWait(INITIAL_RENDER_PROGRESS_DELAY_MS + 100)
        self.app.processEvents()

        self.assertIsNone(self.window.pending_initial_render_progress)
        overlay = self.window.render_progress_overlay
        self.assertFalse(overlay.isHidden())
        self.assertIs(overlay.parentWidget(), self.window.viewer_3d_stack)
        self.assertEqual(overlay.geometry(), self.window.viewer_3d_stack.rect())

    def test_sidebars_are_resizable_shells_with_collapsible_sections(self):
        self.window.resize(1700, 800)
        self.window.viewer_tabs.setCurrentWidget(self.window.viewer_3d_stack)
        self.window.show()
        self.app.processEvents()
        viewer_identity = id(self.window.viewer_3d)
        model_identity = id(self.window.robot_model_3d.mj_model)
        qpos = self.viewer.robot_state.get_qpos()

        for sidebar in (self.window.left_sidebar, self.window.right_sidebar):
            with self.subTest(sidebar=sidebar):
                self.assertFalse(hasattr(sidebar, "set_collapsed"))
                self.assertFalse(hasattr(sidebar, "toggle_collapsed"))
                self.assertFalse(hasattr(sidebar, "collapsed_width"))
                self.assertFalse(sidebar.isHidden())

        self.assertFalse(hasattr(self.viewer, "controls_sidebar"))
        self.assertFalse(hasattr(self.viewer, "viewer_splitter"))
        self.assertIsNone(self.window.right_sidebar_content.current_context_widget())
        self.assertEqual(
            self.window.left_sidebar_content.body_layout.contentsMargins().left(),
            4,
        )
        self.assertEqual(
            self.window.right_sidebar_content.body_layout.contentsMargins().left(),
            0,
        )
        self.assertEqual(
            self.window.left_sidebar.minimumWidth(),
            LEFT_SIDEBAR_MIN_WIDTH,
        )
        self.assertGreater(
            self.window.left_sidebar.maximumWidth(),
            LEFT_SIDEBAR_MIN_WIDTH,
        )
        self.assertEqual(
            self.window.left_sidebar.maximumWidth(),
            SIDEBAR_MAX_WIDTH,
        )
        self.assertEqual(
            self.window.right_sidebar.minimumWidth(),
            RIGHT_SIDEBAR_MIN_WIDTH,
        )
        self.assertEqual(
            self.window.right_sidebar.maximumWidth(),
            SIDEBAR_MAX_WIDTH,
        )
        self.assertGreater(
            self.window.status_panel.maximumWidth(),
            SIDEBAR_MAX_WIDTH,
        )
        left_titles = [
            section.title for section in self.window.left_sidebar_content.sections
        ]
        right_titles = [
            section.title for section in self.window.right_sidebar_content.sections
        ]
        self.assertEqual(
            left_titles,
            ["Target", "Editing Mode", "Planning"],
        )
        self.assertEqual(
            right_titles,
            ["Status", "IK / Constraints"],
        )
        expected_visible = {
            "Target": True,
            "Editing Mode": True,
            "Planning": False,
            "IK / Constraints": False,
            "Status": True,
        }
        for section in (
            self.window.left_sidebar_content.sections
            + self.window.right_sidebar_content.sections
        ):
            self.assertEqual(
                section.content.isVisible(),
                expected_visible[section.title],
            )
        self.assertEqual(
            [action.text().replace("&", "") for action in self.window.menuBar().actions()],
            ["File", "Robot", "View", "Help"],
        )
        toolbar = self.window.findChild(QToolBar, "workflowToolbar")
        self.assertIs(toolbar, self.window.app_toolbar)
        self.assertFalse(hasattr(self.window, "workflow_toolbar"))
        self.assertTrue(self.window.viewer_tabs.tabBar().isHidden())
        toolbar_buttons = {
            button.objectName(): button.text()
            for button in self.window.app_toolbar.findChildren(QToolButton)
            if button.objectName()
        }
        self.assertEqual(toolbar_buttons["planPreviewButton"], "Preview Path")
        self.assertEqual(toolbar_buttons["sliceButton"], "Commit Keyframe")
        self.assertEqual(toolbar_buttons["quickGenerateButton"], "Generate")
        self.assertEqual(toolbar_buttons["playbackToolbarButton"], "Play")
        self.assertEqual(toolbar_buttons["resetToolbarButton"], "Reset")
        self.assertEqual(toolbar_buttons["clearToolbarButton"], "Clear")
        self.assertEqual(toolbar_buttons["moveToolButton"], "Move")
        self.assertEqual(toolbar_buttons["rotateToolButton"], "Rotate")
        self.assertEqual(toolbar_buttons["gizmoVisibilityButton"], "Gizmo")
        self.assertEqual(toolbar_buttons["undoToolbarButton"], "Undo")
        self.assertEqual(toolbar_buttons["redoToolbarButton"], "Redo")
        project_action_texts = [
            action.text()
            for action in self.window.file_menu.actions()
            if action.text()
        ]
        self.assertEqual(
            [text.replace("&", "") for text in project_action_texts],
            [
                "New Project…",
                "Open Project…",
                "Open Recent",
                "Save",
                "Import",
                "Export",
            ],
        )
        self.assertFalse(hasattr(self.window, "save_as_action"))
        self.assertEqual(len(self.window.recent_projects_menu.actions()), 1)
        self.assertEqual(
            self.window.recent_projects_menu.actions()[0].text(),
            "No recent projects",
        )
        self.assertFalse(
            self.window.recent_projects_menu.actions()[0].isEnabled()
        )
        self.assertIn(self.window.model_key, self.window.robot_actions)
        self.assertTrue(
            self.window.robot_actions[self.window.model_key].isChecked()
        )
        self.assertEqual(
            [action.text() for action in self.window.import_menu.actions()],
            ["Robot Model…", "Qpos…", "Trajectory…"],
        )
        self.assertEqual(
            [action.text() for action in self.window.export_menu.actions()],
            ["Qpos…", "Trajectory…"],
        )
        self.assertEqual(self.window.viewer_tabs.tabText(0), "3D Pose")
        self.assertIs(
            self.window.viewer_tabs.widget(0), self.window.viewer_3d_stack
        )
        self.assertEqual(
            [action.text() for action in self.window.view_actions],
            ["3D Pose", "Simulation"],
        )
        self.assertIsNone(self.window.controls.view_panel.parent())

        self.assertTrue(
            self.window.model_source_label.text().startswith("Model source:")
        )
        self.assertFalse(self.window.status_details_button.isChecked())
        self.assertFalse(self.window.status_details_panel.isVisible())
        self.assertIsNone(self.window.backend_label.parent())
        self.assertIsNone(self.window.viewer_time_label.parent())
        self.assertIsNone(self.window.model_source_label.parent())
        self.assertIs(
            self.window.status_text.parent(),
            self.window.status_details_panel,
        )
        robot_labels = [
            label.text()
            for label in self.viewer.robot_context_widget().findChildren(QLabel)
        ]
        self.assertFalse(
            any(text.startswith("Model:") for text in robot_labels)
        )
        self.assertTrue(self.window.controls.phase_label.isHidden())
        self.assertTrue(self.window.controls.phase_box.isHidden())
        self.assertTrue(self.window.controls.table.isColumnHidden(1))
        self.assertTrue(self.window.controls.open_model_button.isHidden())
        self.assertTrue(self.window.controls.choose_mesh_folder_button.isHidden())
        self.assertTrue(self.viewer.trajectory_csv_group.isHidden())
        self.assertTrue(self.viewer.qpos_csv_group.isHidden())
        self.assertTrue(self.viewer.trajectory_import_dt.isHidden())
        self.assertEqual(
            [
                self.window.controls.import_action_box.itemText(index)
                for index in range(self.window.controls.import_action_box.count())
            ],
            ["Model", "Qpos", "Trajectory"],
        )
        self.assertEqual(
            [
                self.window.controls.export_action_box.itemText(index)
                for index in range(self.window.controls.export_action_box.count())
            ],
            ["Qpos", "Trajectory"],
        )
        self.window.controls.import_action_box.setCurrentIndex(1)
        event = QWheelEvent(
            QPointF(100, 15),
            QPointF(100, 15),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.window.controls.import_action_box.wheelEvent(event)
        self.assertEqual(self.window.controls.import_action_box.currentIndex(), 1)
        self.assertEqual(self.viewer.trajectory_csv_group.title(), "Trajectory CSV")
        self.assertEqual(self.viewer.qpos_csv_group.title(), "Qpos CSV")
        self.assertEqual(self.viewer.load_trajectory_button.text(), "Load")
        self.assertEqual(self.viewer.save_trajectory_button.text(), "Save")
        self.assertEqual(self.viewer.load_qpos_button.text(), "Load")
        self.assertEqual(self.viewer.save_qpos_button.text(), "Save")
        trajectory_widgets = set(
            self.window.controls.trajectory_panel.findChildren(QWidget)
        )
        for timeslice_widget in (
            self.window.controls.corner_smoothing_slider,
            self.viewer.timeslice_step_label,
            self.viewer.timeslice_step_input,
            self.viewer.timeslice_duration_label,
            self.viewer.timeslice_duration_input,
            self.viewer.collision_substeps_label,
            self.viewer.collision_substeps,
            self.viewer.playback_speed_label,
            self.viewer.playback_speed,
            self.viewer.ghost_stride_label,
            self.viewer.ghost_stride,
            self.viewer.ghost_alpha_label,
            self.viewer.ghost_alpha,
            self.viewer.preview_alpha_label,
            self.viewer.preview_alpha,
        ):
            self.assertIn(timeslice_widget, trajectory_widgets)
        for removed_widget in (
            self.window.controls.time_slider,
            self.window.controls.add_button,
            self.window.controls.update_button,
            self.window.controls.delete_button,
            self.window.controls.generate_button,
            self.viewer.generate_button,
            self.viewer.play_button,
        ):
            self.assertNotIn(removed_widget, trajectory_widgets)
        self.assertIs(
            self.window.controls.corner_smoothing_slider.parent(),
            self.viewer.timeslice_context_panel,
        )
        self.assertIs(
            self.window.right_sidebar_content.sections[1].content,
            self.window.controls.preview_ik_panel,
        )
        self.assertIs(
            self.window.controls.display_context_widget().parent(),
            self.window.controls.display_context_stack,
        )
        self.assertFalse(hasattr(self.viewer, "frame_slider"))
        self.assertFalse(hasattr(self.viewer, "timeslice_frame_row"))
        self.assertIs(
            self.viewer.timeslice_layout.itemAt(0).widget(),
            self.viewer.timeslice_timeline_group,
        )
        self.assertEqual(self.viewer.timeslice_timeline_group.title(), "Timeline")
        self.assertIs(
            self.viewer.timeslice_timeline_layout.itemAt(0).layout(),
            self.viewer.timeslice_scrubber_layout,
        )
        self.assertIs(
            self.viewer.timeslice_scrubber_layout.itemAt(0).layout(),
            self.viewer.timeslice_time_row,
        )
        self.assertEqual(self.viewer.timeslice_scrubber_layout.count(), 1)
        self.assertIs(
            self.viewer.timeslice_timeline_layout.itemAt(1).layout(),
            self.viewer.timeslice_action_row,
        )
        self.assertIs(
            self.viewer.timeslice_time_row.itemAt(3).widget(),
            self.viewer.timeslice_frame_readout,
        )
        self.assertIs(
            self.window.controls.corner_smoothing_slider.parent(),
            self.viewer.timeslice_context_panel,
        )
        smoothing_row, _ = self.viewer.timeslice_context_layout.getWidgetPosition(
            self.window.controls.corner_smoothing_slider
        )
        collision_row, _ = self.viewer.timeslice_context_layout.getWidgetPosition(
            self.viewer.collision_substeps
        )
        self.assertEqual(collision_row, smoothing_row + 1)
        self.assertIs(
            self.viewer.timeslice_context_layout.labelForField(
                self.viewer.collision_substeps
            ),
            self.viewer.collision_substeps_label,
        )
        playback_speed_row, _ = (
            self.viewer.timeslice_context_layout.getWidgetPosition(
                self.viewer.playback_speed
            )
        )
        playback_spacing_row, _ = (
            self.viewer.timeslice_context_layout.getWidgetPosition(
                self.viewer.ghost_stride
            )
        )
        self.assertEqual(playback_speed_row, collision_row + 1)
        self.assertEqual(playback_spacing_row, playback_speed_row + 1)
        self.assertIs(
            self.viewer.timeslice_context_layout.labelForField(
                self.viewer.playback_speed
            ),
            self.viewer.playback_speed_label,
        )
        self.assertEqual(self.viewer.playback_speed.minimum(), 0.10)
        self.assertEqual(self.viewer.playback_speed.maximum(), 4.00)
        self.assertEqual(self.viewer.playback_speed.value(), 1.00)
        self.assertEqual(self.viewer.playback_speed.suffix(), "×")
        playback_opacity_row, _ = (
            self.viewer.timeslice_context_layout.getWidgetPosition(
                self.viewer.ghost_alpha
            )
        )
        preview_opacity_row, _ = (
            self.viewer.timeslice_context_layout.getWidgetPosition(
                self.viewer.preview_alpha
            )
        )
        self.assertEqual(preview_opacity_row, playback_opacity_row + 1)
        self.assertIs(
            self.viewer.timeslice_context_layout.labelForField(
                self.viewer.preview_alpha
            ),
            self.viewer.preview_alpha_label,
        )
        self.assertIs(
            self.viewer.preview_alpha.parent(),
            self.viewer.timeslice_context_panel,
        )
        for removed_button in (
            "plan_preview_button",
            "accept_preview_button",
            "cancel_preview_button",
        ):
            self.assertFalse(hasattr(self.viewer, removed_button))
        self.assertIs(
            self.viewer.timeslice_context_layout.labelForField(
                self.viewer.timeslice_step_input
            ),
            self.viewer.timeslice_step_label,
        )
        self.assertIs(
            self.viewer.timeslice_context_layout.labelForField(
                self.viewer.timeslice_duration_input
            ),
            self.viewer.timeslice_duration_label,
        )
        self.assertEqual(self.viewer.ghost_stride_label.text(), "Playback spacing")
        self.assertEqual(self.viewer.ghost_alpha_label.text(), "Playback opacity")
        self.assertIs(
            self.viewer.timeslice_step_input.parent(),
            self.viewer.timeslice_context_panel,
        )
        self.assertIs(
            self.viewer.timeslice_duration_input.parent(),
            self.viewer.timeslice_context_panel,
        )
        self.assertIs(
            self.viewer.timeslice_action_row.itemAt(0).widget(),
            self.viewer.delete_timeslice_button,
        )
        self.assertEqual(self.viewer.timeslice_action_row.count(), 1)
        self.assertFalse(hasattr(self.viewer, "quick_clear_button"))
        toolbar_actions = self.window.app_toolbar.actions()
        self.assertLess(
            toolbar_actions.index(self.window.reset_action),
            toolbar_actions.index(self.window.clear_action),
        )
        for section in self.window.left_sidebar_content.sections:
            section.set_expanded(True)
            self.app.processEvents()
            self.assertGreater(
                section.maximumWidth(),
                LEFT_SIDEBAR_MIN_WIDTH,
            )
            section.set_expanded(False)
        self.assertIsNone(self.viewer.status_label.parent())
        self.assertIsNone(self.viewer.timeline_state_label.parent())
        self.assertIsNone(self.viewer.root_pose_label.parent())
        self.viewer.status_label.setText("Viewer status moved")
        self.viewer._update_timeline_label()
        self.viewer._update_root_pose_label()
        self.app.processEvents()
        self.assertEqual(
            self.window.viewer_status_label.text(),
            "Viewer status moved",
        )
        self.assertEqual(self.window.status_icon_label.text(), "●")
        self.assertEqual(
            self.window.status_icon_label.property("severity"),
            "info",
        )
        self.assertTrue(self.window.status_message_label.isHidden())
        self.assertIsNone(self.window.status_frame_label.parent())
        self.assertIsNone(self.window.status_ik_label.parent())
        self.assertIsNone(self.window.status_move_label.parent())
        self.assertIsNone(self.window.viewer_root_pose_label.parent())
        self.assertEqual(self.window.status_frame_label.text(), "-")
        self.assertEqual(self.window.status_ik_label.text(), "-")
        self.assertEqual(self.window.status_move_label.text(), "-")
        self.assertEqual(
            self.window.status_text.toPlainText(),
            "Viewer status moved",
        )
        verbose_status = (
            "axis X; MuJoCo IK converged in 4 iterations; "
            "state is collision-free; accepted=100%; IK error=0.0032; "
            "tasks=3; frame=left_hand; model=G1; preview not committed"
        )
        self.viewer.status_label.setText(verbose_status)
        self.app.processEvents()
        self.assertEqual(
            self.window.viewer_status_label.text(),
            "Preview updated",
        )
        self.assertEqual(
            self.window.status_message_label.text(),
            "Left hand pose is collision-free.",
        )
        self.assertEqual(self.window.status_frame_label.text(), "left hand")
        self.assertEqual(self.window.status_ik_label.text(), "0.0032 m")
        self.assertEqual(self.window.status_move_label.text(), "100%")
        self.assertIn(
            "Solver: MuJoCo IK converged in 4 iterations\n"
            "Collision: state is collision-free",
            self.window.status_text.toPlainText(),
        )
        self.assertIn(
            "Frame: left hand",
            self.window.status_text.toPlainText(),
        )
        self.assertTrue(self.window.viewer_time_label.text().startswith("Time:"))
        self.assertFalse(self.window.viewer_root_pose_label.text().startswith("Root:"))
        singular_status = verbose_status.replace(
            "model=G1",
            "model=G1; near singularity sigma_min=1.00e-10, cond=9.90e+09",
        )
        self.viewer.status_label.setText(singular_status)
        self.app.processEvents()
        self.assertEqual(self.window.viewer_status_label.text(), "Preview warning")
        self.assertEqual(
            self.window.status_icon_label.property("severity"),
            "warning",
        )
        self.assertEqual(
            self.window.status_message_label.text(),
            "Near singularity while updating left hand.",
        )
        self.assertIn("near singularity", self.window.status_text.toPlainText())
        self.assertFalse(self.window.status_details_panel.isVisible())
        self.window.status_details_button.setChecked(True)
        self.app.processEvents()
        self.assertTrue(self.window.status_details_panel.isVisible())
        self.assertEqual(
            self.viewer.target_context_panel.layout()
            .labelForField(self.viewer.target_box)
            .text(),
            "Advanced target",
        )
        self.assertIs(
            self.viewer.target_context_panel.parent(),
            self.window.controls.target_context_stack,
        )
        preview_section = next(
            section for section in self.window.right_sidebar_content.sections
            if section.title == "IK / Constraints"
        )
        preview_section.set_expanded(True)
        ik_tabs = self.viewer.preview_ik_context_widget().findChild(QTabWidget)
        self.assertEqual(
            [ik_tabs.tabText(index) for index in range(ik_tabs.count())],
            ["Tasks", "Weights", "Solver"],
        )
        self.assertEqual(ik_tabs.currentIndex(), 0)
        self.assertGreater(ik_tabs.maximumWidth(), SIDEBAR_MAX_WIDTH)
        self.assertEqual(ik_tabs.objectName(), "ikEditorTabs")
        self.assertGreater(
            self.window.controls.preview_ik_context_stack.maximumWidth(),
            SIDEBAR_MAX_WIDTH,
        )
        expected_group_titles = {
            0: "IK Tasks",
            1: "Joint Weights",
            2: "Solver",
        }
        for tab_index in range(ik_tabs.count()):
            ik_tabs.setCurrentIndex(tab_index)
            self.app.processEvents()
            visible_scroll_areas = [
                area for area in self.viewer.preview_ik_context_widget().findChildren(
                    QScrollArea
                )
                if area.isVisible()
            ]
            self.assertTrue(visible_scroll_areas)
            for area in visible_scroll_areas:
                self.assertEqual(area.objectName(), "ikEditorScroll")
                self.assertEqual(area.viewport().objectName(), "ikEditorViewport")
                self.assertEqual(area.widget().objectName(), "ikEditorTabContent")
                self.assertEqual(area.horizontalScrollBar().maximum(), 0)
                self.assertFalse(area.horizontalScrollBar().isVisible())
                self.assertLessEqual(area.widget().width(), area.viewport().width())
                self.assertGreater(area.maximumWidth(), SIDEBAR_MAX_WIDTH)
            expected_title = expected_group_titles.get(tab_index)
            if expected_title is not None:
                visible_group_titles = [
                    group.title()
                    for group in ik_tabs.currentWidget().findChildren(QGroupBox)
                    if group.isVisible()
                ]
                self.assertIn(expected_title, visible_group_titles)
        weight_tab = ik_tabs.widget(1)
        weight_group_titles = [
            group.title()
            for group in weight_tab.findChildren(QGroupBox)
        ]
        self.assertIn("Joint Weights", weight_group_titles)
        preview_section.set_expanded(False)
        all_titles = left_titles + right_titles
        for removed_title in (
            "Editors", "Display", "Selection", "Properties",
            "3D Selection", "Export / Import", "Playback / Ghosts",
            "Joints / IK", "Transform", "Preview / IK", "Target / Pose",
            "Time Slices", "Selected Object",
        ):
            self.assertNotIn(removed_title, all_titles)
        self.assertLessEqual(self.window.controls.table.maximumHeight(), 180)
        self.assertIs(
            self.window.controls.robot_context_widget(),
            self.viewer.robot_context_widget(),
        )
        self.assertIs(
            self.window.controls.target_context_widget(),
            self.viewer.target_context_widget(),
        )
        self.assertIs(
            self.window.controls.trajectory_context_widget(),
            self.window.controls.trajectory_context_stack.empty_widget,
        )
        self.assertIs(
            self.window.controls.timeslice_context_widget(),
            self.viewer.timeslice_context_widget(),
        )
        self.assertIs(
            self.window.controls.display_context_widget(),
            self.viewer.display_context_widget(),
        )
        self.assertIs(
            self.window.controls.preview_ik_context_widget(),
            self.viewer.preview_ik_context_widget(),
        )

        self.assertEqual(id(self.window.viewer_3d), viewer_identity)
        self.assertEqual(id(self.window.robot_model_3d.mj_model), model_identity)
        np.testing.assert_allclose(self.viewer.robot_state.get_qpos(), qpos)

    def test_sidebars_resize_collapse_and_restore_width(self):
        self.window.resize(1800, 800)
        self.window.show()
        self.app.processEvents()

        splitter = self.window.main_splitter
        splitter.setSizes([380, 1100, 270])
        self.app.processEvents()
        expanded_width = splitter.sizes()[0]
        self.assertGreaterEqual(expanded_width, 380)
        self.assertGreater(
            self.window.left_sidebar.maximumWidth(),
            expanded_width,
        )
        splitter.moveSplitter(600, 1)
        self.app.processEvents()
        self.assertEqual(splitter.sizes()[0], SIDEBAR_MAX_WIDTH)
        splitter.setSizes([380, 1100, 270])
        self.app.processEvents()

        available_width = (
            self.window.left_sidebar_content.body_layout.contentsRect().width()
        )
        for section in self.window.left_sidebar_content.sections:
            self.assertAlmostEqual(section.width(), available_width, delta=1)
        self.assertAlmostEqual(
            self.window.controls.editing_mode_bar.width(),
            self.window.controls.editing_mode_panel.width(),
            delta=1,
        )

        splitter.moveSplitter(20, 1)
        self.app.processEvents()
        self.assertGreaterEqual(
            splitter.sizes()[0],
            LEFT_SIDEBAR_MIN_WIDTH,
        )

        splitter.setSizes([380, 1100, 270])
        self.app.processEvents()
        expanded_width = splitter.sizes()[0]
        collapse_button = splitter.handle(1).toggle_button
        self.assertIsNotNone(collapse_button)
        self.assertTrue(collapse_button.isVisible())

        collapse_button.click()
        self.app.processEvents()
        self.assertEqual(splitter.sizes()[0], 0)
        self.assertEqual(self.window.left_sidebar.minimumWidth(), 0)
        self.assertFalse(self.window.left_sidebar_action.isChecked())
        self.assertEqual(
            collapse_button.arrowType(),
            Qt.ArrowType.RightArrow,
        )

        collapse_button.click()
        self.app.processEvents()
        self.assertGreaterEqual(splitter.sizes()[0], expanded_width)
        self.assertEqual(
            self.window.left_sidebar.minimumWidth(),
            LEFT_SIDEBAR_MIN_WIDTH,
        )
        self.assertTrue(self.window.left_sidebar_action.isChecked())
        self.assertEqual(
            collapse_button.arrowType(),
            Qt.ArrowType.LeftArrow,
        )

        splitter.setSizes([380, 1000, 390])
        splitter.moveSplitter(splitter.width() - 600, 2)
        self.app.processEvents()
        self.assertEqual(splitter.sizes()[2], SIDEBAR_MAX_WIDTH)
        right_expanded_width = splitter.sizes()[2]
        right_available_width = (
            self.window.right_sidebar_content.body_layout.contentsRect().width()
        )
        for section in self.window.right_sidebar_content.sections:
            self.assertAlmostEqual(
                section.width(),
                right_available_width,
                delta=1,
            )

        splitter.moveSplitter(splitter.width() - 20, 2)
        self.app.processEvents()
        self.assertGreaterEqual(
            splitter.sizes()[2],
            RIGHT_SIDEBAR_MIN_WIDTH,
        )

        splitter.setSizes([380, 1000, 390])
        self.app.processEvents()
        right_expanded_width = splitter.sizes()[2]
        right_collapse_button = splitter.handle(2).toggle_button
        self.assertTrue(right_collapse_button.isVisible())
        right_collapse_button.click()
        self.app.processEvents()
        self.assertEqual(splitter.sizes()[2], 0)
        self.assertFalse(self.window.right_sidebar_action.isChecked())
        self.assertEqual(
            right_collapse_button.arrowType(),
            Qt.ArrowType.LeftArrow,
        )
        right_collapse_button.click()
        self.app.processEvents()
        self.assertGreaterEqual(splitter.sizes()[2], right_expanded_width)
        self.assertTrue(self.window.right_sidebar_action.isChecked())
        self.assertEqual(
            right_collapse_button.arrowType(),
            Qt.ArrowType.RightArrow,
        )

        self.window.left_sidebar_action.setChecked(False)
        self.window.right_sidebar_action.setChecked(False)
        self.window.save_ui_settings()
        restored = RobotGuiMainWindow()
        try:
            self.assertFalse(restored.left_sidebar_action.isChecked())
            self.assertFalse(restored.right_sidebar_action.isChecked())
            self.assertEqual(restored.main_splitter.sizes()[0], 0)
            self.assertEqual(restored.main_splitter.sizes()[2], 0)
            restored.resize(1800, 800)
            restored.show()
            self.app.processEvents()
            restored.left_sidebar_action.setChecked(True)
            restored.right_sidebar_action.setChecked(True)
            self.app.processEvents()
            self.assertGreaterEqual(
                restored.main_splitter.sizes()[0],
                expanded_width,
            )
            self.assertGreaterEqual(
                restored.main_splitter.sizes()[2],
                right_expanded_width,
            )
        finally:
            restored.close()

    def test_project_save_and_open_restores_committed_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "reach_test.ghostgui"

            self.window.viewer_tabs.setCurrentIndex(1)
            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.12,
                y=0.08,
                z=1.05,
                roll=0.01,
                pitch=0.02,
                yaw=0.03,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()

            saved_qpos = self.viewer.committed_state.get_qpos()
            saved_qpos[-1] += 0.07
            self.viewer.set_robot_state_for_current_time(saved_qpos)
            self.viewer.update_current_keyframe_from_robot_state(refresh_ghosts=False)

            canvas = self.viewer.canvas
            canvas.camera_yaw = 12.5
            canvas.camera_pitch = 31.0
            canvas.camera_distance = 6.25
            canvas.camera_center = np.asarray([0.4, -0.2, 1.1], dtype=float)
            self.window.controls.show_lines_box.setChecked(False)
            self.window.controls.corner_smoothing_slider.set_value(0.35)

            project = self.window.create_project_at(project_root, "reach_test")
            self.assertTrue((project.root_dir / "ghostgui_project.json").exists())
            self.assertTrue((project.root_dir / "data" / "target_frames.json").exists())
            self.assertTrue((project.root_dir / "data" / "qpos_timeline.npz").exists())

            self.window.trajectory.clear()
            self.window.controls.frame_box.setCurrentText("pelvis")
            self.window.controls.show_lines_box.setChecked(True)
            self.window.controls.corner_smoothing_slider.set_value(0.0)
            self.window.viewer_tabs.setCurrentIndex(0)
            self.viewer.clear_editable_timeline(
                keep_current_pose=False,
                reset_time=0.0,
            )
            canvas.camera_yaw = 90.0
            canvas.camera_pitch = 5.0
            canvas.camera_distance = 2.0
            canvas.camera_center = np.asarray([0.0, 0.0, 0.0], dtype=float)

            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ) as guard:
                self.assertTrue(self.window.open_project_path(project_root))

            guard.assert_called_once()

            self.assertEqual(self.window.current_project.project_name, "reach_test")
            self.assertEqual(len(self.window.trajectory.frames), 1)
            frame = self.window.trajectory.frames[0]
            self.assertEqual(frame.frame_name, "left_hand")
            self.assertAlmostEqual(frame.time, 0.2)
            self.assertAlmostEqual(frame.x, 0.12)
            self.assertAlmostEqual(frame.y, 0.08)
            self.assertAlmostEqual(frame.z, 1.05)
            self.assertAlmostEqual(self.viewer.get_current_time(), 0.2)
            np.testing.assert_allclose(
                self.viewer.state_timeline.get_state(0.2),
                saved_qpos,
            )
            np.testing.assert_allclose(
                self.viewer.committed_state.get_qpos(),
                saved_qpos,
            )
            self.assertEqual(self.window.viewer_tabs.currentIndex(), 1)
            self.assertFalse(self.window.controls.show_lines_box.isChecked())
            self.assertAlmostEqual(self.window.controls.corner_smoothing(), 0.35)
            self.assertAlmostEqual(self.viewer.canvas.camera_yaw, 12.5)
            self.assertAlmostEqual(self.viewer.canvas.camera_pitch, 31.0)
            self.assertAlmostEqual(self.viewer.canvas.camera_distance, 6.25)
            np.testing.assert_allclose(
                self.viewer.canvas.camera_center,
                np.asarray([0.4, -0.2, 1.1], dtype=float),
            )
            self.window.current_project = None

    def test_legacy_removed_view_indices_restore_to_supported_views(self):
        self.window.restore_project_workspace({"active_view_index": 3})
        self.assertEqual(
            self.window.viewer_tabs.tabText(
                self.window.viewer_tabs.currentIndex()
            ),
            "Simulation",
        )

        self.window.restore_project_workspace({"active_view_index": 2})
        self.assertEqual(
            self.window.viewer_tabs.tabText(
                self.window.viewer_tabs.currentIndex()
            ),
            "3D Pose",
        )

    def test_project_create_updates_recent_projects_menu(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "recent_one.ghostgui"

            project = self.window.create_project_at(project_root, "recent_one")

            entries = load_recent_projects()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["project_name"], "recent_one")
            self.assertEqual(Path(entries[0]["path"]), project.root_dir)

            self.window.refresh_recent_projects()
            recent_actions = [
                action
                for action in self.window.recent_projects_menu.actions()
                if action.isEnabled()
            ]
            self.assertEqual(len(recent_actions), 1)
            self.assertEqual(recent_actions[0].data(), str(project.root_dir))
            self.assertEqual(recent_actions[0].text(), "recent_one")

            self.window.current_project = None

    def test_new_project_uses_default_root_and_resets_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            old_project_root = Path(directory) / "old_project.ghostgui"

            self.window.create_project_at(old_project_root, "old_project")
            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.28,
                y=0.11,
                z=1.14,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            self.viewer.canvas.camera_yaw = 91.0
            self.viewer.canvas.camera_pitch = 12.0
            self.assertTrue(self.window.project_dirty)

            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ) as guard, patch(
                "gui.main_window.QInputDialog.getText",
                return_value=("fresh project", True),
            ):
                self.window.on_new_project()

            guard.assert_called_once_with("create a new project")

            projects_root = Path(os.environ["GHOSTGUI_PROJECTS_DIR"]).resolve()
            self.assertEqual(self.window.current_project.root_dir.parent, projects_root)
            self.assertEqual(
                self.window.current_project.root_dir.name,
                "fresh_project.ghostgui",
            )
            self.assertTrue(self.window.current_project.project_file.exists())
            self.assertFalse(self.window.project_dirty)
            self.assertEqual(len(self.window.trajectory.frames), 0)
            self.assertEqual(self.window.active_index, -1)
            self.assertAlmostEqual(self.viewer.get_current_time(), 0.0)
            self.assertEqual(self.viewer.robot_trajectory, [])
            self.assertEqual(self.viewer.robot_trajectory_times, [])
            self.assertAlmostEqual(self.viewer.timeline_duration, 5.0)
            self.assertAlmostEqual(self.viewer.canvas.camera_yaw, 38.0)
            self.assertAlmostEqual(self.viewer.canvas.camera_pitch, 24.0)
            self.assertAlmostEqual(self.viewer.canvas.camera_distance, 5.0)
            np.testing.assert_allclose(
                self.viewer.canvas.camera_center,
                np.asarray([0.0, 0.0, 0.75], dtype=float),
            )

            saved_project = GhostGUIProject.open(self.window.current_project.root_dir)
            saved_frames = saved_project.read_trajectory_dict()["tracks"]
            self.assertTrue(all(len(frames) == 0 for frames in saved_frames.values()))

            self.window.current_project = None

    def test_new_project_cancelled_dirty_guard_keeps_current_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "guarded.ghostgui"

            project = self.window.create_project_at(project_root, "guarded")
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.on_add_keyframe()

            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=False,
            ) as guard, patch(
                "gui.main_window.QInputDialog.getText"
            ) as get_text:
                self.window.on_new_project()

            guard.assert_called_once_with("create a new project")
            get_text.assert_not_called()
            self.assertEqual(self.window.current_project.root_dir, project.root_dir)
            self.assertTrue(self.window.project_dirty)
            self.assertEqual(len(self.window.trajectory.frames), 1)

            self.window.current_project = None

    def test_default_project_root_is_repo_projects_folder(self):
        os.environ.pop("GHOSTGUI_PROJECTS_DIR", None)

        expected_root = Path(__file__).resolve().parents[1] / "projects"
        self.assertEqual(ghostgui_projects_dir(), expected_root)

    def test_save_without_project_uses_default_root_without_resetting_workspace(self):
        self.window.controls.frame_box.setCurrentText("left_hand")
        self.window.controls.set_position_values(
            x=0.19,
            y=0.07,
            z=1.02,
            emit_pose_changed=False,
        )
        self.window.on_add_keyframe()

        with patch(
            "gui.main_window.QInputDialog.getText",
            return_value=("saved workspace", True),
        ):
            self.window.on_save_project()

        projects_root = Path(os.environ["GHOSTGUI_PROJECTS_DIR"]).resolve()
        self.assertEqual(self.window.current_project.root_dir.parent, projects_root)
        self.assertEqual(len(self.window.trajectory.frames), 1)
        frame = self.window.trajectory.frames[0]
        self.assertAlmostEqual(frame.x, 0.19)
        self.assertAlmostEqual(frame.y, 0.07)
        self.assertAlmostEqual(frame.z, 1.02)
        self.assertFalse(self.window.project_dirty)

        self.window.current_project = None

    def test_project_dirty_state_tracks_edits_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "dirty_test.ghostgui"

            project = self.window.create_project_at(project_root, "dirty_test")
            self.assertFalse(self.window.project_dirty)
            self.assertFalse(self.window.windowTitle().endswith("*"))

            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.18,
                y=0.05,
                z=1.04,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()

            self.assertTrue(self.window.project_dirty)
            self.assertTrue(self.window.windowTitle().endswith("*"))

            self.assertTrue(
                self.window.save_current_project(
                    capture_snapshot=False,
                    reason="unit_save",
                )
            )

            self.assertFalse(self.window.project_dirty)
            self.assertFalse(self.window.windowTitle().endswith("*"))
            events = project.read_session_log()
            self.assertEqual(events[-1]["event"], "project_saved")
            self.assertEqual(events[-1]["details"]["reason"], "unit_save")

            self.window.current_project = None

    def test_project_transition_choices_save_autosave_discard_or_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "guard_test.ghostgui"
            project = self.window.create_project_at(project_root, "guard_test")

            self.window.mark_project_dirty("unit edit")
            self.assertFalse(
                self.window.handle_project_transition_choice(
                    "cancel",
                    "open another project",
                )
            )
            self.assertTrue(self.window.project_dirty)
            self.assertEqual(
                project.read_session_log()[-1]["event"],
                "project_transition_cancelled",
            )

            self.assertTrue(
                self.window.handle_project_transition_choice(
                    "autosave",
                    "open another project",
                )
            )
            self.assertFalse(self.window.project_dirty)
            self.assertTrue(project.autosave_exists())
            events = project.read_session_log()
            self.assertEqual(events[-1]["event"], "project_autosaved")
            self.assertEqual(
                events[-1]["details"]["reason"],
                "before_open_another_project",
            )

            self.window.mark_project_dirty("unit edit")
            self.assertTrue(
                self.window.handle_project_transition_choice(
                    "discard",
                    "open another project",
                )
            )
            self.assertFalse(self.window.project_dirty)
            self.assertFalse(project.autosave_exists())
            self.assertEqual(
                project.read_session_log()[-1]["event"],
                "project_dirty_discarded",
            )

            self.window.mark_project_dirty("unit edit")
            self.assertTrue(
                self.window.handle_project_transition_choice(
                    "save",
                    "open another project",
                )
            )
            self.assertFalse(self.window.project_dirty)
            events = project.read_session_log()
            self.assertEqual(events[-1]["event"], "project_saved")
            self.assertEqual(
                events[-1]["details"]["reason"],
                "before_open_another_project",
            )

            self.window.current_project = None

    def test_project_open_cancel_guard_blocks_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            first_root = Path(directory) / "first.ghostgui"
            second_root = Path(directory) / "second.ghostgui"

            first = self.window.create_project_at(first_root, "first")
            second = GhostGUIProject.create(
                second_root,
                "second",
                self.window.model_key,
                self.window.current_model_display_name(),
            )
            second.write_trajectory(self.window.trajectory)
            second.save_qpos_timeline(self.viewer.state_timeline)
            second.write_workspace(self.window.capture_project_workspace())
            second.save_metadata()

            self.window.mark_project_dirty("unit edit")
            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=False,
            ) as guard:
                self.assertFalse(
                    self.window.open_project_path(second_root, source="unit_test")
                )

            guard.assert_called_once()
            self.assertEqual(self.window.current_project.root_dir, first.root_dir)
            self.assertTrue(self.window.project_dirty)

            self.window.current_project = None

    def test_project_close_cancel_guard_ignores_close_event(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "close_guard.ghostgui"

            self.window.create_project_at(project_root, "close_guard")
            self.window.mark_project_dirty("unit edit")
            event = QCloseEvent()

            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=False,
            ) as guard:
                self.window.closeEvent(event)

            guard.assert_called_once_with("close GhostGUI")
            self.assertFalse(event.isAccepted())
            self.assertTrue(self.window.project_dirty)

            self.window.current_project = None

    def test_project_session_log_records_lifecycle_events(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "history_test.ghostgui"

            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.41,
                y=0.12,
                z=1.16,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            project = self.window.create_project_at(project_root, "history_test")

            self.assertTrue(project.paths.session_log.exists())
            events = project.read_session_log()
            self.assertEqual(
                [event["event"] for event in events],
                ["project_created", "project_saved"],
            )
            self.assertEqual(
                events[-1]["details"]["reason"],
                "initial_create",
            )
            self.assertEqual(events[-1]["details"]["frame_count"], 1)

            self.assertTrue(
                self.window.save_current_project(
                    capture_snapshot=False,
                    reason="manual_test",
                )
            )
            QTest.qWait(20)
            self.assertTrue(
                self.window.autosave_current_project(
                    capture_snapshot=False,
                    reason="unit_test",
                )
            )

            self.window.trajectory.clear()
            with patch(
                "gui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                self.assertTrue(
                    self.window.open_project_path(
                        project_root,
                        source="unit_test",
                    )
                )

            events = self.window.current_project.read_session_log()
            names = [event["event"] for event in events]
            self.assertEqual(
                names,
                [
                    "project_created",
                    "project_saved",
                    "project_saved",
                    "project_autosaved",
                    "project_opened",
                ],
            )
            self.assertEqual(events[2]["details"]["reason"], "manual_test")
            self.assertEqual(events[3]["details"]["reason"], "unit_test")
            self.assertEqual(events[-1]["details"]["source"], "unit_test")
            self.assertTrue(events[-1]["details"]["autosave"])
            self.assertEqual(events[-1]["details"]["frame_count"], 1)

            self.window.current_project = None

    def test_recent_project_selection_reopens_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "recent_reopen.ghostgui"

            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.31,
                y=0.09,
                z=1.18,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            project = self.window.create_project_at(project_root, "recent_reopen")

            self.window.trajectory.clear()
            self.window.controls.frame_box.setCurrentText("pelvis")

            self.window.refresh_recent_projects()
            recent_action = next(
                action
                for action in self.window.recent_projects_menu.actions()
                if action.data() == str(project.root_dir)
            )
            recent_action.trigger()

            self.assertEqual(self.window.current_project.root_dir, project.root_dir)
            self.assertEqual(len(self.window.trajectory.frames), 1)
            frame = self.window.trajectory.frames[0]
            self.assertEqual(frame.frame_name, "left_hand")
            self.assertAlmostEqual(frame.time, 0.2)
            self.assertAlmostEqual(frame.x, 0.31)
            self.assertAlmostEqual(frame.y, 0.09)
            self.assertAlmostEqual(frame.z, 1.18)

            self.window.current_project = None

    def test_project_browser_shows_snapshot_preview_and_opens_project(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "preview_reopen.ghostgui"

            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.22,
                y=0.13,
                z=1.08,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            project = self.window.create_project_at(project_root, "preview_reopen")

            snapshot = QPixmap(96, 54)
            snapshot.fill(Qt.GlobalColor.red)
            self.assertTrue(snapshot.save(str(project.paths.last_snapshot)))

            dialog = self.window.build_project_browser_dialog()
            self.assertEqual(dialog.project_list.count(), 1)
            item = dialog.project_list.item(0)
            self.assertEqual(item.data(Qt.ItemDataRole.UserRole), str(project.root_dir))
            self.assertIn("preview_reopen", item.text())
            self.assertFalse(item.icon().isNull())
            self.assertTrue(dialog.open_button.isEnabled())

            self.window.trajectory.clear()
            with patch.object(dialog, "exec", return_value=True):
                dialog.selected_project_path = str(project.root_dir)
                dialog.browse_requested = False
                with patch.object(
                    self.window,
                    "build_project_browser_dialog",
                    return_value=dialog,
                ):
                    self.window.on_open_project()

            self.assertEqual(self.window.current_project.root_dir, project.root_dir)
            self.assertEqual(len(self.window.trajectory.frames), 1)
            frame = self.window.trajectory.frames[0]
            self.assertEqual(frame.frame_name, "left_hand")
            self.assertAlmostEqual(frame.x, 0.22)
            self.assertAlmostEqual(frame.y, 0.13)
            self.assertAlmostEqual(frame.z, 1.08)

            dialog.close()
            self.window.current_project = None

    def test_project_open_can_restore_newer_autosave(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "autosave_test.ghostgui"

            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.12,
                y=0.08,
                z=1.05,
                roll=0.01,
                pitch=0.02,
                yaw=0.03,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            project = self.window.create_project_at(project_root, "autosave_test")

            QTest.qWait(20)
            self.window.trajectory.clear()
            self.window.controls.set_position_values(
                x=0.44,
                y=0.18,
                z=1.22,
                roll=0.04,
                pitch=0.05,
                yaw=0.06,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            autosaved_qpos = self.viewer.committed_state.get_qpos()
            autosaved_qpos[-1] += 0.11
            self.viewer.set_robot_state_for_current_time(autosaved_qpos)
            self.viewer.update_current_keyframe_from_robot_state(refresh_ghosts=False)
            self.viewer.canvas.camera_yaw = 17.0
            self.assertTrue(
                self.window.autosave_current_project(capture_snapshot=False)
            )
            self.assertTrue(project.autosave_exists())
            self.assertTrue(project.is_autosave_newer())

            self.window.trajectory.clear()
            self.viewer.clear_editable_timeline(
                keep_current_pose=False,
                reset_time=0.0,
            )
            self.viewer.canvas.camera_yaw = 90.0

            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ) as guard, patch(
                "gui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ) as question:
                self.assertTrue(self.window.open_project_path(project_root))

            guard.assert_called_once()
            question.assert_called_once()
            self.assertEqual(len(self.window.trajectory.frames), 1)
            frame = self.window.trajectory.frames[0]
            self.assertAlmostEqual(frame.x, 0.44)
            self.assertAlmostEqual(frame.y, 0.18)
            self.assertAlmostEqual(frame.z, 1.22)
            np.testing.assert_allclose(
                self.viewer.state_timeline.get_state(0.2),
                autosaved_qpos,
            )
            self.assertAlmostEqual(self.viewer.canvas.camera_yaw, 17.0)
            self.assertIn("autosaved project", self.window.status_text.toPlainText())
            self.window.current_project = None

    def test_project_open_can_discard_autosave_and_use_last_save(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "discard_autosave.ghostgui"

            self.window.on_viewer_timeslice_time_changed(0.2)
            self.window.controls.frame_box.setCurrentText("left_hand")
            self.window.controls.set_position_values(
                x=0.12,
                y=0.08,
                z=1.05,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            project = self.window.create_project_at(project_root, "discard_autosave")

            QTest.qWait(20)
            self.window.trajectory.clear()
            self.window.controls.set_position_values(
                x=0.77,
                y=0.22,
                z=1.33,
                emit_pose_changed=False,
            )
            self.window.on_add_keyframe()
            self.assertTrue(
                self.window.autosave_current_project(capture_snapshot=False)
            )
            self.assertTrue(project.is_autosave_newer())

            self.window.trajectory.clear()
            with patch.object(
                self.window,
                "confirm_project_transition",
                return_value=True,
            ) as guard, patch(
                "gui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ) as question:
                self.assertTrue(self.window.open_project_path(project_root))

            guard.assert_called_once()
            question.assert_called_once()
            self.assertFalse(project.autosave_exists())
            self.assertEqual(len(self.window.trajectory.frames), 1)
            frame = self.window.trajectory.frames[0]
            self.assertAlmostEqual(frame.x, 0.12)
            self.assertAlmostEqual(frame.y, 0.08)
            self.assertAlmostEqual(frame.z, 1.05)
            self.assertNotIn("autosaved project", self.window.status_text.toPlainText())
            events = self.window.current_project.read_session_log()
            names = [event["event"] for event in events]
            self.assertIn("project_autosave_discarded", names)
            self.assertEqual(names[-1], "project_opened")
            self.window.current_project = None

    def test_help_center_opens_without_written_guide_button(self):
        self.window.help_center_action.trigger()
        self.app.processEvents()

        self.assertIsNotNone(self.window.help_dialog)
        self.assertTrue(self.window.help_dialog.isVisible())
        self.assertIn(
            "First Motion Walkthrough",
            self.window.help_dialog.browser.toPlainText(),
        )
        section_titles = [
            self.window.help_dialog.section_list.item(index).text()
            for index in range(self.window.help_dialog.section_list.count())
        ]
        self.assertIn("Keyboard / Mouse Shortcuts", section_titles)
        self.window.help_dialog.section_list.setCurrentRow(
            section_titles.index("Keyboard / Mouse Shortcuts")
        )
        self.assertIn("T switches", self.window.help_dialog.browser.toPlainText())
        self.assertIn("Ctrl+Shift+Z", self.window.help_dialog.browser.toPlainText())
        self.assertIsNone(
            self.window.help_dialog.findChild(QPushButton, "writtenGuideButton")
        )

    def test_help_center_starts_guided_tutorial_overlay(self):
        self.window.resize(1700, 800)
        self.window.show()
        self.app.processEvents()

        self.window.help_center_action.trigger()
        self.app.processEvents()

        start_button = self.window.help_dialog.findChild(
            QPushButton, "startTutorialButton"
        )
        self.assertIsNotNone(start_button)
        start_button.click()
        self.app.processEvents()

        manager = self.window.tutorial_manager
        self.assertTrue(manager.active)
        self.assertFalse(self.window.help_dialog.isVisible())
        self.assertTrue(manager.overlay.isVisible())
        self.assertTrue(manager.card.isVisible())
        self.assertIn("First Motion", manager.card.title_label.text())

        next_button = manager.card.findChild(QPushButton, "tutorialNextButton")
        back_button = manager.card.findChild(QPushButton, "tutorialBackButton")
        skip_button = manager.card.findChild(QPushButton, "tutorialSkipButton")
        self.assertIsNotNone(next_button)
        self.assertIsNotNone(back_button)
        self.assertIsNotNone(skip_button)
        self.assertFalse(back_button.isEnabled())

        next_button.click()
        self.app.processEvents()

        self.assertEqual(manager.steps[manager.current_index].id, "choose_model")
        self.assertFalse(manager.overlay.target_rect.isNull())
        self.assertIn("Choose A Robot", manager.card.title_label.text())
        self.assertTrue(self.window.app_toolbar.isVisible())
        self.assertTrue(self.window.menuBar().isVisible())
        self.assertIn(self.window.robot_menu.menuAction(), self.window.menuBar().actions())
        self.assertTrue(back_button.isEnabled())

        back_button.click()
        self.app.processEvents()
        self.assertEqual(manager.current_index, 0)

        skip_button.click()
        self.app.processEvents()
        self.assertFalse(manager.active)
        self.assertFalse(manager.overlay.isVisible())
        self.assertFalse(manager.card.isVisible())

    def test_model_colors_toggle_defaults_on_without_mutating_materials(self):
        before = self.window.robot_model_3d.mj_model.mat_rgba.copy()
        self.window.viewer_tabs.setCurrentWidget(self.window.viewer_3d_stack)
        self.window.update_editor_context()
        self.assertTrue(self.viewer.model_colors_box.isChecked())
        self.assertTrue(self.viewer.canvas.use_model_colors)
        display_layout = self.viewer.display_context_widget().layout()
        display_widgets = [
            display_layout.itemAt(index).widget()
            for index in range(display_layout.count())
        ]
        self.assertEqual(display_widgets[:4], [
            self.viewer.model_colors_box,
            self.window.controls.show_keyframes_box,
            self.window.controls.show_lines_box,
            self.viewer.show_ghosts,
        ])
        self.viewer.model_colors_box.setChecked(False)
        self.assertFalse(self.viewer.canvas.use_model_colors)
        self.viewer.model_colors_box.setChecked(True)
        np.testing.assert_allclose(
            self.window.robot_model_3d.mj_model.mat_rgba, before
        )

    def test_controls_store_full_rpy_in_keyframe_and_table(self):
        self.window.controls.set_position_values(
            roll=0.12, pitch=-0.23, yaw=0.34,
            emit_pose_changed=False,
        )
        self.window.on_add_keyframe()
        frame = self.window.trajectory.frames[0]
        self.assertAlmostEqual(frame.roll, 0.12)
        self.assertAlmostEqual(frame.pitch, -0.23)
        self.assertAlmostEqual(frame.yaw, 0.34)
        headers = [
            self.window.controls.table.horizontalHeaderItem(index).text()
            for index in range(self.window.controls.table.columnCount())
        ]
        self.assertEqual(headers[-3:], ["roll", "pitch", "yaw"])

    def test_pose_controls_use_compact_static_labels(self):
        controls = self.window.controls
        expected = {
            controls.x_slider: "X [m]",
            controls.y_slider: "Y [m]",
            controls.z_slider: "Z [m]",
            controls.roll_slider: "Roll [°]",
            controls.pitch_slider: "Pitch [°]",
            controls.yaw_slider: "Yaw [°]",
        }

        for slider, label in expected.items():
            with self.subTest(label=label):
                self.assertEqual(slider.label.text(), label)
                self.assertNotIn(":", slider.label.text())
                self.assertLessEqual(slider.label.maximumWidth(), 86)
                self.assertFalse(slider.slider.editor.isVisible())
                self.assertGreaterEqual(slider.slider.minimumHeight(), 24)

    def test_corner_smoothing_slider_uses_fraction_scale(self):
        controls = self.window.controls

        self.assertEqual(controls.corner_smoothing_slider.label.text(), "Smoothing")
        self.assertGreaterEqual(
            controls.corner_smoothing_slider.label.minimumWidth(),
            controls.corner_smoothing_slider.label.sizeHint().width(),
        )
        self.assertAlmostEqual(
            controls.corner_smoothing_slider.slider.logical_maximum(), 1.0
        )
        self.assertAlmostEqual(
            controls.corner_smoothing_slider.slider.logical_single_step(), 0.01
        )
        self.assertEqual(
            controls.corner_smoothing_slider.slider.format_value(), "0%"
        )
        self.assertEqual(
            controls.corner_smoothing_slider.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding,
        )
        self.assertAlmostEqual(controls.corner_smoothing(), 0.0)

        controls.corner_smoothing_slider.set_value(0.5)

        self.assertAlmostEqual(controls.corner_smoothing(), 0.5)

    def test_refresh_display_passes_trajectory_display_options_to_3d_view(self):
        captured = {}

        def capture(name):
            def update_scene(*args, **kwargs):
                captured[name] = {
                    "smoothing": kwargs["trajectory_smoothing"],
                    "show_lines": kwargs["show_trajectory_lines"],
                    "show_keyframes": kwargs["show_keyframes"],
                }
            return update_scene

        self.window.viewer_3d.update_scene = capture("viewer_3d")
        self.window.controls.corner_smoothing_slider.set_value(0.75)
        self.window.controls.show_keyframes_box.setChecked(False)
        self.window.controls.show_lines_box.setChecked(False)

        self.window.refresh_display()

        self.assertEqual(captured, {
            "viewer_3d": {
                "smoothing": 0.75,
                "show_lines": False,
                "show_keyframes": False,
            },
        })

    def test_accepted_3d_rotation_is_upserted_into_keyframe(self):
        self.window.on_3d_target_frame_changed("left_hand")
        self.viewer._set_target_to_selected_pose()
        position = self.viewer.last_valid_target_position.copy()
        start_rpy = quat_to_rpy(self.viewer.last_valid_target_quaternion)
        target_rpy = (
            start_rpy[0] + 0.05,
            start_rpy[1] - 0.03,
            start_rpy[2] + 0.04,
        )
        self.viewer._on_transform_moved(position, rpy_to_quat(*target_rpy))
        self.assertTrue(self.viewer.preview_active)
        self.viewer.accept_preview()

        frame = self.window.trajectory.tracks["left_hand"][0]
        solved_rpy = quat_to_rpy(self.viewer.last_valid_target_quaternion)
        self.assertAlmostEqual(frame.roll, solved_rpy[0], delta=0.011)
        self.assertAlmostEqual(frame.pitch, solved_rpy[1], delta=0.011)
        self.assertAlmostEqual(frame.yaw, solved_rpy[2], delta=0.011)

    def test_load_edit_accept_and_save_headerless_qpos_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            expected = self.viewer.robot_model.home_qpos.copy()
            np.savetxt(source, expected[None, :], delimiter=",")
            self.viewer.load_qpos_csv(source)
            np.testing.assert_allclose(
                self.viewer.committed_state.get_qpos(), expected
            )
            np.testing.assert_allclose(self.viewer.get_current_keyframe(), expected)

            joint_name = self.viewer.preview_state.get_joint_names()[-1]
            self.viewer._joint_changed(
                joint_name,
                self.viewer.preview_state.get_joint_value(joint_name) + 0.02,
            )
            self.viewer.accept_preview()

            output = self.viewer.save_qpos_csv(Path(directory) / "updated")
            self.assertEqual(output.suffix, ".csv")
            saved = np.loadtxt(output, delimiter=",")
        np.testing.assert_allclose(saved, self.viewer.committed_state.get_qpos())
        self.assertFalse(np.allclose(saved, expected))

    def test_missing_playback_cache_is_rebuilt_from_loaded_trajectory(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05
        self.viewer.set_robot_trajectory(
            [first, second],
            times=[0.0, 0.1],
        )

        cache_path = mujoco_playback_cache_path()
        panel = self.window.viewer_3d_mujoco
        panel.set_trajectory_metadata(cache_path, [0.0, 0.1])
        self.assertFalse(cache_path.exists())
        self.assertFalse(panel.play_button.isEnabled())

        self.assertTrue(panel._ensure_trajectory_file())
        self.assertTrue(cache_path.exists())
        self.assertTrue(panel.play_button.isEnabled())
        _, rows = load_trajectory_csv(
            cache_path,
            qpos_width=self.viewer.robot_model.mj_model.nq,
        )
        np.testing.assert_allclose(rows[0][RAW_QPOS_KEY], first)
        np.testing.assert_allclose(rows[1][RAW_QPOS_KEY], second)

        cache_path.unlink()
        self.assertTrue(panel._ensure_trajectory_file())
        self.assertTrue(cache_path.exists())

    def test_playback_is_disabled_without_cache_or_solved_trajectory(self):
        panel = self.window.viewer_3d_mujoco
        panel.clear_trajectory()
        self.viewer.clear_robot_trajectory()
        self.window.backend_interface.clear_last_solution()

        self.assertFalse(panel._ensure_trajectory_file())
        self.assertFalse(panel.play_button.isEnabled())
        self.assertIn(
            "Generate or import a trajectory first",
            panel.log_box.toPlainText(),
        )

    def test_save_timeline_as_headerless_time_plus_qpos_csv(self):
        at_zero = self.viewer.get_current_keyframe()
        self.viewer.set_current_time(0.2)
        changed = self.viewer.committed_state.get_qpos()
        changed[-1] += 0.05
        self.viewer.set_robot_state_for_current_time(changed)
        self.viewer.update_current_keyframe_from_robot_state()

        with tempfile.TemporaryDirectory() as directory:
            output = self.viewer.save_trajectory_csv(
                Path(directory) / "timed_trajectory"
            )
            saved = np.loadtxt(output, delimiter=",")
            _, rows = load_trajectory_csv(
                output, qpos_width=self.viewer.robot_model.mj_model.nq
            )

        self.assertEqual(output.suffix, ".csv")
        self.assertEqual(saved.shape, (2, self.viewer.robot_model.mj_model.nq + 1))
        np.testing.assert_allclose(saved[:, 0], [0.0, 0.2])
        np.testing.assert_allclose(rows[0][RAW_QPOS_KEY], at_zero)
        np.testing.assert_allclose(rows[1][RAW_QPOS_KEY], changed)

    def test_save_generated_trajectory_uses_backend_times(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.05

        self.viewer.load_backend_states([
            PythonRobotConfiguration(time=0.0, qpos=first),
            PythonRobotConfiguration(time=0.1, qpos=second),
        ])

        with tempfile.TemporaryDirectory() as directory:
            output = self.viewer.save_trajectory_csv(
                Path(directory) / "generated_trajectory"
            )
            saved = np.loadtxt(output, delimiter=",")
            _, rows = load_trajectory_csv(
                output, qpos_width=self.viewer.robot_model.mj_model.nq
            )

        self.assertEqual(saved.shape, (2, self.viewer.robot_model.mj_model.nq + 1))
        np.testing.assert_allclose(saved[:, 0], [0.0, 0.1])
        np.testing.assert_allclose(rows[0][RAW_QPOS_KEY], first)
        np.testing.assert_allclose(rows[1][RAW_QPOS_KEY], second)

    def test_load_headerless_time_plus_qpos_trajectory_csv(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.07

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.csv"
            np.savetxt(
                source,
                np.vstack(
                    [
                        np.concatenate(([0.0], first)),
                        np.concatenate(([6.25], second)),
                    ]
                ),
                delimiter=",",
            )
            self.viewer.load_trajectory_csv(source)
            mujoco_panel_path = self.window.viewer_3d_mujoco.trajectory_csv_path
            mujoco_panel_times = list(self.window.viewer_3d_mujoco.trajectory_times)

        self.assertEqual(len(self.viewer.robot_trajectory), 2)
        np.testing.assert_allclose(self.viewer.robot_trajectory_times, [0.0, 6.25])
        np.testing.assert_allclose(self.viewer.robot_trajectory[0], first)
        np.testing.assert_allclose(self.viewer.robot_trajectory[1], second)
        np.testing.assert_allclose(self.viewer.committed_state.get_qpos(), first)
        np.testing.assert_allclose(self.viewer.state_timeline.times(), [0.0, 6.25])
        self.assertEqual(
            len(self.window.trajectory.frames),
            2 * len(self.window.editable_logical_frame_names()),
        )
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.0, 6.25})
        self.assertAlmostEqual(self.viewer.timeline_duration, 6.25)
        self.assertEqual(self.viewer.timeslice_slider.maximum(), 625)
        self.assertEqual(self.window.controls.time_slider.slider.maximum(), 625)
        self.assertEqual(mujoco_panel_path, source)
        np.testing.assert_allclose(mujoco_panel_times, [0.0, 6.25])

    def test_loaded_trajectory_csv_import_interval_downsamples_editable_keyframes(self):
        base = self.viewer.robot_model.home_qpos.copy()
        rows = []
        for index in range(11):
            qpos = base.copy()
            qpos[-1] += index * 0.01
            rows.append(np.concatenate(([index * 0.01], qpos)))

        self.viewer.trajectory_import_dt.setValue(0.05)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dense_trajectory.csv"
            np.savetxt(source, np.vstack(rows), delimiter=",")
            self.viewer.load_trajectory_csv(source)

        editable_times = sorted({frame.time for frame in self.window.trajectory.frames})
        self.assertEqual(len(self.viewer.robot_trajectory), 11)
        np.testing.assert_allclose(self.viewer.robot_trajectory_times, np.arange(0, 0.11, 0.01))
        np.testing.assert_allclose(editable_times, [0.0, 0.05, 0.10])
        self.assertEqual(
            len(self.window.trajectory.frames),
            3 * len(self.window.editable_logical_frame_names()),
        )
        self.assertEqual(self.viewer.timeslice_slider.defined_times, {0.0, 0.05, 0.10})

    def test_setup_trajectory_import_prompts_for_time_step(self):
        base = self.viewer.robot_model.home_qpos.copy()
        rows = []
        for index in range(11):
            qpos = base.copy()
            qpos[-1] += index * 0.01
            rows.append(np.concatenate(([index * 0.01], qpos)))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dense_trajectory.csv"
            np.savetxt(source, np.vstack(rows), delimiter=",")
            with (
                patch.object(
                    self.viewer, "_open_csv_file_dialog"
                ) as open_dialog,
                patch(
                    "gui.main_window.QInputDialog.getDouble",
                    return_value=(0.05, True),
                ) as prompt,
            ):
                self.window.on_setup_import_requested("trajectory")
                open_dialog.call_args.kwargs["selected"](str(source))
                for _attempt in range(500):
                    if not self.window.background_jobs.is_busy():
                        break
                    QTest.qWait(10)
                self.assertFalse(self.window.background_jobs.is_busy())

        prompt.assert_called_once()
        self.assertAlmostEqual(self.viewer.trajectory_import_dt.value(), 0.05)
        editable_times = sorted({frame.time for frame in self.window.trajectory.frames})
        np.testing.assert_allclose(editable_times, [0.0, 0.05, 0.10])

    def test_loaded_trajectory_csv_can_be_cleared_from_keyframe_controls(self):
        first = self.viewer.robot_model.home_qpos.copy()
        second = first.copy()
        second[-1] += 0.07

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.csv"
            np.savetxt(
                source,
                np.vstack(
                    [
                        np.concatenate(([0.0], first)),
                        np.concatenate(([0.15], second)),
                    ]
                ),
                delimiter=",",
            )
            self.viewer.load_trajectory_csv(source)

        self.assertTrue(self.window.trajectory.frames)
        self.assertTrue(self.viewer.robot_trajectory)

        original_question = QMessageBox.question
        try:
            QMessageBox.question = staticmethod(
                lambda *args, **kwargs: QMessageBox.StandardButton.Yes
            )
            self.window.on_clear_trajectory()
        finally:
            QMessageBox.question = original_question

        self.assertEqual(self.window.trajectory.frames, [])
        self.assertEqual(self.viewer.robot_trajectory, [])
        self.assertEqual(self.viewer.timeslice_slider.defined_times, set())

    def test_load_qpos_rejects_wrong_value_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            source.write_text("1,2,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 36 qpos values"):
                self.viewer.load_qpos_csv(source)


if __name__ == "__main__":
    unittest.main()
