import os
import struct
import unittest
import numpy as np
import mujoco
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from OpenGL import GL
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gui.main_window import RobotGuiMainWindow
from core.models import MuJoCoRobotAdapter
from gui.viewers.robot_canvas_3d import RobotCanvas3D
from core.ik import CollisionAwareIKSolver, CollisionChecker
from core.models import resolve_mesh_path, validate_model_assets
from application.model_importer import (
    default_model_library_root,
    discover_imported_models,
    import_robot_model,
)
from core.models import PROJECT_ROOT, ROBOT_MODELS
from gui.viewers.transform_gizmo import GizmoInteractionState


def tiny_binary_stl(name=b"part"):
    triangles = (
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    )
    stl = bytearray(struct.pack("<80sI", name, len(triangles)))
    for triangle in triangles:
        stl.extend(struct.pack("<12fH", *triangle, 0))
    return stl


class MouseEventStub:
    def __init__(self, button, x, y, modifiers=Qt.KeyboardModifier.NoModifier):
        self._button = button
        self._position = QPointF(float(x), float(y))
        self._modifiers = modifiers

    def button(self):
        return self._button

    def position(self):
        return self._position

    def modifiers(self):
        return self._modifiers


class WheelEventStub:
    def __init__(self, delta_y):
        self._delta_y = int(delta_y)

    def angleDelta(self):
        class Delta:
            def __init__(self, y):
                self._y = y

            def y(self):
                return self._y

        return Delta(self._delta_y)


class RobotModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_unitree_bundled_model_display_names(self):
        self.assertEqual(ROBOT_MODELS["g1"].display_name, "Unitree G1")
        self.assertEqual(ROBOT_MODELS["go2"].display_name, "Unitree Go2")
        self.assertEqual(ROBOT_MODELS["h2"].display_name, "Unitree H2")
        self.assertEqual(ROBOT_MODELS["z1"].display_name, "Unitree Z1")

    def test_g1_metadata_and_logical_frames(self):
        adapter = MuJoCoRobotAdapter("g1")
        self.assertEqual(adapter.model_type, "humanoid")
        self.assertEqual(len(adapter.actuated_joints), 29)
        self.assertEqual(
            adapter.resolve_logical_frame("left_hand"),
            ("site", "robot/left_palm"),
        )
        self.assertGreater(len(adapter.get_kinematic_edges()), 0)

    def test_go2_urdf_loads_with_quadruped_metadata(self):
        adapter = MuJoCoRobotAdapter("go2")
        self.assertEqual(adapter.model_type, "quadruped")
        self.assertEqual(len(adapter.actuated_joints), 12)
        self.assertEqual(adapter.root_body, "base")
        self.assertIn("FL_foot", adapter.trajectory_frames)
        self.assertEqual(
            adapter.resolve_logical_frame("RR_foot"), ("site", "RR_foot")
        )
        self.assertIn(("base", "FL_hip"), adapter.get_kinematic_edges())
        self.assertIn("converted 17 COLLADA visual references", adapter.load_warning)
        self.assertEqual(adapter.model_path.name, "go2_description.urdf")
        self.assertGreater(adapter.mj_model.nmesh, 0)

    def test_home_pose_lowest_robot_point_is_grounded(self):
        for key in ("g1", "go2"):
            with self.subTest(model=key):
                adapter = MuJoCoRobotAdapter(key)
                lowest = adapter._lowest_robot_geom_z(adapter.mj_data)
                self.assertAlmostEqual(lowest, 0.002, places=6)

    def test_generic_free_root_home_pose_is_grounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "low.xml"
            source.write_text(
                """
<mujoco model="low">
  <worldbody>
    <body name="base">
      <freejoint name="floating_base"/>
      <geom type="sphere" size="0.05" pos="0 0 -0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
            )

            adapter = MuJoCoRobotAdapter(model=None, model_path=source)

            lowest = adapter._lowest_robot_geom_z(adapter.mj_data)
            self.assertAlmostEqual(lowest, 0.002, places=6)

    def test_go2_window_uses_go2_controls(self):
        window = RobotGuiMainWindow("go2")
        try:
            self.assertEqual(len(window.viewer_3d.joint_controls), 12)
            frames = [
                window.controls.frame_box.itemText(index)
                for index in range(window.controls.frame_box.count())
            ]
            self.assertEqual(
                frames,
                ["base", "trunk", "FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            )
        finally:
            window.close()

    def test_go2_real_mesh_visuals_are_rendered_instead_of_collision_primitives(self):
        adapter = MuJoCoRobotAdapter("go2")
        render_ids = RobotCanvas3D.render_geom_ids(adapter.mj_model)
        self.assertGreater(len(render_ids), 0)
        self.assertTrue(all(
            int(adapter.mj_model.geom_type[index])
            == int(mujoco.mjtGeom.mjGEOM_MESH)
            for index in render_ids
        ))
        self.assertTrue(all(
            int(adapter.mj_model.geom_group[index]) == 1
            for index in render_ids
        ))

    def test_registered_mesh_assets_resolve_without_current_working_directory(self):
        for key, info in ROBOT_MODELS.items():
            with self.subTest(model=key):
                results = validate_model_assets(info.model_path, info.package_map)
                self.assertGreater(len(results), 0)
                self.assertTrue(all(result.error is None for result in results))
        resolved = resolve_mesh_path(
            "package://go2_description/dae/base.dae",
            Path("/tmp"),
            ROBOT_MODELS["go2"].package_map,
        )
        self.assertIsNone(resolved.error)
        self.assertEqual(resolved.path.name, "base.dae")

    def test_missing_mesh_has_actionable_error(self):
        result = resolve_mesh_path("package://missing/dae/nope.dae", Path("/tmp"))
        self.assertIsNone(result.path)
        self.assertIn("unresolved mesh", result.error)

    def test_default_import_library_is_repo_models_folder(self):
        self.assertEqual(default_model_library_root(), PROJECT_ROOT / "models")

    def test_import_urdf_copies_package_meshes_to_model_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "go3_description"
            source_dir = package / "urdf"
            mesh_dir = package / "meshes"
            source_dir.mkdir(parents=True)
            mesh_dir.mkdir()
            (mesh_dir / "body.stl").write_bytes(tiny_binary_stl(b"body"))
            source = source_dir / "go3.urdf"
            source.write_text(
                """
<robot name="go3">
  <link name="base">
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://go3_description/meshes/body.stl"/>
      </geometry>
    </visual>
  </link>
</robot>
""".strip()
            )

            info = import_robot_model(source, root / "models")
            self.assertEqual(info.key, "go3")
            self.assertEqual(info.model_path, root / "models" / "go3.urdf")
            self.assertTrue((root / "models" / "assets-go3" / "body.stl").exists())
            saved = info.model_path.read_text()
            self.assertIn('filename="assets-go3/body.stl"', saved)

    def test_import_urdf_can_use_chosen_mesh_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            mesh_dir = root / "robot_parts"
            source_dir.mkdir()
            mesh_dir.mkdir()
            (mesh_dir / "pelvis.STL").write_bytes(tiny_binary_stl(b"pelvis"))
            source = source_dir / "go3.urdf"
            source.write_text(
                """
<robot name="go3">
  <link name="base">
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="meshes/pelvis.dae"/>
      </geometry>
    </visual>
  </link>
</robot>
""".strip()
            )

            info = import_robot_model(
                source, root / "models", mesh_roots=[mesh_dir]
            )
            self.assertTrue((root / "models" / "assets-go3" / "pelvis.STL").exists())
            saved = info.model_path.read_text()
            self.assertIn('filename="assets-go3/pelvis.STL"', saved)

    def test_import_urdf_with_virtual_world_root_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "z1.urdf"
            source.write_text(
                """
<robot name="z1">
  <link name="world"/>
  <joint name="base_static_joint" type="fixed">
    <parent link="world"/>
    <child link="link00"/>
  </joint>
  <link name="link00">
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="link00"/>
    <child link="link01"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-1" upper="1" velocity="1"/>
  </joint>
  <link name="link01">
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
    </collision>
  </link>
</robot>
""".strip()
            )

            with patch.dict(
                os.environ, {"GHOSTGUI_CACHE_DIR": str(root / "cache")}
            ):
                info = import_robot_model(source, root / "models")
                adapter = MuJoCoRobotAdapter(info)

            self.assertEqual(info.display_name, "Unitree Z1")
            self.assertEqual(adapter.root_body, "link00")
            self.assertIn("joint1", adapter.actuated_joints)

    def test_import_mjcf_rewrites_meshdir_to_model_asset_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            mesh_dir = source_dir / "meshes"
            mesh_dir.mkdir(parents=True)
            (mesh_dir / "part.stl").write_bytes(tiny_binary_stl())
            source = source_dir / "go3.xml"
            source.write_text(
                """
<mujoco model="go3">
  <compiler angle="radian" meshdir="meshes"/>
  <asset>
    <mesh name="part" file="part.stl"/>
  </asset>
  <worldbody/>
</mujoco>
""".strip()
            )

            info = import_robot_model(source, root / "models")
            self.assertTrue((root / "models" / "assets-go3" / "part.stl").exists())
            saved = info.model_path.read_text()
            self.assertIn('meshdir="assets-go3"', saved)
            self.assertIn('file="part.stl"', saved)

    def test_invalid_import_does_not_persist_model_or_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "go3.xml"
            source.write_text(
                """
<mujoco model="go3">
  <worldbody>
    <body name="base">
      <geom type="not_a_geom" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
            )

            with self.assertRaises(Exception):
                import_robot_model(source, root / "models")

            self.assertFalse((root / "models" / "go3.xml").exists())
            self.assertFalse((root / "models" / "assets-go3").exists())
            self.assertEqual(list((root / "models").glob(".import-*")), [])

    def test_discover_imported_models_does_not_load_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            broken = model_dir / "go3.xml"
            broken.write_text("<mujoco>")

            models = discover_imported_models(model_dir)

            self.assertIn("go3", models)
            self.assertEqual(models["go3"].model_path, broken.resolve())
            self.assertEqual(models["go3"].display_name, "Go3")

    def test_discover_imported_models_infers_known_unitree_display_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            model_dir.mkdir()
            (model_dir / "h2-5.urdf").write_text("<robot/>")
            (model_dir / "unitree-z1.xml").write_text("<mujoco/>")

            models = discover_imported_models(model_dir)

            self.assertEqual(models["h2-5"].display_name, "Unitree H2")
            self.assertEqual(models["unitree-z1"].display_name, "Unitree Z1")

    def test_discover_imported_models_skips_builtin_repo_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            model_dir.mkdir()
            (model_dir / "g1_29dof.xml").write_text("<mujoco/>")
            (model_dir / "go2_description.urdf").write_text("<robot/>")
            (model_dir / "h2.urdf").write_text("<robot/>")
            (model_dir / "z1.urdf").write_text("<robot/>")
            (model_dir / "custom_bot.urdf").write_text("<robot/>")

            models = discover_imported_models(model_dir)

            self.assertEqual(set(models), {"custom_bot"})

    def test_startup_registers_persisted_models_without_loading_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "go3.xml").write_text("<mujoco>")

            with patch(
                "gui.main_window.default_model_library_root",
                return_value=model_dir,
            ):
                window = RobotGuiMainWindow("g1")
            try:
                self.assertGreaterEqual(window.controls.model_box.findData("go3"), 0)
                self.assertEqual(window.model_key, "g1")
            finally:
                window.close()

    def test_open_model_file_button_imports_and_loads_custom_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "go3.xml"
            source.write_text(
                """
<mujoco model="go3">
  <worldbody>
    <body name="base">
      <freejoint name="floating_base"/>
      <geom type="sphere" size="0.05"/>
      <site name="tool" pos="0 0 0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
            )
            window = RobotGuiMainWindow("g1")
            try:
                window.model_library_root = root / "models"
                window.import_model_file(str(source))
                for _attempt in range(500):
                    if not window.background_jobs.is_busy():
                        break
                    QTest.qWait(10)
                self.assertFalse(window.background_jobs.is_busy())
                loader = window.model_loaders.get("go3")
                if loader is not None:
                    loader.wait()
                    self.app.processEvents()

                self.assertEqual(window.model_key, "go3")
                self.assertTrue((root / "models" / "go3.xml").exists())
                self.assertTrue((root / "models" / "assets-go3").is_dir())
                self.assertIn("tool", window.robot_model_3d.trajectory_frames)
            finally:
                window.close()

    def test_choose_mesh_folder_button_sets_import_mesh_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh_dir = Path(tmp) / "meshes"
            mesh_dir.mkdir()
            window = RobotGuiMainWindow("g1")
            try:
                with patch.object(
                    window.model_file_selection_stage,
                    "select_file",
                    side_effect=lambda **kwargs: (
                        kwargs["selected"](str(mesh_dir)) or True
                    ),
                ):
                    window.on_choose_mesh_folder()
                self.assertEqual(window.import_mesh_folder, mesh_dir.resolve())
                self.assertIn(str(mesh_dir), window.status_text.toPlainText())
            finally:
                window.close()

    def test_canvas_is_opaque_and_preview_alpha_is_clamped(self):
        canvas = RobotCanvas3D()
        try:
            self.assertEqual(canvas.format().alphaBufferSize(), 0)
            self.assertTrue(canvas.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent))
            self.assertFalse(canvas.testAttribute(
                Qt.WidgetAttribute.WA_TranslucentBackground
            ))
            canvas.set_preview_alpha(5.0)
            self.assertEqual(canvas.preview_alpha, 1.0)
            canvas.set_preview_alpha(0.0)
            self.assertEqual(canvas.preview_alpha, 0.1)
        finally:
            canvas.close()

    def test_gizmo_highlight_applies_only_while_hovered(self):
        canvas = RobotCanvas3D()
        try:
            base_color = (0.9, 0.1, 0.1)
            canvas.gizmo.state = GizmoInteractionState.HOVER_TRANSLATE_X
            self.assertEqual(
                canvas._gizmo_color("x", "TRANSLATE", base_color),
                (1.0, 0.9, 0.15),
            )
            canvas.gizmo.state = GizmoInteractionState.DRAG_TRANSLATE_X
            self.assertEqual(
                canvas._gizmo_color("x", "TRANSLATE", base_color),
                base_color,
            )
        finally:
            canvas.close()

    def test_selected_body_highlight_applies_only_to_owner_geoms(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        canvas = RobotCanvas3D()
        try:
            canvas.set_robot_states(state, adapter.create_state())
            calf_id = mujoco.mj_name2id(
                adapter.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FL_calf"
            )
            selected_geom = next(
                geom_id for geom_id in RobotCanvas3D.render_geom_ids(adapter.mj_model)
                if int(adapter.mj_model.geom_bodyid[geom_id]) == calf_id
            )
            other_geom = next(
                geom_id for geom_id in RobotCanvas3D.render_geom_ids(adapter.mj_model)
                if int(adapter.mj_model.geom_bodyid[geom_id]) != calf_id
            )

            canvas.set_selected_target("body", "FL_calf", calf_id)

            self.assertTrue(canvas._geom_is_selected_body(
                adapter.mj_model, selected_geom
            ))
            self.assertFalse(canvas._geom_is_selected_body(
                adapter.mj_model, other_geom
            ))
            original = adapter.get_geom_rgba(selected_geom)
            highlighted = canvas._selected_body_rgba(original)
            self.assertEqual(highlighted[3], original[3])
            self.assertTrue(np.all(np.asarray(highlighted[:3]) >= original[:3]))
        finally:
            canvas.close()

    def test_site_target_highlight_uses_owning_body(self):
        window = RobotGuiMainWindow("g1")
        try:
            viewer = window.viewer_3d
            kind, site_name = viewer.robot_model.resolve_logical_frame("left_hand")
            viewer.select_target(kind, site_name, emit=False)
            viewer._set_target_to_selected_pose()

            site_id = mujoco.mj_name2id(
                viewer.robot_model.mj_model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )
            owner_body_id = int(viewer.robot_model.mj_model.site_bodyid[site_id])
            self.assertEqual(viewer.canvas.selected_target_kind, "site")
            self.assertEqual(viewer.canvas.selected_target_name, site_name)
            self.assertEqual(viewer.canvas.selected_body_id, owner_body_id)
        finally:
            window.close()

    def test_target_marker_rotation_matrix_uses_selected_quaternion(self):
        matrix = RobotCanvas3D._quaternion_rotation_matrix(
            np.array([0.0, 0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(
            matrix,
            np.array([
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            atol=1e-12,
        )

    def test_transform_gizmo_hotkeys_switch_modes_and_cancel(self):
        canvas = RobotCanvas3D()
        try:
            changed_modes = []
            cancelled = []
            canvas.gizmo_mode_changed.connect(changed_modes.append)
            canvas.transform_drag_cancel_requested.connect(lambda: cancelled.append(True))

            canvas.keyPressEvent(QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_R,
                Qt.KeyboardModifier.NoModifier,
            ))
            self.assertEqual(canvas.gizmo.mode, "rotate")
            self.assertEqual(changed_modes[-1], "rotate")

            canvas.keyPressEvent(QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_T,
                Qt.KeyboardModifier.NoModifier,
            ))
            self.assertEqual(canvas.gizmo.mode, "translate")
            self.assertEqual(changed_modes[-1], "translate")

            canvas.gizmo.state = GizmoInteractionState.DRAG_TRANSLATE_X
            canvas.keyPressEvent(QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_E,
                Qt.KeyboardModifier.NoModifier,
            ))
            self.assertEqual(canvas.gizmo.state, GizmoInteractionState.NONE)
            self.assertEqual(cancelled, [True])
        finally:
            canvas.close()

    def test_3d_view_mouse_controls_match_mujoco_style_camera(self):
        canvas = RobotCanvas3D()
        try:
            with patch.object(canvas.gizmo, "begin_drag", return_value=False):
                canvas.mousePressEvent(MouseEventStub(
                    Qt.MouseButton.LeftButton, 100, 100
                ))
            self.assertTrue(canvas.rotating_camera)
            yaw_before = canvas.camera_yaw
            pitch_before = canvas.camera_pitch
            canvas.mouseMoveEvent(MouseEventStub(
                Qt.MouseButton.NoButton, 120, 90
            ))
            self.assertLess(canvas.camera_yaw, yaw_before)
            self.assertLess(canvas.camera_pitch, pitch_before)
            canvas.mouseReleaseEvent(MouseEventStub(
                Qt.MouseButton.LeftButton, 120, 90
            ))
            self.assertFalse(canvas.rotating_camera)

            center_before = canvas.camera_center.copy()
            distance_before = canvas.camera_distance
            canvas.mousePressEvent(MouseEventStub(
                Qt.MouseButton.RightButton, 100, 100
            ))
            self.assertTrue(canvas.panning_camera)
            canvas.mouseMoveEvent(MouseEventStub(
                Qt.MouseButton.NoButton, 130, 75
            ))
            self.assertFalse(np.allclose(canvas.camera_center, center_before))
            self.assertEqual(canvas.camera_distance, distance_before)
            canvas.mouseReleaseEvent(MouseEventStub(
                Qt.MouseButton.RightButton, 130, 75
            ))
            self.assertFalse(canvas.panning_camera)

            distance_before = canvas.camera_distance
            canvas.mousePressEvent(MouseEventStub(
                Qt.MouseButton.MiddleButton, 100, 100
            ))
            self.assertTrue(canvas.zooming_camera)
            canvas.mouseMoveEvent(MouseEventStub(
                Qt.MouseButton.NoButton, 100, 80
            ))
            self.assertGreater(canvas.camera_distance, distance_before)
            canvas.mouseMoveEvent(MouseEventStub(
                Qt.MouseButton.NoButton, 100, 120
            ))
            self.assertLess(canvas.camera_distance, distance_before + 0.1)
            canvas.mouseReleaseEvent(MouseEventStub(
                Qt.MouseButton.MiddleButton, 100, 120
            ))
            self.assertFalse(canvas.zooming_camera)
        finally:
            canvas.close()

    def test_scroll_up_zooms_out_and_scroll_down_zooms_in(self):
        canvas = RobotCanvas3D()
        try:
            distance_before = canvas.camera_distance
            canvas.wheelEvent(WheelEventStub(120))
            self.assertGreater(canvas.camera_distance, distance_before)
            zoomed_out = canvas.camera_distance
            canvas.wheelEvent(WheelEventStub(-120))
            self.assertLess(canvas.camera_distance, zoomed_out)
        finally:
            canvas.close()

    def test_transparent_pass_preserves_framebuffer_alpha_and_restores_state(self):
        with (
            patch.object(GL, "glEnable") as enable,
            patch.object(GL, "glDisable") as disable,
            patch.object(GL, "glBlendFuncSeparate") as blend,
            patch.object(GL, "glDepthMask") as depth_mask,
            patch.object(GL, "glColorMask") as color_mask,
        ):
            RobotCanvas3D._begin_transparent_pass()
            blend.assert_called_once_with(
                GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE
            )
            color_mask.assert_called_with(
                GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE
            )
            RobotCanvas3D._end_transparent_pass()
            color_mask.assert_called_with(
                GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE
            )
            depth_mask.assert_called_with(GL.GL_TRUE)
            disable.assert_called_with(GL.GL_BLEND)
            enable.assert_called_with(GL.GL_BLEND)

    def test_go2_runtime_has_lit_scene_and_distinct_model_colors(self):
        adapter = MuJoCoRobotAdapter("go2")
        model = adapter.mj_model
        self.assertGreaterEqual(float(model.vis.headlight.ambient[0]), 0.4)
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground"), 0
        )
        colors = np.unique(np.round(model.geom_rgba, 2), axis=0)
        self.assertGreaterEqual(len(colors), 5)

    def test_go2_foot_drag_converges_without_orientation_overconstraint(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        checker = CollisionChecker(adapter)
        solver = CollisionAwareIKSolver(
            adapter, checker, orientation_weight=0.0
        )
        kind, name = adapter.resolve_logical_frame("FL_foot")
        position, quaternion = state.get_body_pose(name, kind)
        result = solver.solve_drag(
            state.get_qpos(), position, quaternion,
            position + np.array([0.03, 0.0, 0.0]), quaternion,
            object_name=name, kind=kind,
        )
        self.assertTrue(result.success, result.status)
        self.assertEqual(result.accepted_fraction, 1.0)

    def test_go2_viewer_drag_is_preview_only_until_accept(self):
        window = RobotGuiMainWindow("go2")
        try:
            viewer = window.viewer_3d
            kind, name = viewer.robot_model.resolve_logical_frame("FL_foot")
            viewer.select_target(kind, name, emit=False)
            viewer._set_target_to_selected_pose()
            committed = viewer.committed_state.get_qpos()
            position = viewer.last_valid_target_position.copy()
            quaternion = viewer.last_valid_target_quaternion.copy()
            viewer._on_transform_moved(
                position + np.array([0.02, 0.0, 0.0]), quaternion
            )
            np.testing.assert_allclose(viewer.committed_state.get_qpos(), committed)
            self.assertFalse(np.allclose(viewer.preview_state.get_qpos(), committed))
            viewer.accept_preview()
            np.testing.assert_allclose(
                viewer.committed_state.get_qpos(), viewer.preview_state.get_qpos()
            )
        finally:
            window.close()

    def test_picked_bodies_map_to_logical_frames_for_both_models(self):
        g1 = MuJoCoRobotAdapter("g1")
        self.assertEqual(
            g1.logical_frame_for_body("robot/left_wrist_yaw_link"), "left_hand"
        )
        self.assertEqual(
            g1.logical_frame_for_body("robot/right_ankle_roll_link"), "right_foot"
        )
        go2 = MuJoCoRobotAdapter("go2")
        self.assertEqual(go2.logical_frame_for_body("FL_hip"), "FL_foot")
        self.assertEqual(go2.logical_frame_for_body("base"), "base")
        z1 = MuJoCoRobotAdapter("z1")
        self.assertEqual(z1.trajectory_frames, ["base", "tool", "wrist"])
        self.assertEqual(z1.logical_frame_for_body("link05"), "wrist")
        self.assertEqual(z1.logical_frame_for_body("link06"), "tool")

    def test_ray_pick_returns_a_robot_body(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        canvas = RobotCanvas3D()
        canvas.set_robot_states(state, adapter.create_state())
        calf_id = mujoco.mj_name2id(
            adapter.mj_model, mujoco.mjtObj.mjOBJ_BODY, "FL_calf"
        )
        geom_id = min(
            (
                index for index in RobotCanvas3D.render_geom_ids(adapter.mj_model)
                if int(adapter.mj_model.geom_bodyid[index]) == calf_id
            ),
            key=lambda index: state.mj_data.geom_xpos[index][2],
        )
        center = state.mj_data.geom_xpos[geom_id]
        picked = canvas.pick_robot_body_from_ray(
            center + np.array([0.0, 2.0, 0.0]), np.array([0.0, -1.0, 0.0])
        )
        self.assertIsNotNone(picked)
        self.assertEqual(adapter.logical_frame_for_body(picked), "FL_foot")

    def test_z1_ray_pick_prefers_wrist_and_tool_over_large_parent_bounds(self):
        adapter = MuJoCoRobotAdapter("z1")
        state = adapter.create_state()
        canvas = RobotCanvas3D()
        canvas.set_robot_states(state, adapter.create_state())
        try:
            cases = {
                "link05": "wrist",
                "link06": "tool",
            }
            for body_name, logical_frame in cases.items():
                with self.subTest(body=body_name):
                    body_id = mujoco.mj_name2id(
                        adapter.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
                    )
                    geom_id = next(
                        index for index in RobotCanvas3D.render_geom_ids(adapter.mj_model)
                        if int(adapter.mj_model.geom_bodyid[index]) == body_id
                    )
                    center = state.mj_data.geom_xpos[geom_id]
                    picked = canvas.pick_robot_body_from_ray(
                        center + np.array([0.0, 2.0, 0.0]),
                        np.array([0.0, -1.0, 0.0]),
                    )
                    self.assertEqual(picked, body_name)
                    self.assertEqual(
                        adapter.logical_frame_for_body(picked), logical_frame
                    )
        finally:
            canvas.close()

    def test_double_click_body_selection_updates_frame_editor(self):
        g1_window = RobotGuiMainWindow("g1")
        go2_window = RobotGuiMainWindow("go2")
        z1_window = RobotGuiMainWindow("z1")
        try:
            g1_window.viewer_3d._on_body_double_clicked(
                "robot/left_wrist_yaw_link"
            )
            self.assertEqual(g1_window.controls.frame_box.currentText(), "left_hand")
            self.assertEqual(
                g1_window.viewer_3d._selected_target(),
                ("site", "robot/left_palm"),
            )
            go2_window.viewer_3d._on_body_double_clicked("FR_thigh")
            self.assertEqual(go2_window.controls.frame_box.currentText(), "FR_foot")
            self.assertEqual(
                go2_window.viewer_3d._selected_target(), ("site", "FR_foot")
            )
            self.assertEqual(
                [
                    z1_window.controls.frame_box.itemText(index)
                    for index in range(z1_window.controls.frame_box.count())
                ],
                ["base", "tool", "wrist"],
            )
            z1_window.viewer_3d._on_body_double_clicked("link05")
            self.assertEqual(z1_window.controls.frame_box.currentText(), "wrist")
            self.assertEqual(
                z1_window.viewer_3d._selected_target(), ("body", "link05")
            )
            z1_window.viewer_3d._on_body_double_clicked("link06")
            self.assertEqual(z1_window.controls.frame_box.currentText(), "tool")
            self.assertEqual(
                z1_window.viewer_3d._selected_target(), ("body", "link06")
            )
        finally:
            g1_window.close()
            go2_window.close()
            z1_window.close()

    def test_model_switch_preserves_selected_editor_tab(self):
        window = RobotGuiMainWindow("g1")
        try:
            for mode in ("simulation", "3d"):
                selected = (
                    window.viewer_3d_mujoco if mode == "simulation"
                    else window.viewer_3d_stack
                )
                window.viewer_tabs.setCurrentWidget(selected)
                target_key = "go2" if window.model_key == "g1" else "g1"
                window.on_model_changed(target_key)
                loader = window.model_loaders.get(target_key)
                if loader is not None:
                    loader.wait()
                    self.app.processEvents()
                self.assertEqual(window.model_key, target_key)
                self.assertIs(window.viewer_tabs.currentWidget(), selected)
                if mode == "3d":
                    self.assertIs(selected.currentWidget(), window.viewer_3d)
        finally:
            window.close()

    def test_model_session_is_reused_after_switching_back(self):
        window = RobotGuiMainWindow("g1")
        try:
            g1_viewer = window.viewer_3d
            window.on_model_changed("go2")
            loader = window.model_loaders.get("go2")
            if loader is not None:
                loader.wait()
                self.app.processEvents()
            go2_viewer = window.viewer_3d
            window.on_model_changed("g1")
            self.assertIs(window.viewer_3d, g1_viewer)
            window.on_model_changed("go2")
            self.assertIs(window.viewer_3d, go2_viewer)
        finally:
            window.close()

    def test_user_urdf_uses_versioned_persistent_cache(self):
        urdf = """<robot name='cache_test'>
          <link name='base'><inertial><mass value='1'/><inertia ixx='1' iyy='1' izz='1' ixy='0' ixz='0' iyz='0'/></inertial><collision><geometry><box size='1 1 1'/></geometry></collision></link>
        </robot>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "robot.urdf"
            source.write_text(urdf, encoding="utf-8")
            old_cache = os.environ.get("GHOSTGUI_CACHE_DIR")
            os.environ["GHOSTGUI_CACHE_DIR"] = str(temp / "cache")
            try:
                first = MuJoCoRobotAdapter.load_model(source)
                second = MuJoCoRobotAdapter.load_model(source)
                self.assertEqual(first.runtime_model_path, second.runtime_model_path)
                self.assertTrue(first.runtime_model_path.exists())
                self.assertTrue(first.runtime_model_path.with_name("metadata.json").exists())
            finally:
                if old_cache is None:
                    os.environ.pop("GHOSTGUI_CACHE_DIR", None)
                else:
                    os.environ["GHOSTGUI_CACHE_DIR"] = old_cache


if __name__ == "__main__":
    unittest.main()
