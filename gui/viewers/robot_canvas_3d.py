"""
viewer_3d.py

Purpose:
    OpenGL 3D viewer/editor for the target reference frame and trajectory.

The editor exposes a small scene-update contract:
    - update_scene(trajectory, active_frame)
    - target_dragged(x, z)
"""

import math

import numpy as np

from OpenGL import GL
from OpenGL import GLU
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QMatrix4x4, QSurfaceFormat, QVector3D
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from gui.render_scheduler import RenderRequestCoalescer
from .camera import OrbitCamera
from .trajectory_colors import gl_color_for_frame
from core.trajectory import rpy_to_quat
from .transform_gizmo import GizmoInteractionState, TransformGizmo

TRAJECTORY_LINE_DT = 0.02


class RobotCanvas3D(QOpenGLWidget):
    target_dragged = Signal(float, float)
    target_pose_dragged = Signal(float, float, float)
    target_transform_dragged = Signal(object, object)
    transform_drag_finished = Signal()
    transform_drag_cancel_requested = Signal()
    gizmo_mode_changed = Signal(str)
    geometry_progress = Signal(int, int)
    body_double_clicked = Signal(str)
    camera_changed = Signal()

    @property
    def camera_distance(self):
        return self.camera_controller.distance

    @camera_distance.setter
    def camera_distance(self, value):
        self.camera_controller.distance = float(value)

    @property
    def camera_yaw(self):
        return self.camera_controller.yaw

    @camera_yaw.setter
    def camera_yaw(self, value):
        self.camera_controller.yaw = float(value)

    @property
    def camera_pitch(self):
        return self.camera_controller.pitch

    @camera_pitch.setter
    def camera_pitch(self, value):
        self.camera_controller.pitch = float(value)

    @property
    def camera_center(self):
        return self.camera_controller.center

    @camera_center.setter
    def camera_center(self, value):
        self.camera_controller.center = np.asarray(value, dtype=float)

    def __init__(self):
        super().__init__()

        # This widget is an opaque scene. Request no composited alpha channel
        # and tell Qt not to expose the desktop through transparent GL pixels.
        surface_format = QSurfaceFormat(self.format())
        surface_format.setAlphaBufferSize(0)
        self.setFormat(surface_format)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self.setMinimumSize(650, 500)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.trajectory = None
        self.show_trajectory_lines = True
        self.show_keyframes = True
        self.trajectory_smoothing = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.9
        self.target_yaw = 0.0

        self.camera_controller = OrbitCamera()
        self.render_requests = RenderRequestCoalescer(self.update, self)

        self.dragging_target = False
        self.rotating_camera = False
        self.panning_camera = False
        self.zooming_camera = False
        self.last_mouse_pos = None

        self._model_view = QMatrix4x4()
        self._projection = QMatrix4x4()
        self._viewport = QRect(0, 0, 1, 1)
        self.robot_state = None
        self.preview_state = None
        self.preview_visible = False
        self.preview_alpha = 0.65
        self.ghost_renderer = None
        self.show_ghosts = False
        self.ghost_alpha = 0.18
        self.use_model_colors = True
        self.selected_target_kind = None
        self.selected_target_name = None
        self.selected_body_id = None
        self.preview_collision_geom_ids = set()
        self.preview_collision_body_ids = set()
        self._geom_lists = []
        self._mesh_display_lists = {}
        self._quadric = None
        self.gizmo = TransformGizmo(
            (self.target_x, self.target_y, self.target_z)
        )
        self.transform_gizmo_visible = True
        self.transform_gizmo_interactive = True
        self._geometry_build_count = 0
        self._geometry_queue = []
        self._geometry_total = 0
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setInterval(0)
        self._geometry_timer.timeout.connect(self._compile_next_geometry)
        self._shutdown = False
        self._resource_context = None

    def set_robot_state(self, robot_state, ghost_renderer=None):
        self.set_robot_states(robot_state, None, ghost_renderer)

    def set_robot_states(
        self, committed_state, preview_state=None, ghost_renderer=None
    ):
        self.robot_state = committed_state
        self.preview_state = preview_state
        self.ghost_renderer = ghost_renderer
        if self.isValid():
            self._build_robot_geometry()
        self.update()

    def set_preview_visible(self, visible):
        self.preview_visible = bool(visible and self.preview_state is not None)
        self.update()

    def set_preview_alpha(self, alpha):
        self.preview_alpha = max(0.1, min(1.0, float(alpha)))
        self.update()

    def set_preview_collisions(self, collisions):
        """Highlight visible preview geometry involved in MuJoCo contacts."""
        geom_ids = set()
        body_ids = set()
        model = self.robot_state.mj_model if self.robot_state is not None else None

        for collision in collisions or []:
            for side in (1, 2):
                geom_id = getattr(collision, f"geom{side}_id", None)
                body_id = getattr(collision, f"body{side}_id", None)
                if geom_id is not None and int(geom_id) >= 0:
                    geom_id = int(geom_id)
                    geom_ids.add(geom_id)
                    if body_id is None and model is not None:
                        body_id = int(model.geom_bodyid[geom_id])
                if body_id is not None and int(body_id) > 0:
                    body_ids.add(int(body_id))

        self.preview_collision_geom_ids = geom_ids
        self.preview_collision_body_ids = body_ids
        self.update()

    def set_ghost_options(self, visible, alpha=0.18):
        self.show_ghosts = bool(visible)
        self.ghost_alpha = max(0.02, min(0.8, float(alpha)))
        self.update()

    def set_use_model_colors(self, enabled):
        self.use_model_colors = bool(enabled)
        self.update()

    def set_selected_target(self, kind=None, name=None, owner_body_id=None):
        self.selected_target_kind = kind
        self.selected_target_name = name
        self.selected_body_id = (
            None if owner_body_id is None else int(owner_body_id)
        )
        self.update()

    def set_target_pose(self, position, quaternion=None):
        self.target_x, self.target_y, self.target_z = map(float, position)
        self.gizmo.set_pose(position, quaternion)
        self.update()

    def set_transform_gizmo_interactive(self, enabled):
        enabled = bool(enabled)
        if self.transform_gizmo_interactive == enabled:
            return
        self.transform_gizmo_interactive = enabled
        if not enabled:
            self.cancel_transform_drag()
            self.gizmo.end_drag()
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_transform_gizmo_visible(self, visible):
        visible = bool(visible)
        if self.transform_gizmo_visible == visible:
            return
        self.transform_gizmo_visible = visible
        if not visible:
            self.cancel_transform_drag()
            self.gizmo.end_drag()
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    # ============================================================
    # Scene API
    # ============================================================

    def update_scene(
        self,
        trajectory,
        active_frame=None,
        show_trajectory_lines=True,
        trajectory_smoothing=0.0,
        show_keyframes=True,
    ):
        self.trajectory = trajectory
        self.show_trajectory_lines = show_trajectory_lines
        self.show_keyframes = show_keyframes
        self.trajectory_smoothing = max(0.0, min(1.0, float(trajectory_smoothing)))

        if active_frame is not None:
            self.target_x = active_frame.x
            self.target_y = active_frame.y
            self.target_z = active_frame.z
            self.target_yaw = active_frame.yaw
            self.gizmo.set_pose(
                (active_frame.x, active_frame.y, active_frame.z),
                rpy_to_quat(
                    active_frame.roll,
                    active_frame.pitch,
                    active_frame.yaw,
                ),
            )

        self.request_render()

    def request_render(self):
        """Schedule one repaint for any number of same-turn invalidations."""
        return self.render_requests.request()

    # ============================================================
    # OpenGL lifecycle
    # ============================================================

    def initializeGL(self):
        self._shutdown = False
        self._connect_context_lifecycle()
        GL.glClearColor(0.08, 0.09, 0.10, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(8.0)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.28, 0.28, 0.28, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.78, 0.78, 0.78, 1.0))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
        self._quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self._quadric, GLU.GLU_SMOOTH)
        # Display lists belong to this OpenGL context. A restored/cached widget
        # keeps them; a genuinely new context rebuilds them incrementally.
        self._geom_lists = []
        self._mesh_display_lists = {}
        self._geometry_queue = []
        self._build_robot_geometry()

    def _connect_context_lifecycle(self):
        context = self.context()
        if context is None or context is self._resource_context:
            return
        self._resource_context = context
        context.aboutToBeDestroyed.connect(
            self._on_context_about_to_be_destroyed,
            Qt.ConnectionType.DirectConnection,
        )

    def _on_context_about_to_be_destroyed(self):
        self.cleanup_gl_resources()

    def cleanup_gl_resources(self, *, context_current=False):
        """Release display lists and GLU objects exactly once per context."""
        self._geometry_timer.stop()
        self._geometry_queue = []
        list_ids = {
            int(list_id)
            for list_id in (
                *self._geom_lists,
                *self._mesh_display_lists.values(),
            )
            if list_id
        }
        quadric = self._quadric
        if not list_ids and quadric is None:
            self._geom_lists = []
            self._mesh_display_lists = {}
            return

        made_current = False
        can_delete = bool(context_current)
        if not can_delete and self.isValid():
            try:
                self.makeCurrent()
                made_current = True
                can_delete = True
            except RuntimeError:
                can_delete = False
        try:
            if can_delete:
                for list_id in list_ids:
                    GL.glDeleteLists(list_id, 1)
                if quadric is not None:
                    GLU.gluDeleteQuadric(quadric)
        finally:
            self._geom_lists = []
            self._mesh_display_lists = {}
            self._quadric = None
            if made_current:
                try:
                    self.doneCurrent()
                except RuntimeError:
                    pass

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        self.render_requests.shutdown()
        self._geometry_timer.stop()
        self.cancel_transform_drag()
        self.cleanup_gl_resources()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def paintGL(self):
        # Reassert an opaque framebuffer on every frame. Transparent robot
        # passes preserve destination alpha rather than changing window alpha.
        GL.glClearColor(0.08, 0.09, 0.10, 1.0)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.configure_camera()

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(self.matrix_values(self._projection))

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(self.matrix_values(self._model_view))

        self.draw_ground_grid()
        self.draw_world_axes()
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (2.0, -3.0, 5.0, 1.0))
        self.draw_robot()
        self.draw_trajectory_ghosts()
        self.draw_preview_robot()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_LIGHT0)
        self.draw_trajectory()
        self.draw_selected_target_marker()
        if self.transform_gizmo_visible:
            self.draw_transform_gizmo()

    def _build_robot_geometry(self):
        """Queue local geometry so Qt can repaint between expensive meshes."""
        if self.robot_state is None or self._geom_lists:
            return
        model = self.robot_state.mj_model
        self._geom_lists = [None] * model.ngeom
        render_ids = self.render_geom_ids(model)
        # Native MJCF models usually use visual group 2; imported URDF visuals
        # use non-colliding group 1. render_geom_ids handles both conventions.
        self._geometry_queue = sorted(render_ids)
        self._geometry_total = len(self._geometry_queue)
        self.geometry_progress.emit(0, self._geometry_total)
        if self._geometry_queue:
            self._geometry_timer.start()

    def _compile_next_geometry(self):
        if not self._geometry_queue:
            self._geometry_timer.stop()
            return
        if self._shutdown or not self.isValid() or self.robot_state is None:
            return
        geom_id = self._geometry_queue.pop(0)
        model = self.robot_state.mj_model
        import mujoco
        mesh_id = (
            int(model.geom_dataid[geom_id])
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            else None
        )
        if mesh_id is not None and mesh_id in self._mesh_display_lists:
            self._geom_lists[geom_id] = self._mesh_display_lists[mesh_id]
            self._finish_geometry_item()
            return
        self.makeCurrent()
        try:
            list_id = GL.glGenLists(1)
            GL.glNewList(list_id, GL.GL_COMPILE)
            self._draw_local_geom(model, geom_id)
            GL.glEndList()
            self._geom_lists[geom_id] = list_id
            if mesh_id is not None:
                self._mesh_display_lists[mesh_id] = list_id
        finally:
            self.doneCurrent()
        self._finish_geometry_item()

    def _finish_geometry_item(self):
        complete = self._geometry_total - len(self._geometry_queue)
        self.geometry_progress.emit(complete, self._geometry_total)
        self.update()
        if not self._geometry_queue:
            self._geometry_timer.stop()
            self._geometry_build_count += 1

    @staticmethod
    def render_geom_ids(model):
        visual_ids = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_group[geom_id]) == 2
        }
        if not visual_ids:
            # MuJoCo imports URDF <visual> elements as non-colliding group-1
            # geoms. Prefer those over the primitive collision shapes.
            visual_ids = {
                geom_id for geom_id in range(model.ngeom)
                if int(model.geom_group[geom_id]) == 1
                and int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            }
        return visual_ids or set(range(model.ngeom))

    def _draw_local_geom(self, model, geom_id):
        import mujoco

        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            vertices = model.mesh_vert[
                vertex_start:vertex_start + int(model.mesh_vertnum[mesh_id])
            ]
            normal_start = int(model.mesh_normaladr[mesh_id])
            normals = model.mesh_normal[
                normal_start:normal_start + int(model.mesh_normalnum[mesh_id])
            ]
            faces = model.mesh_face[
                face_start:face_start + int(model.mesh_facenum[mesh_id])
            ]
            face_normals = model.mesh_facenormal[
                face_start:face_start + int(model.mesh_facenum[mesh_id])
            ]
            GL.glBegin(GL.GL_TRIANGLES)
            for face, normal_ids in zip(faces, face_normals):
                fallback_normal = np.cross(
                    vertices[int(face[1])] - vertices[int(face[0])],
                    vertices[int(face[2])] - vertices[int(face[0])],
                )
                fallback_norm = float(np.linalg.norm(fallback_normal))
                if fallback_norm > 1e-12:
                    fallback_normal /= fallback_norm
                for vertex_id, normal_id in zip(face, normal_ids):
                    if int(normal_id) >= 0:
                        GL.glNormal3fv(normals[int(normal_id)])
                    elif fallback_norm > 1e-12:
                        GL.glNormal3fv(fallback_normal)
                    GL.glVertex3fv(vertices[int(vertex_id)])
            GL.glEnd()
            return
        if self._quadric is None:
            return
        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            GLU.gluSphere(self._quadric, float(size[0]), 12, 8)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            GL.glPushMatrix()
            GL.glScalef(float(size[0]), float(size[1]), float(size[2]))
            GLU.gluSphere(self._quadric, 1.0, 12, 8)
            GL.glPopMatrix()
        elif geom_type in (
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        ):
            radius, half_length = float(size[0]), float(size[1])
            GL.glTranslatef(0.0, 0.0, -half_length)
            GLU.gluCylinder(self._quadric, radius, radius, 2.0 * half_length, 12, 1)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                GLU.gluSphere(self._quadric, radius, 12, 8)
                GL.glTranslatef(0.0, 0.0, 2.0 * half_length)
                GLU.gluSphere(self._quadric, radius, 12, 8)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            x, y, z = (float(v) for v in size)
            vertices = [
                (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
            ]
            faces = [
                ((0, 1, 2, 3), (0, 0, -1)),
                ((4, 7, 6, 5), (0, 0, 1)),
                ((0, 4, 5, 1), (0, -1, 0)),
                ((1, 5, 6, 2), (1, 0, 0)),
                ((2, 6, 7, 3), (0, 1, 0)),
                ((4, 0, 3, 7), (-1, 0, 0)),
            ]
            GL.glBegin(GL.GL_QUADS)
            for face, normal in faces:
                GL.glNormal3fv(normal)
                for vertex in face:
                    GL.glVertex3fv(vertices[vertex])
            GL.glEnd()

    @staticmethod
    def _transform_matrix(position, rotation):
        matrix = [0.0] * 16
        rotation = rotation.reshape(3, 3)
        for row in range(3):
            for column in range(3):
                matrix[column * 4 + row] = float(rotation[row, column])
        matrix[15] = 1.0
        matrix[12:15] = [float(v) for v in position]
        return matrix

    def _draw_robot_transforms(
        self,
        positions,
        rotations,
        alpha_scale=1.0,
        color_override=None,
        show_preview_collisions=False,
    ):
        if self.robot_state is None or not self._geom_lists:
            return
        model = self.robot_state.mj_model
        for geom_id, list_id in enumerate(self._geom_lists):
            if list_id is None:
                continue
            selected = color_override is None and self._geom_is_selected_body(
                model, geom_id
            )
            collision_warning = (
                show_preview_collisions
                and self._geom_is_preview_collision(model, geom_id)
            )
            if collision_warning:
                rgba = (1.0, 0.03, 0.03, 1.0)
            elif color_override is not None:
                rgba = color_override
            elif self.use_model_colors:
                rgba = self.robot_state.robot_model.get_geom_rgba(geom_id)
            else:
                rgba = (0.55, 0.55, 0.55, 1.0)
            if selected:
                rgba = self._selected_body_rgba(rgba)
            self._apply_geom_material(
                model,
                geom_id,
                selected=selected,
                warning=collision_warning,
            )
            GL.glColor4f(float(rgba[0]), float(rgba[1]), float(rgba[2]),
                         float(rgba[3]) * alpha_scale)
            GL.glPushMatrix()
            GL.glMultMatrixf(self._transform_matrix(positions[geom_id], rotations[geom_id]))
            GL.glCallList(list_id)
            GL.glPopMatrix()

    def _geom_is_selected_body(self, model, geom_id):
        return (
            self.selected_body_id is not None
            and int(model.geom_bodyid[geom_id]) == int(self.selected_body_id)
        )

    def _geom_is_preview_collision(self, model, geom_id):
        return (
            int(geom_id) in self.preview_collision_geom_ids
            or int(model.geom_bodyid[geom_id]) in self.preview_collision_body_ids
        )

    @staticmethod
    def _selected_body_rgba(rgba):
        rgba = np.asarray(rgba, dtype=float)
        bright = np.clip(
            rgba[:3] * 1.35 + np.array([0.18, 0.14, 0.03]),
            0.0,
            1.0,
        )
        return (float(bright[0]), float(bright[1]), float(bright[2]), float(rgba[3]))

    @staticmethod
    def _apply_geom_material(model, geom_id, selected=False, warning=False):
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            specular = float(model.mat_specular[material_id])
            shininess = float(model.mat_shininess[material_id]) * 128.0
            emission = float(model.mat_emission[material_id])
        else:
            specular, shininess, emission = 0.2, 32.0, 0.0
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK,
            GL.GL_SPECULAR,
            (specular, specular, specular, 1.0),
        )
        GL.glMaterialf(
            GL.GL_FRONT_AND_BACK,
            GL.GL_SHININESS,
            max(0.0, min(128.0, shininess)),
        )
        if warning:
            emission_rgba = (
                min(1.0, emission + 0.28),
                min(1.0, emission + 0.02),
                min(1.0, emission + 0.02),
                1.0,
            )
        else:
            emission_rgba = (
                min(1.0, emission + (0.18 if selected else 0.0)),
                min(1.0, emission + (0.14 if selected else 0.0)),
                min(1.0, emission + (0.04 if selected else 0.0)),
                1.0,
            )
        GL.glMaterialfv(
            GL.GL_FRONT_AND_BACK,
            GL.GL_EMISSION,
            emission_rgba,
        )

    def draw_robot(self):
        if self.robot_state is None:
            return
        data = self.robot_state.mj_data
        self._draw_robot_transforms(data.geom_xpos, data.geom_xmat)

    def draw_preview_robot(self):
        if not self.preview_visible or self.preview_state is None:
            return
        data = self.preview_state.mj_data
        if self.preview_alpha >= 0.999:
            self._draw_robot_transforms(
                data.geom_xpos,
                data.geom_xmat,
                color_override=(1.0, 0.38, 0.04, 1.0),
                show_preview_collisions=True,
            )
            return
        self._begin_transparent_pass()
        try:
            self._draw_robot_transforms(
                data.geom_xpos,
                data.geom_xmat,
                alpha_scale=self.preview_alpha,
                color_override=(1.0, 0.38, 0.04, 1.0),
                show_preview_collisions=True,
            )
        finally:
            self._end_transparent_pass()

    def draw_trajectory_ghosts(self):
        if not self.show_ghosts or self.ghost_renderer is None:
            return
        self._begin_transparent_pass()
        try:
            for positions, rotations in self.ghost_renderer.transforms:
                # The animated main pose may exactly equal one cached waypoint.
                # Skipping that duplicate avoids coincident transparent surfaces.
                if self.robot_state is not None and np.allclose(
                    positions,
                    self.robot_state.mj_data.geom_xpos,
                    atol=1e-8,
                    rtol=0.0,
                ):
                    continue
                self._draw_robot_transforms(positions, rotations, self.ghost_alpha)
        finally:
            self._end_transparent_pass()

    @staticmethod
    def _begin_transparent_pass():
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA,
            GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ZERO,
            GL.GL_ONE,
        )
        GL.glDepthMask(GL.GL_FALSE)
        # Preserve the opaque alpha written by glClear. Standard blending also
        # blends destination alpha, which made QOpenGLWidget composite desktop
        # pixels through the preview on some window managers.
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE)

    @staticmethod
    def _end_transparent_pass():
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    def configure_camera(self):
        width = max(1, self.width())
        height = max(1, self.height())
        aspect = width / height

        eye_array = self._camera_eye()
        eye = QVector3D(*map(float, eye_array))
        center = QVector3D(*map(float, self.camera_center))
        up = QVector3D(0.0, 0.0, 1.0)

        self._projection = QMatrix4x4()
        self._projection.perspective(45.0, aspect, 0.05, 100.0)

        self._model_view = QMatrix4x4()
        self._model_view.lookAt(eye, center, up)

        self._viewport = QRect(0, 0, width, height)

    def matrix_values(self, matrix):
        data = matrix.data()
        return [data[i] for i in range(16)]

    # ============================================================
    # Drawing helpers
    # ============================================================

    def draw_ground_grid(self):
        GL.glLineWidth(1.0)
        GL.glColor3f(0.30, 0.33, 0.35)

        GL.glBegin(GL.GL_LINES)
        for i in range(-10, 11):
            v = i * 0.25
            GL.glVertex3f(-2.5, v, 0.0)
            GL.glVertex3f(2.5, v, 0.0)
            GL.glVertex3f(v, -2.5, 0.0)
            GL.glVertex3f(v, 2.5, 0.0)
        GL.glEnd()

        GL.glLineWidth(2.0)
        GL.glColor3f(0.55, 0.58, 0.60)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(-2.5, 0.0, 0.0)
        GL.glVertex3f(2.5, 0.0, 0.0)
        GL.glVertex3f(0.0, -2.5, 0.0)
        GL.glVertex3f(0.0, 2.5, 0.0)
        GL.glEnd()

    def draw_world_axes(self):
        GL.glLineWidth(3.0)
        GL.glBegin(GL.GL_LINES)

        GL.glColor3f(0.90, 0.15, 0.12)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.6, 0.0, 0.02)

        GL.glColor3f(0.20, 0.75, 0.25)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.0, 0.6, 0.02)

        GL.glColor3f(0.20, 0.45, 0.95)
        GL.glVertex3f(0.0, 0.0, 0.02)
        GL.glVertex3f(0.0, 0.0, 0.6)

        GL.glEnd()

    def draw_trajectory(self):
        if self.trajectory is None or len(self.trajectory.frames) == 0:
            return

        if self.show_trajectory_lines:
            samples = self.trajectory.sample_tracks_uniform_dt(
                dt=TRAJECTORY_LINE_DT,
                smoothing=self.trajectory_smoothing,
            )
            frames_by_name = {}
            for sample in samples:
                for frame_name, target in sample["targets"].items():
                    frames_by_name.setdefault(frame_name, []).append(target)

            GL.glLineWidth(2.0)

            for frame_name, targets in frames_by_name.items():
                if len(targets) < 2:
                    continue

                GL.glColor3f(*gl_color_for_frame(frame_name))
                GL.glBegin(GL.GL_LINE_STRIP)
                for target in targets:
                    GL.glVertex3f(target.x, target.y, target.z)
                GL.glEnd()

        if self.show_keyframes:
            GL.glPointSize(7.0)
            GL.glBegin(GL.GL_POINTS)
            for frame in self.trajectory.frames:
                GL.glColor3f(*gl_color_for_frame(frame.frame_name))
                GL.glVertex3f(frame.x, frame.y, frame.z)
            GL.glEnd()

    def draw_target_frame(self):
        x = self.target_x
        y = self.target_y
        z = self.target_z

        GL.glPointSize(12.0)
        GL.glColor3f(1.0, 0.12, 0.08)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3f(x, y, z)
        GL.glEnd()

    def draw_selected_target_marker(self):
        if self.selected_target_name is None:
            return
        origin = np.asarray(self.gizmo.position, dtype=float)
        rotation = self._quaternion_rotation_matrix(self.gizmo.quaternion)
        length = max(self.gizmo.sphere_radius * 2.6, 0.045)
        colors = (
            ((1.0, 0.10, 0.08), rotation[:, 0]),
            ((0.10, 0.85, 0.25), rotation[:, 1]),
            ((0.22, 0.50, 1.0), rotation[:, 2]),
        )

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(4.0)
        GL.glBegin(GL.GL_LINES)
        for color, axis in colors:
            endpoint = origin + np.asarray(axis, dtype=float) * length
            GL.glColor3f(*color)
            GL.glVertex3fv(origin)
            GL.glVertex3fv(endpoint)
        GL.glEnd()
        GL.glPointSize(7.0)
        GL.glColor3f(1.0, 0.92, 0.18)
        GL.glBegin(GL.GL_POINTS)
        GL.glVertex3fv(origin)
        GL.glEnd()
        GL.glEnable(GL.GL_DEPTH_TEST)

    @staticmethod
    def _quaternion_rotation_matrix(quaternion):
        w, x, y, z = np.asarray(quaternion, dtype=float)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-12:
            return np.eye(3)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ], dtype=float)

    def draw_transform_gizmo(self):
        if not self.transform_gizmo_visible:
            return
        self._sync_gizmo_screen_scale()
        origin = self.gizmo.position
        show_sphere, translation_axes, rotation_axes = (
            self.gizmo.visible_handles()
        )
        colors = {
            "x": (0.9, 0.1, 0.1),
            "y": (0.1, 0.8, 0.2),
            "z": (0.2, 0.45, 1.0),
        }
        GL.glDisable(GL.GL_DEPTH_TEST)
        self._draw_gizmo_drag_guide(colors)
        sphere_hovered = (
            self.gizmo.state == GizmoInteractionState.HOVER_TRANSLATE_FREE
        )
        sphere_color = (
            (1.0, 0.9, 0.15) if sphere_hovered else (0.95, 0.75, 0.2)
        )
        GL.glColor3f(*sphere_color)
        if show_sphere and self._quadric is not None:
            GL.glPushMatrix()
            GL.glTranslatef(*map(float, origin))
            GLU.gluSphere(self._quadric, self.gizmo.sphere_radius, 16, 10)
            GL.glPopMatrix()

        for axis, delta in (("x", (self.gizmo.arrow_length, 0.0, 0.0)),
                            ("y", (0.0, self.gizmo.arrow_length, 0.0)),
                            ("z", (0.0, 0.0, self.gizmo.arrow_length))):
            if axis not in translation_axes:
                continue
            self._draw_gizmo_arrow(
                origin,
                axis,
                self._gizmo_color(axis, "TRANSLATE", colors[axis]),
            )

        self._begin_transparent_pass()
        GL.glLineWidth(4.0 if self.gizmo.is_dragging else 3.2)
        for axis in rotation_axes:
            self._draw_gizmo_ring(
                axis,
                self._gizmo_color(axis, "ROTATE", colors[axis]),
            )
        self._end_transparent_pass()
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _gizmo_color(self, axis, operation, base_color):
        state_name = self.gizmo.state.name
        if (
            state_name.startswith("HOVER_")
            and operation in state_name
            and state_name.endswith(axis.upper())
        ):
            return (1.0, 0.9, 0.15)
        return base_color

    def _sync_gizmo_screen_scale(self):
        self.configure_camera()
        origin = np.asarray(self.project_point(*self.gizmo.position), dtype=float)
        sample = 0.1
        pixels_per_unit = []
        for vector in ((sample, 0.0, 0.0), (0.0, sample, 0.0), (0.0, 0.0, sample)):
            point = self.gizmo.position + np.asarray(vector, dtype=float)
            distance = float(np.linalg.norm(np.asarray(self.project_point(*point)) - origin))
            if distance > 1e-6:
                pixels_per_unit.append(distance / sample)
        if pixels_per_unit:
            self.gizmo.set_screen_scale(1.0 / float(np.median(pixels_per_unit)))

    def _draw_gizmo_arrow(self, origin, axis, color):
        if self._quadric is None:
            return
        vector = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }[axis]
        shaft_start = self.gizmo.sphere_radius * 1.25
        cone_length = min(self.gizmo.arrow_length * 0.32, self.gizmo.sphere_radius * 1.85)
        shaft_length = max(0.01, self.gizmo.arrow_length - shaft_start - cone_length)
        shaft_radius = max(self.gizmo.sphere_radius * 0.18, self.gizmo.arrow_length * 0.018)
        cone_radius = max(self.gizmo.sphere_radius * 0.42, shaft_radius * 2.2)

        GL.glColor3f(*color)
        GL.glPushMatrix()
        GL.glTranslatef(*map(float, np.asarray(origin) + vector * shaft_start))
        self._orient_z_to_axis(axis)
        GLU.gluCylinder(self._quadric, shaft_radius, shaft_radius, shaft_length, 12, 1)
        GL.glTranslatef(0.0, 0.0, shaft_length)
        GLU.gluCylinder(self._quadric, cone_radius, 0.0, cone_length, 18, 1)
        GL.glPopMatrix()

    def _draw_gizmo_ring(self, axis, color):
        eye = self._camera_eye()
        origin = np.asarray(self.gizmo.position, dtype=float)
        points = self.gizmo.ring_points(axis)
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
            facing = float(np.dot(midpoint - origin, eye - origin))
            alpha = 0.88 if facing >= 0.0 else 0.32
            GL.glColor4f(float(color[0]), float(color[1]), float(color[2]), alpha)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3fv(start)
            GL.glVertex3fv(end)
            GL.glEnd()

    def _draw_gizmo_drag_guide(self, colors):
        if not self.gizmo.is_dragging:
            return
        state = self.gizmo.state
        axis = getattr(self.gizmo, "_drag_axis", None)
        if axis not in colors:
            return
        self._begin_transparent_pass()
        try:
            color = colors[axis]
            GL.glColor4f(float(color[0]), float(color[1]), float(color[2]), 0.28)
            GL.glLineWidth(2.0)
            if "TRANSLATE" in state.name:
                vector = {
                    "x": np.array([1.0, 0.0, 0.0]),
                    "y": np.array([0.0, 1.0, 0.0]),
                    "z": np.array([0.0, 0.0, 1.0]),
                }[axis]
                origin = np.asarray(self.gizmo.position, dtype=float)
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3fv(origin - vector * 2.0)
                GL.glVertex3fv(origin + vector * 2.0)
                GL.glEnd()
            elif "ROTATE" in state.name:
                GL.glLineWidth(6.0)
                self._draw_gizmo_ring(axis, color)
        finally:
            self._end_transparent_pass()

    @staticmethod
    def _orient_z_to_axis(axis):
        if axis == "x":
            GL.glRotatef(90.0, 0.0, 1.0, 0.0)
        elif axis == "y":
            GL.glRotatef(-90.0, 1.0, 0.0, 0.0)

    def _camera_eye(self):
        return self.camera_controller.eye()

    def _camera_offset(self):
        return self.camera_controller.offset()

    def _camera_basis(self):
        return self.camera_controller.basis()

    def _orbit_camera(self, dx, dy):
        self.camera_controller.orbit(dx, dy)

    def _pan_camera(self, dx, dy):
        self.camera_controller.pan(
            dx,
            dy,
            viewport_height=self.height(),
        )

    def _zoom_camera(self, amount):
        self.camera_controller.zoom(amount)

    # ============================================================
    # Mouse editing
    # ============================================================

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.last_mouse_pos = event.position()

        if event.button() == Qt.MouseButton.LeftButton:
            if (
                self.transform_gizmo_visible
                and self.transform_gizmo_interactive
            ):
                self._sync_gizmo_screen_scale()
                if self.gizmo.begin_drag(
                    event.position().x(),
                    event.position().y(),
                    self.project_point,
                    self.screen_ray,
                ):
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.update()
                    return
            self.rotating_camera = True
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.panning_camera = True
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.zooming_camera = True
            return

        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            body_name = self.pick_robot_body(
                event.position().x(), event.position().y()
            )
            if body_name:
                self.body_double_clicked.emit(body_name)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.gizmo.is_dragging:
            self._sync_gizmo_screen_scale()
            modifiers = event.modifiers()
            position, quaternion = self.gizmo.drag(
                event.position().x(),
                event.position().y(),
                self.project_point,
                self.screen_ray,
                fine=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
                snap=bool(modifiers & Qt.KeyboardModifier.ControlModifier),
            )
            self.target_x, self.target_y, self.target_z = map(float, position)
            self.target_transform_dragged.emit(position, quaternion)
            self.update()
            return

        if self.rotating_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self._orbit_camera(delta.x(), delta.y())
            self.last_mouse_pos = event.position()
            self.update()
            return

        if self.panning_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self._pan_camera(delta.x(), delta.y())
            self.last_mouse_pos = event.position()
            self.update()
            return

        if self.zooming_camera and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self._zoom_camera(-delta.y() * 0.02)
            self.last_mouse_pos = event.position()
            self.update()
            return

        if (
            self.transform_gizmo_visible
            and self.transform_gizmo_interactive
        ):
            old_state = self.gizmo.state
            self._sync_gizmo_screen_scale()
            new_state = self.gizmo.hover(
                event.position().x(), event.position().y(), self.project_point
            )
            if new_state != old_state:
                self.setCursor(
                    Qt.CursorShape.OpenHandCursor
                    if new_state != GizmoInteractionState.NONE
                    else Qt.CursorShape.ArrowCursor
                )
                self.update()
        elif self.gizmo.state != GizmoInteractionState.NONE:
            self.gizmo.end_drag()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        transform_was_dragging = self.gizmo.is_dragging
        camera_was_interacting = (
            self.rotating_camera or self.panning_camera or self.zooming_camera
        )
        self.dragging_target = False
        self.rotating_camera = False
        self.panning_camera = False
        self.zooming_camera = False
        self.last_mouse_pos = None
        self.gizmo.end_drag()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        if transform_was_dragging:
            self.transform_drag_finished.emit()
            return
        if camera_was_interacting:
            self.camera_changed.emit()
            return
        super().mouseReleaseEvent(event)

    def cancel_transform_drag(self, emit_cancelled=False):
        """Cancel interaction state without emitting a completed edit."""
        self.gizmo.end_drag()
        self.dragging_target = False
        self.rotating_camera = False
        self.panning_camera = False
        self.zooming_camera = False
        self.last_mouse_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        if emit_cancelled:
            self.transform_drag_cancel_requested.emit()

    def set_gizmo_mode(self, mode):
        """Set the transform interaction mode and keep external controls in sync."""
        self.gizmo.set_mode(mode)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.gizmo_mode_changed.emit(mode)
        self.update()

    def keyPressEvent(self, event):
        if (
            self.transform_gizmo_visible
            and self.transform_gizmo_interactive
            and event.key() == Qt.Key.Key_T
        ):
            self.set_gizmo_mode("translate")
            event.accept()
            return
        if (
            self.transform_gizmo_visible
            and self.transform_gizmo_interactive
            and event.key() == Qt.Key.Key_R
        ):
            self.set_gizmo_mode("rotate")
            event.accept()
            return
        if event.key() in (Qt.Key.Key_E, Qt.Key.Key_Escape):
            self.cancel_transform_drag(emit_cancelled=True)
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self._zoom_camera(steps * 0.35)
        self.update()
        self.camera_changed.emit()

    # ============================================================
    # Projection helpers
    # ============================================================

    def project_point(self, x, y, z):
        self.configure_camera()
        point = QVector3D(x, y, z)
        screen = point.project(self._model_view, self._projection, self._viewport)
        return screen.x(), self.height() - screen.y()

    def screen_ray(self, sx, sy):
        self.configure_camera()
        near = QVector3D(sx, self.height() - sy, 0.0).unproject(
            self._model_view, self._projection, self._viewport
        )
        far = QVector3D(sx, self.height() - sy, 1.0).unproject(
            self._model_view, self._projection, self._viewport
        )
        origin = np.array([near.x(), near.y(), near.z()], dtype=float)
        direction = np.array(
            [far.x() - near.x(), far.y() - near.y(), far.z() - near.z()],
            dtype=float,
        )
        direction /= max(1e-12, float(np.linalg.norm(direction)))
        return origin, direction

    def pick_robot_body(self, sx, sy):
        return self.pick_robot_body_from_ray(*self.screen_ray(sx, sy))

    def pick_robot_body_from_ray(self, origin, direction):
        """Return the closest MuJoCo body hit by geom bounding spheres."""
        if self.robot_state is None:
            return None
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction /= max(1e-12, float(np.linalg.norm(direction)))
        model = self.robot_state.mj_model
        data = (
            self.preview_state.mj_data
            if self.preview_visible and self.preview_state is not None
            else self.robot_state.mj_data
        )
        best = None
        for geom_id in self.render_geom_ids(model):
            body_id = int(model.geom_bodyid[geom_id])
            if body_id == 0:
                continue
            center = np.asarray(data.geom_xpos[geom_id], dtype=float)
            radius = max(0.015, float(model.geom_rbound[geom_id]))
            offset = origin - center
            b = float(np.dot(offset, direction))
            c = float(np.dot(offset, offset) - radius * radius)
            discriminant = b * b - c
            if discriminant < 0.0:
                continue
            root = math.sqrt(discriminant)
            distance = -b - root
            if distance < 0.0:
                distance = -b + root
            if distance < 0.0:
                continue
            closest = origin + direction * max(0.0, -b)
            ray_distance = float(np.linalg.norm(closest - center))
            # Large imported mesh bounds can enclose nearby wrist/tool parts.
            # Bucket the center-line distance so tiny symmetry differences still
            # use front-most depth, while obvious center hits beat oversized bounds.
            ray_distance_bucket = int(ray_distance / 0.005)
            if best is None or (ray_distance_bucket, distance) < best[:2]:
                best = (ray_distance_bucket, distance, body_id)
        if best is None:
            return None
        import mujoco
        return mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, best[2]
        )

    def screen_to_edit_plane(self, sx, sy):
        """
        Convert a screen point to the Y=0 edit plane.

        The reference-frame editor currently exposes X/Z controls, so the 3D
        editor drags the target within that same side-view plane.
        """

        self.configure_camera()

        near = QVector3D(sx, self.height() - sy, 0.0).unproject(
            self._model_view,
            self._projection,
            self._viewport,
        )
        far = QVector3D(sx, self.height() - sy, 1.0).unproject(
            self._model_view,
            self._projection,
            self._viewport,
        )

        direction = far - near
        if abs(direction.y()) < 1e-6:
            return self.target_x, self.target_z

        t = -near.y() / direction.y()
        point = near + direction * t
        return point.x(), point.z()
