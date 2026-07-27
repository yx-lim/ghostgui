"""Persistent world-axis transform gizmo interaction and quaternion math."""

from __future__ import annotations

from enum import Enum, auto
import math

import numpy as np

from core.math3d import normalize_quaternion, quaternion_slerp

class GizmoInteractionState(Enum):
    NONE = auto()
    HOVER_TRANSLATE_FREE = auto()
    DRAG_TRANSLATE_FREE = auto()
    HOVER_TRANSLATE_X = auto()
    HOVER_TRANSLATE_Y = auto()
    HOVER_TRANSLATE_Z = auto()
    DRAG_TRANSLATE_X = auto()
    DRAG_TRANSLATE_Y = auto()
    DRAG_TRANSLATE_Z = auto()
    HOVER_ROTATE_X = auto()
    HOVER_ROTATE_Y = auto()
    HOVER_ROTATE_Z = auto()
    DRAG_ROTATE_X = auto()
    DRAG_ROTATE_Y = auto()
    DRAG_ROTATE_Z = auto()


AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}
GIZMO_MODES = ("translate", "rotate")

HOVER_TRANSLATE = {
    "x": GizmoInteractionState.HOVER_TRANSLATE_X,
    "y": GizmoInteractionState.HOVER_TRANSLATE_Y,
    "z": GizmoInteractionState.HOVER_TRANSLATE_Z,
}
DRAG_TRANSLATE = {
    "x": GizmoInteractionState.DRAG_TRANSLATE_X,
    "y": GizmoInteractionState.DRAG_TRANSLATE_Y,
    "z": GizmoInteractionState.DRAG_TRANSLATE_Z,
}
HOVER_ROTATE = {
    "x": GizmoInteractionState.HOVER_ROTATE_X,
    "y": GizmoInteractionState.HOVER_ROTATE_Y,
    "z": GizmoInteractionState.HOVER_ROTATE_Z,
}
DRAG_ROTATE = {
    "x": GizmoInteractionState.DRAG_ROTATE_X,
    "y": GizmoInteractionState.DRAG_ROTATE_Y,
    "z": GizmoInteractionState.DRAG_ROTATE_Z,
}


def quaternion_multiply(left, right):
    w1, x1, y1, z1 = normalize_quaternion(left)
    w2, x2, y2, z2 = normalize_quaternion(right)
    return normalize_quaternion(np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]))


def axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= max(1e-12, float(np.linalg.norm(axis)))
    half = 0.5 * angle
    return np.array([math.cos(half), *(axis * math.sin(half))])


def _point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared < 1e-9:
        return float(np.linalg.norm(point - start))
    amount = min(1.0, max(0.0, float(np.dot(point - start, segment) / length_squared)))
    return float(np.linalg.norm(point - (start + amount * segment)))


class TransformGizmo:
    """World-space arrows/rings with precise screen-space handle picking."""

    def __init__(self, position=(0.0, 0.0, 0.9), quaternion=(1.0, 0.0, 0.0, 0.0)):
        self.position = np.asarray(position, dtype=float)
        self.quaternion = normalize_quaternion(quaternion)
        self.coordinate_mode = "world"
        self.arrow_length = 0.2
        self.sphere_radius = 0.035
        self.ring_radius = 0.15
        self.pick_tolerance_pixels = 7.0
        self.mode = "translate"
        self.state = GizmoInteractionState.NONE
        self._drag_axis = None
        self._drag_position = None
        self._drag_quaternion = None
        self._drag_mouse = None
        self._drag_value = None
        self._rotation_start_vector = None
        self._free_plane_normal = None
        self._free_start_intersection = None

    def set_pose(self, position, quaternion=None):
        self.position = np.asarray(position, dtype=float).copy()
        if quaternion is not None:
            self.quaternion = normalize_quaternion(quaternion)

    def set_mode(self, mode):
        if mode not in GIZMO_MODES:
            raise ValueError(f"Unknown transform gizmo mode: {mode}")
        if self.mode != mode:
            self.end_drag()
            self.mode = mode

    def set_screen_scale(self, world_units_per_pixel):
        world_units_per_pixel = max(1e-6, float(world_units_per_pixel))
        self.arrow_length = min(0.55, max(0.12, 96.0 * world_units_per_pixel))
        self.sphere_radius = min(0.08, max(0.018, 12.0 * world_units_per_pixel))
        self.ring_radius = min(0.42, max(0.10, 74.0 * world_units_per_pixel))
        self.pick_tolerance_pixels = 11.0

    def ring_points(self, axis, count=64):
        first, second = {
            "x": (AXES["y"], AXES["z"]),
            "y": (AXES["x"], AXES["z"]),
            "z": (AXES["x"], AXES["y"]),
        }[axis]
        return [
            self.position + self.ring_radius * (
                math.cos(angle) * first + math.sin(angle) * second
            )
            for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        ]

    def pick(self, sx, sy, project):
        pointer = (float(sx), float(sy))
        origin = project(*self.position)
        show_sphere, translation_axes, rotation_axes = self.visible_handles()
        projected_radius = (
            max(
                math.dist(origin, project(*(self.position + axis * self.sphere_radius)))
                for axis in AXES.values()
            )
            if show_sphere else 0.0
        )
        sphere_pick_radius = min(18.0, max(6.0, projected_radius + 2.0))
        if show_sphere and math.dist(pointer, origin) <= sphere_pick_radius:
            return GizmoInteractionState.HOVER_TRANSLATE_FREE, "free"

        candidates = []
        for axis, vector in AXES.items():
            if axis not in translation_axes:
                continue
            endpoint = project(*(self.position + vector * self.arrow_length))
            shaft_start = project(*(
                self.position + vector * self.sphere_radius * 1.25
            ))
            distance = _point_segment_distance(pointer, shaft_start, endpoint)
            if distance <= self.pick_tolerance_pixels:
                candidates.append((0, distance, HOVER_TRANSLATE[axis], axis))
        for axis in rotation_axes:
            points = [project(*point) for point in self.ring_points(axis)]
            distance = min(
                _point_segment_distance(pointer, points[index], points[(index + 1) % len(points)])
                for index in range(len(points))
            )
            ring_tolerance = self.pick_tolerance_pixels + 4.0
            if distance <= ring_tolerance:
                candidates.append((1, distance, HOVER_ROTATE[axis], axis))
        if not candidates:
            return GizmoInteractionState.NONE, None
        _, _, state, axis = min(candidates, key=lambda item: (item[0], item[1]))
        return state, axis

    def hover(self, sx, sy, project):
        if self.is_dragging:
            return self.state
        self.state, _ = self.pick(sx, sy, project)
        return self.state

    @property
    def is_dragging(self):
        return self.state in (
            {GizmoInteractionState.DRAG_TRANSLATE_FREE}
            | set(DRAG_TRANSLATE.values())
            | set(DRAG_ROTATE.values())
        )

    def visible_handles(self):
        """Return the center, translation axes, and rotation axes to draw."""
        translation_axes = tuple(AXES) if self.mode == "translate" else ()
        rotation_axes = tuple(AXES) if self.mode == "rotate" else ()
        show_sphere = self.mode == "translate"
        if not self.is_dragging:
            return show_sphere, translation_axes, rotation_axes
        if self.state == GizmoInteractionState.DRAG_TRANSLATE_FREE:
            return True, (), ()
        if self.state in DRAG_TRANSLATE.values():
            return False, (self._drag_axis,), ()
        return False, (), (self._drag_axis,)

    def begin_drag(self, sx, sy, project, screen_ray):
        hover_state, axis = self.pick(sx, sy, project)
        if axis is None:
            self.state = GizmoInteractionState.NONE
            return False
        self._drag_axis = axis
        self._drag_position = self.position.copy()
        self._drag_quaternion = self.quaternion.copy()
        self._drag_mouse = np.array([sx, sy], dtype=float)
        self._drag_value = None
        if hover_state == GizmoInteractionState.HOVER_TRANSLATE_FREE:
            ray = screen_ray(sx, sy)
            self._free_plane_normal = np.asarray(ray[1], dtype=float)
            self._free_plane_normal /= max(
                1e-12, float(np.linalg.norm(self._free_plane_normal))
            )
            self._free_start_intersection = self._ray_plane_intersection(
                ray, self._free_plane_normal, plane_point=self._drag_position
            )
            if self._free_start_intersection is None:
                self.state = GizmoInteractionState.NONE
                return False
            self.state = GizmoInteractionState.DRAG_TRANSLATE_FREE
            return True
        if hover_state in HOVER_TRANSLATE.values():
            self.state = DRAG_TRANSLATE[axis]
            return True
        intersection = self._ray_plane_intersection(screen_ray(sx, sy), AXES[axis])
        if intersection is None:
            self.state = GizmoInteractionState.NONE
            return False
        vector = intersection - self.position
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            self.state = GizmoInteractionState.NONE
            return False
        self._rotation_start_vector = vector / norm
        self.state = DRAG_ROTATE[axis]
        return True

    def drag(self, sx, sy, project, screen_ray, fine=False, snap=False):
        if not self.is_dragging:
            return self.position.copy(), self.quaternion.copy()
        fine_scale = 0.25 if fine else 1.0
        if self.state == GizmoInteractionState.DRAG_TRANSLATE_FREE:
            intersection = self._ray_plane_intersection(
                screen_ray(sx, sy),
                self._free_plane_normal,
                plane_point=self._drag_position,
            )
            if intersection is not None:
                delta = (intersection - self._free_start_intersection) * fine_scale
                if snap:
                    snap_size = 0.001 if fine else 0.01
                    delta = np.round(delta / snap_size) * snap_size
                self.position = self._drag_position + delta
                self.position[2] = max(0.0, self.position[2])
                self._drag_value = self.position - self._drag_position
        elif self.state in DRAG_TRANSLATE.values():
            axis = AXES[self._drag_axis]
            start_screen = np.asarray(project(*self._drag_position), dtype=float)
            end_screen = np.asarray(
                project(*(self._drag_position + axis * self.arrow_length)), dtype=float
            )
            screen_axis = end_screen - start_screen
            length_squared = float(np.dot(screen_axis, screen_axis))
            if length_squared > 1e-8:
                mouse_delta = np.array([sx, sy], dtype=float) - self._drag_mouse
                world_delta = self.arrow_length * float(
                    np.dot(mouse_delta, screen_axis) / length_squared
                ) * fine_scale
                if snap:
                    snap_size = 0.001 if fine else 0.01
                    world_delta = round(world_delta / snap_size) * snap_size
                self.position = self._drag_position + axis * world_delta
                self.position[2] = max(0.0, self.position[2])
                self._drag_value = world_delta
        else:
            axis = AXES[self._drag_axis]
            intersection = self._ray_plane_intersection(screen_ray(sx, sy), axis)
            if intersection is not None:
                vector = intersection - self._drag_position
                norm = float(np.linalg.norm(vector))
                if norm > 1e-8:
                    current = vector / norm
                    angle = math.atan2(
                        float(np.dot(axis, np.cross(self._rotation_start_vector, current))),
                        float(np.dot(self._rotation_start_vector, current)),
                    ) * fine_scale
                    if snap:
                        snap_angle = math.radians(1.0 if fine else 5.0)
                        angle = round(angle / snap_angle) * snap_angle
                    delta = axis_angle_quaternion(axis, angle)
                    self.quaternion = quaternion_multiply(delta, self._drag_quaternion)
                    self._drag_value = angle
        return self.position.copy(), self.quaternion.copy()

    def drag_status(self):
        if self._drag_value is None or self._drag_axis is None:
            return ""
        if self.state == GizmoInteractionState.DRAG_TRANSLATE_FREE:
            dx, dy, dz = (float(value) for value in self._drag_value)
            return f"Move {dx:+.3f}, {dy:+.3f}, {dz:+.3f} m"
        if self.state in DRAG_TRANSLATE.values():
            return f"{self._drag_axis.upper()} {float(self._drag_value):+.3f} m"
        return f"Rot {self._drag_axis.upper()} {math.degrees(float(self._drag_value)):+.1f} deg"

    def end_drag(self):
        self.state = GizmoInteractionState.NONE
        self._drag_axis = None
        self._drag_value = None
        self._rotation_start_vector = None
        self._free_plane_normal = None
        self._free_start_intersection = None

    def _ray_plane_intersection(self, ray, normal, plane_point=None):
        origin, direction = (np.asarray(value, dtype=float) for value in ray)
        plane_point = self.position if plane_point is None else np.asarray(
            plane_point, dtype=float
        )
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-8:
            return None
        distance = float(np.dot(normal, plane_point - origin) / denominator)
        if distance < 0.0:
            return None
        return origin + distance * direction
