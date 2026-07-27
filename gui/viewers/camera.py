"""Pure orbit-camera state and navigation math."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class OrbitCamera:
    distance: float = 5.0
    yaw: float = 38.0
    pitch: float = 24.0
    center: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.75], dtype=float)
    )

    def offset(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        return np.array(
            [
                self.distance * math.cos(pitch) * math.sin(yaw),
                -self.distance * math.cos(pitch) * math.cos(yaw),
                self.distance * math.sin(pitch),
            ],
            dtype=float,
        )

    def eye(self):
        return self.center + self.offset()

    def basis(self):
        forward = self.center - self.eye()
        forward /= max(1e-12, float(np.linalg.norm(forward)))
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= float(np.linalg.norm(right))
        up = np.cross(right, forward)
        up /= max(1e-12, float(np.linalg.norm(up)))
        return right, up, forward

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= float(dx) * 0.4
        self.pitch = max(-85.0, min(85.0, self.pitch + float(dy) * 0.3))

    def pan(
        self,
        dx: float,
        dy: float,
        *,
        viewport_height: int,
        vertical_fov_degrees: float = 45.0,
    ) -> None:
        right, up, _ = self.basis()
        view_height = (
            2.0
            * self.distance
            * math.tan(math.radians(vertical_fov_degrees) * 0.5)
        )
        units_per_pixel = view_height / max(1, int(viewport_height))
        self.center += (
            -right * float(dx) * units_per_pixel
            + up * float(dy) * units_per_pixel
        )

    def zoom(self, amount: float) -> None:
        self.distance = max(
            0.5,
            min(10.0, self.distance + float(amount)),
        )
