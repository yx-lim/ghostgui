"""Frame-pose lookup contracts shared by visualization components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from core.math3d import normalize_quaternion


class FramePoseError(RuntimeError):
    """Raised when a requested visualization frame cannot be resolved."""


@dataclass(frozen=True)
class FramePose:
    frame_name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    reference_frame: str = "world"

    def __post_init__(self):
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("frame position must contain three finite values")
        quaternion = normalize_quaternion(self.quaternion_wxyz)
        object.__setattr__(
            self,
            "position",
            tuple(float(value) for value in position),
        )
        object.__setattr__(
            self,
            "quaternion_wxyz",
            tuple(float(value) for value in quaternion),
        )


@runtime_checkable
class FramePoseProvider(Protocol):
    def frame_names(self) -> tuple[str, ...]:
        """Return logical frame names available in the active scene."""

    def pose(self, frame_name: str) -> FramePose:
        """Return a world pose for one logical frame."""

    def snapshot(self) -> dict[str, FramePose]:
        """Return an internally consistent pose snapshot."""


class RobotFramePoseProvider:
    """Resolve logical frames against the active adapter and MuJoCo state."""

    def __init__(
        self,
        adapter_provider: Callable[[], object | None],
        state_provider: Callable[[], object | None],
    ):
        self._adapter_provider = adapter_provider
        self._state_provider = state_provider

    def _bindings(self) -> dict[str, tuple[str, str]]:
        adapter = self._adapter_provider()
        bindings = getattr(adapter, "logical_frame_bindings", None)
        if not bindings:
            return {}
        return dict(bindings)

    def frame_names(self) -> tuple[str, ...]:
        return tuple(self._bindings())

    def pose(self, frame_name: str) -> FramePose:
        bindings = self._bindings()
        try:
            kind, object_name = bindings[str(frame_name)]
        except KeyError as exc:
            raise FramePoseError(f"unknown logical frame: {frame_name}") from exc
        state = self._state_provider()
        if state is None:
            raise FramePoseError("the active visualization has no robot state")
        try:
            position, quaternion = state.get_body_pose(object_name, kind=kind)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise FramePoseError(
                f"could not resolve logical frame {frame_name}: {exc}"
            ) from exc
        return FramePose(
            frame_name=str(frame_name),
            position=tuple(position),
            quaternion_wxyz=tuple(quaternion),
        )

    def snapshot(self) -> dict[str, FramePose]:
        return {name: self.pose(name) for name in self.frame_names()}
