"""Shared weighted pose-IK orchestration for interactive and batch workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.math3d import quaternion_angle, rpy_to_quaternion
from .tasks import TCPOrientationTask, TCPPositionTask


ROOT_FRAME_NAMES = frozenset({"pelvis", "base", "root"})


@dataclass(frozen=True)
class IKSolverSettings:
    max_iterations: int = 80
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.03
    orientation_weight: float = 0.25
    damping: float = 0.04
    step_size: float = 0.7
    max_step: float = 0.08

    def __post_init__(self):
        if int(self.max_iterations) <= 0:
            raise ValueError("IK max_iterations must be positive")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        for name in (
            "position_tolerance",
            "orientation_tolerance",
            "damping",
            "step_size",
            "max_step",
        ):
            value = float(getattr(self, name))
            finite_required = name != "orientation_tolerance"
            if (
                value <= 0.0
                or math.isnan(value)
                or (finite_required and not math.isfinite(value))
            ):
                raise ValueError(f"IK {name} must be positive and finite")
            object.__setattr__(self, name, value)
        orientation_weight = float(self.orientation_weight)
        if not math.isfinite(orientation_weight) or orientation_weight < 0.0:
            raise ValueError("IK orientation_weight must be finite and nonnegative")
        object.__setattr__(self, "orientation_weight", orientation_weight)

    @classmethod
    def from_mapping(cls, values=None, **overrides):
        values = dict(values or {})
        values.update(
            {
                key: value
                for key, value in overrides.items()
                if value is not None
            }
        )
        known = {
            field_name
            for field_name in cls.__dataclass_fields__
        }
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True)
class PoseIKResult:
    ik_result: object
    position_error: float
    orientation_error: float
    solved_frame_names: tuple[str, ...]
    ignored_frame_names: tuple[str, ...]


def solve_pose_targets(
    state,
    active_targets,
    frame_bindings,
    *,
    frame_weights=None,
    joint_weights=None,
    settings=None,
):
    """Solve logical target frames through ``RobotState3D.solve_weighted_tasks``."""
    settings = settings or IKSolverSettings()
    frame_weights = dict(frame_weights or {})
    tasks = []
    specifications = []
    ignored = []
    for frame_name, target in active_targets.items():
        if frame_name in ROOT_FRAME_NAMES:
            continue
        binding = frame_bindings.get(frame_name)
        if binding is None:
            ignored.append(frame_name)
            continue
        kind, object_name = binding
        weight = max(0.0, float(frame_weights.get(frame_name, 1.0)))
        target_position = np.asarray(
            [target.x, target.y, target.z],
            dtype=float,
        )
        target_quaternion = rpy_to_quaternion(
            target.roll,
            target.pitch,
            target.yaw,
        )
        tasks.append(TCPPositionTask(
            name=f"{frame_name} position",
            weight=weight,
            priority=2,
            required=True,
            tolerance=settings.position_tolerance,
            object_name=object_name,
            kind=kind,
            target_position=target_position,
        ))
        if settings.orientation_weight > 0.0:
            tasks.append(TCPOrientationTask(
                name=f"{frame_name} orientation",
                weight=weight * settings.orientation_weight,
                priority=2,
                required=True,
                tolerance=settings.orientation_tolerance,
                object_name=object_name,
                kind=kind,
                target_quaternion=target_quaternion,
            ))
        specifications.append(
            (
                frame_name,
                kind,
                object_name,
                target_position,
                target_quaternion,
            )
        )

    ik_result = state.solve_weighted_tasks(
        tasks,
        joint_weights=joint_weights,
        max_iterations=settings.max_iterations,
        damping=settings.damping,
        step_size=settings.step_size,
        max_step=settings.max_step,
    )
    state.forward_kinematics()
    position_errors = []
    orientation_errors = []
    for (
        _frame_name,
        kind,
        object_name,
        target_position,
        target_quaternion,
    ) in specifications:
        position, quaternion = state.get_body_pose(object_name, kind)
        position_errors.append(
            float(np.linalg.norm(target_position - position))
        )
        if settings.orientation_weight > 0.0:
            orientation_errors.append(
                quaternion_angle(quaternion, target_quaternion)
            )
    return PoseIKResult(
        ik_result=ik_result,
        position_error=max(position_errors, default=0.0),
        orientation_error=max(orientation_errors, default=0.0),
        solved_frame_names=tuple(item[0] for item in specifications),
        ignored_frame_names=tuple(sorted(ignored)),
    )
