"""Lightweight weighted IK task representations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass
class TaskLinearization:
    error: np.ndarray
    jacobian: np.ndarray
    error_norm: float
    tolerance: float
    required: bool


@dataclass
class IKTask:
    name: str
    weight: float = 1.0
    priority: int = 2
    enabled: bool = True
    required: bool = True
    tolerance: float = 0.005

    def linearize(self, model, data, dof_addresses, qpos_addresses):
        raise NotImplementedError

    def _scaled(self, error, jacobian, error_norm, normalization=1.0):
        # Aggregate tasks such as posture have one row per joint. Normalizing
        # their total weight keeps the same UI value from becoming stronger
        # merely because a robot has more controllable joints.
        scale = math.sqrt(
            max(0.0, float(self.weight)) / max(1.0, float(normalization))
        )
        return TaskLinearization(
            np.asarray(error, dtype=float) * scale,
            np.asarray(jacobian, dtype=float) * scale,
            float(error_norm),
            float(self.tolerance),
            bool(self.required),
        )


def _object_id(model, kind, name):
    object_type = (
        mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
    )
    return mujoco.mj_name2id(model, object_type, name)


def _pose_and_jacobian(model, data, kind, name, dof_addresses):
    object_id = _object_id(model, kind, name)
    if object_id < 0:
        raise KeyError(f"Unknown MuJoCo {kind}: {name}")
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    if kind == "site":
        mujoco.mj_jacSite(model, data, jacp, jacr, object_id)
        position = data.site_xpos[object_id].copy()
        rotation = data.site_xmat[object_id].reshape(3, 3).copy()
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, object_id)
        position = data.xpos[object_id].copy()
        rotation = data.xmat[object_id].reshape(3, 3).copy()
    return position, rotation, jacp[:, dof_addresses], jacr[:, dof_addresses]


@dataclass
class TCPPositionTask(IKTask):
    object_name: str = ""
    kind: str = "site"
    target_position: object = None

    def linearize(self, model, data, dof_addresses, qpos_addresses):
        position, _, jacp, _ = _pose_and_jacobian(
            model, data, self.kind, self.object_name, dof_addresses
        )
        error = np.asarray(self.target_position, dtype=float) - position
        return self._scaled(error, jacp, np.linalg.norm(error))


@dataclass
class TCPOrientationTask(IKTask):
    object_name: str = ""
    kind: str = "site"
    target_quaternion: object = None

    def linearize(self, model, data, dof_addresses, qpos_addresses):
        _, rotation, _, jacr = _pose_and_jacobian(
            model, data, self.kind, self.object_name, dof_addresses
        )
        target_rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(
            target_rotation, np.asarray(self.target_quaternion, dtype=float)
        )
        target_rotation = target_rotation.reshape(3, 3)
        error = 0.5 * sum(
            np.cross(rotation[:, axis], target_rotation[:, axis])
            for axis in range(3)
        )
        relative = rotation.T @ target_rotation
        angle = math.acos(float(np.clip(
            (np.trace(relative) - 1.0) * 0.5, -1.0, 1.0
        )))
        return self._scaled(error, jacr, angle)


@dataclass
class BodyPoseTask(IKTask):
    object_name: str = ""
    kind: str = "body"
    target_position: object = None
    target_quaternion: object = None

    def linearize(self, model, data, dof_addresses, qpos_addresses):
        position_task = TCPPositionTask(
            name=self.name + " position", weight=self.weight,
            priority=self.priority, enabled=self.enabled, required=self.required,
            tolerance=self.tolerance, object_name=self.object_name,
            kind=self.kind, target_position=self.target_position,
        ).linearize(model, data, dof_addresses, qpos_addresses)
        orientation_task = TCPOrientationTask(
            name=self.name + " orientation", weight=self.weight,
            priority=self.priority, enabled=self.enabled, required=self.required,
            tolerance=self.tolerance, object_name=self.object_name,
            kind=self.kind, target_quaternion=self.target_quaternion,
        ).linearize(model, data, dof_addresses, qpos_addresses)
        return TaskLinearization(
            np.concatenate((position_task.error, orientation_task.error)),
            np.vstack((position_task.jacobian, orientation_task.jacobian)),
            max(position_task.error_norm, orientation_task.error_norm),
            self.tolerance,
            self.required,
        )


@dataclass
class FootLockTask(TCPPositionTask):
    priority: int = 1


@dataclass
class RootPoseTask(TCPOrientationTask):
    priority: int = 1


@dataclass
class PostureTask(IKTask):
    reference_qpos: object = None

    def linearize(self, model, data, dof_addresses, qpos_addresses):
        reference = np.asarray(self.reference_qpos, dtype=float)[qpos_addresses]
        current = data.qpos[qpos_addresses]
        error = reference - current
        joint_count = len(qpos_addresses)
        return self._scaled(
            error,
            np.eye(joint_count),
            np.linalg.norm(error),
            normalization=joint_count,
        )


@dataclass
class JointRegularizationTask(PostureTask):
    pass
