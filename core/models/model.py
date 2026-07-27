"""MuJoCo model, state, IK, and trajectory abstractions for the live 3D UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from core.resources import bundled_resource_root
from core.robotics import QposContract, validate_trajectory_arrays

try:
    import mujoco
    import numpy as np
except ImportError:  # The rest of GhostGUI remains usable without MuJoCo.
    mujoco = None
    np = None


PROJECT_ROOT = bundled_resource_root()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_id: int
    qpos_address: int
    dof_address: int
    joint_type: int
    limits: Optional[tuple[float, float]]


@dataclass(frozen=True)
class FreeJointInfo:
    name: str
    joint_id: int
    body_id: int
    qpos_address: int
    dof_address: int


@dataclass
class IKResult:
    success: bool
    error: float
    iterations: int
    message: str
    near_singularity: bool = False
    min_singular_value: float = float("inf")
    condition_number: float = 0.0


class RobotModel3D:
    """Immutable robot description loaded once and shared by UI subsystems."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        if mujoco is None or np is None:
            raise RuntimeError("MuJoCo and NumPy are required for the 3D robot viewer.")

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Robot model not found: {self.model_path}")

        self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.joints: dict[str, JointInfo] = {}
        self.free_joints_by_body: dict[int, FreeJointInfo] = {}
        self.body_names = self._names(mujoco.mjtObj.mjOBJ_BODY, self.mj_model.nbody)
        self.site_names = self._names(mujoco.mjtObj.mjOBJ_SITE, self.mj_model.nsite)
        self._discover_joints()
        self.home_qpos = self._load_home_qpos()

    def _names(self, object_type, count: int) -> list[str]:
        names = []
        for object_id in range(count):
            name = mujoco.mj_id2name(self.mj_model, object_type, object_id)
            if name:
                names.append(name)
        return names

    @staticmethod
    def plain_name(name: str) -> str:
        return name[len("robot/"):] if name.startswith("robot/") else name

    def _discover_joints(self) -> None:
        supported = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
        for joint_id in range(self.mj_model.njnt):
            joint_type = int(self.mj_model.jnt_type[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                raw_name = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                ) or f"free_joint_{joint_id}"
                body_id = int(self.mj_model.jnt_bodyid[joint_id])
                self.free_joints_by_body[body_id] = FreeJointInfo(
                    name=self.plain_name(raw_name),
                    joint_id=joint_id,
                    body_id=body_id,
                    qpos_address=int(self.mj_model.jnt_qposadr[joint_id]),
                    dof_address=int(self.mj_model.jnt_dofadr[joint_id]),
                )
                continue
            if joint_type not in supported:
                continue
            raw_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not raw_name:
                continue
            limits = None
            if self.mj_model.jnt_limited[joint_id]:
                lo, hi = self.mj_model.jnt_range[joint_id]
                limits = (float(lo), float(hi))
            name = self.plain_name(raw_name)
            self.joints[name] = JointInfo(
                name=name,
                joint_id=joint_id,
                qpos_address=int(self.mj_model.jnt_qposadr[joint_id]),
                dof_address=int(self.mj_model.jnt_dofadr[joint_id]),
                joint_type=joint_type,
                limits=limits,
            )

    def _load_home_qpos(self):
        data = mujoco.MjData(self.mj_model)
        if self.mj_model.nkey:
            mujoco.mj_resetDataKeyframe(self.mj_model, data, 0)
        else:
            mujoco.mj_resetData(self.mj_model, data)
        qpos = data.qpos.copy()
        # Preserve the model's standing height but make root XY/orientation
        # explicitly neutral for a predictable "origin/default" reset.
        for joint in self.free_joints_by_body.values():
            address = joint.qpos_address
            qpos[address:address + 2] = 0.0
            qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
        return qpos

    def free_joint_for_body(self, body_id: int) -> Optional[FreeJointInfo]:
        return self.free_joints_by_body.get(int(body_id))

    def get_geom_rgba(self, geom_id: int):
        """Resolve the color MuJoCo renders, including assigned materials."""
        geom_id = int(geom_id)
        material_id = int(self.mj_model.geom_matid[geom_id])
        if material_id >= 0:
            return self.mj_model.mat_rgba[material_id].copy()
        return self.mj_model.geom_rgba[geom_id].copy()

    def get_visual_texture_warnings(self):
        """Describe textured visual materials unsupported by the custom GL path."""
        warnings = []
        seen = set()
        for geom_id in range(self.mj_model.ngeom):
            if int(self.mj_model.geom_group[geom_id]) != 2:
                continue
            material_id = int(self.mj_model.geom_matid[geom_id])
            if material_id < 0 or material_id in seen:
                continue
            seen.add(material_id)
            texture_ids = np.asarray(self.mj_model.mat_texid[material_id]).ravel()
            if np.any(texture_ids >= 0):
                material_name = mujoco.mj_id2name(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_MATERIAL,
                    material_id,
                ) or f"material#{material_id}"
                warnings.append(
                    f"{material_name} uses a texture; live OpenGL falls back "
                    "to its material RGBA"
                )
        return warnings

    def create_state(self) -> "RobotState3D":
        return RobotState3D(self)

    def get_joint_names(self) -> list[str]:
        return list(self.joints)

    def get_joint_limits(self, joint_name: str) -> Optional[tuple[float, float]]:
        return self._joint(joint_name).limits

    def _joint(self, name: str) -> JointInfo:
        plain = self.plain_name(name)
        if plain not in self.joints:
            raise KeyError(f"Unknown controllable joint: {name}")
        return self.joints[plain]


class RobotState3D:
    """Mutable qpos and FK state backed by one persistent ``MjData`` object."""

    SINGULARITY_MIN_SINGULAR_VALUE = 1e-4
    SINGULARITY_CONDITION_NUMBER = 1e3

    def __init__(self, robot_model: RobotModel3D):
        self.robot_model = robot_model
        self.mj_model = robot_model.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        self.set_qpos(robot_model.home_qpos)

    def reset_to_default(self) -> None:
        self.set_qpos(self.robot_model.home_qpos)

    def get_joint_names(self) -> list[str]:
        return self.robot_model.get_joint_names()

    def get_joint_limits(self, joint_name: str) -> Optional[tuple[float, float]]:
        return self.robot_model.get_joint_limits(joint_name)

    def get_joint_value(self, joint_name: str) -> float:
        joint = self.robot_model._joint(joint_name)
        return float(self.mj_data.qpos[joint.qpos_address])

    def set_joint_value(self, joint_name: str, value: float, run_fk: bool = True) -> None:
        joint = self.robot_model._joint(joint_name)
        if joint.limits is not None:
            value = float(np.clip(value, *joint.limits))
        self.mj_data.qpos[joint.qpos_address] = value
        if run_fk:
            self.forward_kinematics()

    def set_joint_values(
        self, values: Mapping[str, float] | Iterable[float], run_fk: bool = True
    ) -> None:
        if isinstance(values, Mapping):
            items = values.items()
        else:
            array = np.asarray(list(values), dtype=float)
            if array.shape == (self.mj_model.nq,):
                self.set_qpos(array)
                return
            names = self.get_joint_names()
            if array.shape != (len(names),):
                raise ValueError(
                    f"Expected {len(names)} joint values or {self.mj_model.nq} qpos values."
                )
            items = zip(names, array)

        for name, value in items:
            self.set_joint_value(name, float(value), run_fk=False)
        if run_fk:
            self.forward_kinematics()

    def set_qpos(self, qpos: Iterable[float]) -> None:
        qpos = QposContract(self.mj_model.nq).validate(qpos)
        self.mj_data.qpos[:] = qpos
        self._clamp_joints()
        self.forward_kinematics()

    def get_qpos(self):
        return self.mj_data.qpos.copy()

    def forward_kinematics(self) -> None:
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def resolve_object(self, name: str, kind: Optional[str] = None):
        candidates = []
        if kind in (None, "site"):
            candidates.append(("site", mujoco.mjtObj.mjOBJ_SITE))
        if kind in (None, "body"):
            candidates.append(("body", mujoco.mjtObj.mjOBJ_BODY))
        for candidate_kind, object_type in candidates:
            object_id = mujoco.mj_name2id(self.mj_model, object_type, name)
            if object_id >= 0:
                return candidate_kind, object_id
        raise KeyError(f"Unknown MuJoCo body/site: {name}")

    def get_body_pose(self, body_name: str, kind: Optional[str] = None):
        kind, object_id = self.resolve_object(body_name, kind)
        if kind == "site":
            position = self.mj_data.site_xpos[object_id].copy()
            rotation = self.mj_data.site_xmat[object_id].copy()
        else:
            position = self.mj_data.xpos[object_id].copy()
            rotation = self.mj_data.xmat[object_id].copy()
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, rotation)
        return position, quaternion

    def _clamp_joints(self) -> None:
        for joint in self.robot_model.joints.values():
            if joint.limits is not None:
                self.mj_data.qpos[joint.qpos_address] = np.clip(
                    self.mj_data.qpos[joint.qpos_address], *joint.limits
                )

    @classmethod
    def _singularity_metrics(cls, jacobian):
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if singular_values.size == 0:
            return False, float("inf"), 0.0
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        if sigma_min <= 1e-12:
            condition_number = float("inf")
        else:
            condition_number = sigma_max / sigma_min
        near_singularity = (
            sigma_min < cls.SINGULARITY_MIN_SINGULAR_VALUE
            or condition_number > cls.SINGULARITY_CONDITION_NUMBER
        )
        return near_singularity, sigma_min, condition_number

    def solve_ik(
        self,
        object_name: str,
        target_position,
        target_quaternion=None,
        *,
        kind: Optional[str] = None,
        max_iterations: int = 80,
        damping: float = 0.04,
        position_weight: float = 1.0,
        orientation_weight: float = 0.2,
        tolerance: float = 0.005,
        orientation_tolerance: float = 0.03,
        step_size: float = 0.7,
        max_step: float = 0.08,
        joint_weights: Optional[Mapping[str, float]] = None,
    ) -> IKResult:
        """Damped least-squares body/site IK over controllable scalar joints."""
        try:
            kind, object_id = self.resolve_object(object_name, kind)
        except KeyError as exc:
            return IKResult(False, float("inf"), 0, str(exc))

        target_position = np.asarray(target_position, dtype=float)

        # A top-level body driven by a free joint cannot be moved by the
        # hinge/slide Jacobian below. Set its seven qpos values directly in this
        # candidate state; the caller still performs collision acceptance.
        free_joint = (
            self.robot_model.free_joint_for_body(object_id)
            if kind == "body" else None
        )
        if free_joint is not None:
            address = free_joint.qpos_address
            self.mj_data.qpos[address:address + 3] = target_position
            if target_quaternion is not None:
                quaternion = np.asarray(target_quaternion, dtype=float)
                norm = float(np.linalg.norm(quaternion))
                if norm < 1e-12:
                    return IKResult(False, float("inf"), 0, "Invalid root quaternion")
                self.mj_data.qpos[address + 3:address + 7] = quaternion / norm
            self.forward_kinematics()
            current_position, current_quaternion = self.get_body_pose(
                object_name, kind
            )
            position_error = float(np.linalg.norm(target_position - current_position))
            orientation_error = 0.0
            if target_quaternion is not None:
                orientation_error = 1.0 - abs(float(np.dot(
                    current_quaternion,
                    np.asarray(target_quaternion, dtype=float),
                )))
            error = position_error + orientation_weight * orientation_error
            return IKResult(
                error < tolerance,
                error,
                1,
                "Free-root pose updated" if error < tolerance else "Free-root update failed",
            )

        if (
            kind == "body"
            and object_id != 0
            and int(self.mj_model.body_parentid[object_id]) == 0
        ):
            return IKResult(
                False,
                float("inf"),
                0,
                "Selected root body has no movable MuJoCo free joint",
            )
        from core.ik.tasks import TCPOrientationTask, TCPPositionTask

        tasks = [TCPPositionTask(
            name="TCP position",
            weight=position_weight,
            priority=2,
            required=True,
            tolerance=tolerance,
            object_name=object_name,
            kind=kind,
            target_position=target_position,
        )]
        if target_quaternion is not None and orientation_weight > 0.0:
            tasks.append(TCPOrientationTask(
                name="TCP orientation",
                weight=orientation_weight,
                priority=2,
                required=True,
                tolerance=orientation_tolerance,
                object_name=object_name,
                kind=kind,
                target_quaternion=target_quaternion,
            ))
        return self.solve_weighted_tasks(
            tasks,
            joint_weights=joint_weights,
            max_iterations=max_iterations,
            damping=damping,
            step_size=step_size,
            max_step=max_step,
        )

    def solve_weighted_tasks(
        self,
        tasks,
        *,
        joint_weights: Optional[Mapping[str, float]] = None,
        max_iterations: int = 80,
        damping: float = 0.04,
        step_size: float = 0.7,
        max_step: float = 0.08,
    ) -> IKResult:
        """Weighted multi-task DLS over controllable scalar joints.

        Tasks carry priority metadata, but v1 intentionally solves one weighted
        stack. Strict null-space priority projection is a future extension.
        """
        joints = list(self.robot_model.joints.values())
        dofs = [joint.dof_address for joint in joints]
        qpos_addresses = [joint.qpos_address for joint in joints]
        influence = np.array([
            max(0.0, float((joint_weights or {}).get(joint.name, 1.0)))
            for joint in joints
        ])
        enabled_tasks = sorted(
            (task for task in tasks if task.enabled and task.weight > 0.0),
            key=lambda task: task.priority,
        )
        if not enabled_tasks:
            return IKResult(True, 0.0, 0, "No enabled IK tasks")
        if not np.any(influence > 1e-12):
            return IKResult(False, float("inf"), 0, "All IK joints are locked")

        final_error = float("inf")
        active_count = len(enabled_tasks)
        near_singularity = False
        min_singular_value = float("inf")
        condition_number = 0.0
        for iteration in range(max(1, int(max_iterations))):
            self.forward_kinematics()
            linearizations = [
                task.linearize(self.mj_model, self.mj_data, dofs, qpos_addresses)
                for task in enabled_tasks
            ]
            jacobian = np.vstack([item.jacobian for item in linearizations])
            weighted_jacobian = jacobian * influence[np.newaxis, :]
            (
                current_near_singularity,
                current_min_singular_value,
                current_condition_number,
            ) = self._singularity_metrics(weighted_jacobian)
            near_singularity = near_singularity or current_near_singularity
            min_singular_value = min(
                min_singular_value, current_min_singular_value
            )
            condition_number = max(condition_number, current_condition_number)
            required = [item for item in linearizations if item.required]
            convergence_set = required or linearizations
            final_error = max(
                (item.error_norm for item in convergence_set), default=0.0
            )
            if all(
                item.error_norm <= item.tolerance for item in convergence_set
            ):
                return IKResult(
                    True, final_error, iteration,
                    f"Weighted IK converged ({active_count} active tasks)",
                    near_singularity,
                    min_singular_value,
                    condition_number,
                )

            error = np.concatenate([item.error for item in linearizations])
            lhs = (
                weighted_jacobian @ jacobian.T
                + float(damping) ** 2 * np.eye(jacobian.shape[0])
            )
            try:
                task_delta = np.linalg.solve(lhs, error)
            except np.linalg.LinAlgError:
                task_delta = np.linalg.lstsq(lhs, error, rcond=None)[0]
            delta = influence * (jacobian.T @ task_delta)
            delta = np.clip(delta, -float(max_step), float(max_step))
            for joint, amount in zip(joints, delta):
                self.mj_data.qpos[joint.qpos_address] += float(step_size) * amount
            self._clamp_joints()

        self.forward_kinematics()
        linearizations = [
            task.linearize(self.mj_model, self.mj_data, dofs, qpos_addresses)
            for task in enabled_tasks
        ]
        jacobian = np.vstack([item.jacobian for item in linearizations])
        weighted_jacobian = jacobian * influence[np.newaxis, :]
        (
            current_near_singularity,
            current_min_singular_value,
            current_condition_number,
        ) = self._singularity_metrics(weighted_jacobian)
        near_singularity = near_singularity or current_near_singularity
        min_singular_value = min(min_singular_value, current_min_singular_value)
        condition_number = max(condition_number, current_condition_number)
        required = [item for item in linearizations if item.required]
        convergence_set = required or linearizations
        final_error = max(
            (item.error_norm for item in convergence_set), default=0.0
        )
        success = all(
            item.error_norm <= item.tolerance * 2.0
            for item in convergence_set
        )
        message = (
            f"Weighted IK reached tolerance ({active_count} active tasks)"
            if success else
            f"Weighted IK did not converge ({active_count} active tasks)"
        )
        return IKResult(
            success,
            final_error,
            max_iterations,
            message,
            near_singularity,
            min_singular_value,
            condition_number,
        )


class RobotStateTimeline:
    """Time-keyed qpos source of truth for interactive 3D editing."""

    def __init__(self, robot_model: RobotModel3D, initial_time=0.0, initial_qpos=None):
        self.robot_model = robot_model
        self.states: dict[float, object] = {}
        self.set_state(
            initial_time,
            robot_model.home_qpos if initial_qpos is None else initial_qpos,
        )

    @staticmethod
    def time_key(time):
        return round(float(time), 6)

    def set_state(self, time, qpos):
        key = self.time_key(time)
        if not np.isfinite(key) or key < 0.0:
            raise ValueError("timeline time must be finite and nonnegative")
        qpos = QposContract(self.robot_model.mj_model.nq).validate(qpos)
        self.states[key] = qpos

    def get_state(self, time):
        state = self.states.get(self.time_key(time))
        return None if state is None else state.copy()

    def delete_state(self, time):
        return self.states.pop(self.time_key(time), None) is not None

    def reset(self, time=0.0, qpos=None):
        self.states.clear()
        self.set_state(
            time,
            self.robot_model.home_qpos if qpos is None else qpos,
        )

    def ensure_state(self, time, fallback_qpos=None):
        key = self.time_key(time)
        existing = self.states.get(key)
        if existing is not None:
            return existing.copy()

        qpos = self.sample_state(key, fallback_qpos=fallback_qpos)
        self.states[key] = qpos.copy()
        return qpos

    def sample_state(self, time, fallback_qpos=None):
        """Interpolate a timeline pose without inserting a new editable state."""
        key = self.time_key(time)
        existing = self.states.get(key)
        if existing is not None:
            return existing.copy()

        times = sorted(self.states)
        lower = max((value for value in times if value < key), default=None)
        upper = min((value for value in times if value > key), default=None)
        if lower is not None and upper is not None:
            fraction = (key - lower) / (upper - lower)
            qpos = self._interpolate(self.states[lower], self.states[upper], fraction)
        elif lower is not None:
            qpos = self.states[lower].copy()
        elif upper is not None:
            qpos = self.states[upper].copy()
        elif fallback_qpos is not None:
            qpos = np.asarray(fallback_qpos, dtype=float).copy()
        else:
            qpos = self.robot_model.home_qpos.copy()
        return qpos

    def _interpolate(self, start, end, fraction):
        # MuJoCo's position manifold helpers correctly interpolate free-joint
        # quaternions instead of linearly blending their four components.
        velocity = np.zeros(self.robot_model.mj_model.nv, dtype=float)
        mujoco.mj_differentiatePos(
            self.robot_model.mj_model, velocity, 1.0, start, end
        )
        result = np.asarray(start, dtype=float).copy()
        mujoco.mj_integratePos(
            self.robot_model.mj_model, result, velocity, float(fraction)
        )
        return result

    def times(self):
        return sorted(self.states)

    def qpos_trajectory(self):
        return [self.states[time].copy() for time in self.times()]

    def save_npz(self, path):
        """Save exact time-keyed qpos states for a GhostGUI project."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        times = np.asarray(self.times(), dtype=float)
        if len(times):
            qpos = np.stack([self.states[time] for time in times]).astype(float)
        else:
            qpos = np.empty((0, self.robot_model.mj_model.nq), dtype=float)
        np.savez(path, schema_version=np.asarray([1]), times=times, qpos=qpos)

    def load_npz(self, path):
        """Load exact time-keyed qpos states from a GhostGUI project."""
        path = Path(path).expanduser()
        with np.load(path, allow_pickle=False) as data:
            schema_version = int(np.asarray(data["schema_version"]).flat[0])
            if schema_version != 1:
                raise ValueError(
                    f"Unsupported qpos timeline schema: {schema_version}"
                )
            times = np.asarray(data["times"], dtype=float)
            qpos = np.asarray(data["qpos"], dtype=float)

        expected_width = int(self.robot_model.mj_model.nq)
        if times.ndim != 1:
            raise ValueError("timeline times must be a 1D array")
        if qpos.ndim != 2 or qpos.shape[1] != expected_width:
            raise ValueError(
                f"timeline qpos must have shape (N, {expected_width})"
            )
        if qpos.shape[0] != times.shape[0]:
            raise ValueError("timeline times and qpos rows must have equal length")
        normalized_times, normalized_qpos = validate_trajectory_arrays(
            times,
            qpos,
            expected_width,
        )
        if len(normalized_times) and any(
            earlier == later
            for earlier, later in zip(
                normalized_times,
                normalized_times[1:],
            )
        ):
            raise ValueError("timeline times must be strictly increasing")

        self.states.clear()
        if not len(normalized_times):
            self.reset()
            return
        for time, state in zip(normalized_times, normalized_qpos):
            self.set_state(float(time), state)


class TrajectoryGhostRenderer:
    """Caches sampled qpos and FK transforms; no visual copies grow per frame."""

    def __init__(self, robot_model: RobotModel3D):
        self.robot_model = robot_model
        self._scratch = robot_model.create_state()
        self._signature = None
        self.transforms: list[tuple[object, object]] = []

    def update(self, trajectory, stride: int = 5) -> bool:
        stride = max(1, int(stride))
        qposes = [np.asarray(q, dtype=float) for q in trajectory]
        signature = (stride, tuple(q.tobytes() for q in qposes))
        if signature == self._signature:
            return False
        self._signature = signature
        self.transforms = []
        for qpos in qposes[::stride]:
            self._scratch.set_qpos(qpos)
            self.transforms.append(
                (
                    self._scratch.mj_data.geom_xpos.copy(),
                    self._scratch.mj_data.geom_xmat.copy(),
                )
            )
        return True

    def clear(self) -> None:
        self._signature = None
        self.transforms = []


def interpolate_qpos(start, target, frames: int = 60):
    if frames < 2:
        raise ValueError("A demo trajectory needs at least two frames.")
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if start.shape != target.shape:
        raise ValueError("Trajectory endpoints must have matching shapes.")
    return [start + alpha * (target - start) for alpha in np.linspace(0.0, 1.0, frames)]
