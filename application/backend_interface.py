"""
backend_interface.py

Purpose:
    Connect GUI trajectory x(t) to backend q(t).

Current behavior:
    - Uses C++ robot_backend if available
    - Falls back to Python version if C++ backend is not compiled
    - Can export q(t) CSV
"""

from dataclasses import dataclass
import csv
from enum import Enum
import math
from pathlib import Path
import warnings

from application.paths import (
    BUNDLED_DATA_ROOT,
    atomic_text_writer,
    prepare_csv_save_path,
)
from core.math3d import rpy_to_quaternion as _shared_rpy_to_quaternion
from core.ik import IKSolverSettings, solve_pose_targets

try:
    import mujoco
    import numpy as np

    MUJOCO_IK_AVAILABLE = True
except ImportError:
    mujoco = None
    np = None
    MUJOCO_IK_AVAILABLE = False


try:
    import robot_backend

    CPP_BACKEND_AVAILABLE = True

except ImportError:
    robot_backend = None
    CPP_BACKEND_AVAILABLE = False


LAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


DEFAULT_JOINT_POSITIONS = [0.0 for _ in LAB_JOINT_NAMES]


JOINT_INDEX = {
    name: index
    for index, name in enumerate(LAB_JOINT_NAMES)
}


PROJECT_ROOT = BUNDLED_DATA_ROOT
MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


IK_TASKS = {
    "torso": ("body", "robot/torso_link", 0.6),
    "left_foot": ("site", "robot/left_foot", 1.0),
    "right_foot": ("site", "robot/right_foot", 1.0),
    "left_hand": ("site", "robot/left_palm", 0.8),
    "right_hand": ("site", "robot/right_palm", 0.8),
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


@dataclass
class PythonRobotConfiguration:
    time: float = 0.0

    base_x: float = 0.0
    base_y: float = 0.0
    base_z: float = 0.9

    base_qw: float = 1.0
    base_qx: float = 0.0
    base_qy: float = 0.0
    base_qz: float = 0.0

    joint_names: list = None
    joint_positions: list = None

    ik_error: float = 0.0
    orientation_error: float = 0.0
    success: bool = True
    status: str = "Approximate analytic solve"
    qpos: object = None


def rpy_to_quaternion(roll, pitch, yaw):
    """Compatibility wrapper for the shared MuJoCo wxyz/radian contract."""
    return tuple(
        float(value)
        for value in _shared_rpy_to_quaternion(roll, pitch, yaw)
    )


class BackendKind(str, Enum):
    MUJOCO = "mujoco"
    CPP = "cpp"
    ANALYTIC = "analytic"


class FallbackPolicy(str, Enum):
    ERROR = "error"
    ALLOW_APPROXIMATE = "allow_approximate"


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    whole_body_pose_ik: bool
    grouped_targets: bool
    exact_qpos: bool
    approximate: bool


@dataclass(frozen=True)
class BackendSelection:
    requested: BackendKind
    selected: BackendKind
    capabilities: BackendCapabilities
    degraded: bool = False
    reason: str | None = None


MUJOCO_CAPABILITIES = BackendCapabilities(True, True, True, False)
CPP_CAPABILITIES = BackendCapabilities(False, False, False, True)
ANALYTIC_CAPABILITIES = BackendCapabilities(False, True, False, True)


def target_point(target, default_y=0.0):
    y = target.y

    if abs(y) < 1e-9:
        y = default_y

    return target.x, y, target.z


def two_link_flexion(distance, upper_len, lower_len):
    d = clamp(distance, 1e-6, upper_len + lower_len - 1e-6)
    cos_angle = (
        upper_len * upper_len
        + lower_len * lower_len
        - d * d
    ) / (2.0 * upper_len * lower_len)
    return math.pi - math.acos(clamp(cos_angle, -1.0, 1.0))


class PythonTrajectoryBackend:
    def __init__(self):
        self.joint_names = LAB_JOINT_NAMES
        self.default_joint_positions = DEFAULT_JOINT_POSITIONS
        self.last_solution = []

    def make_default_configuration(self, time=0.0):
        return PythonRobotConfiguration(
            time=time,
            joint_names=self.joint_names,
            joint_positions=self.default_joint_positions.copy(),
        )

    def backend_label(self):
        return "Approximate analytic trajectory backend"

    def copy_configuration_at_time(self, q_prev, time):
        return PythonRobotConfiguration(
            time=time,
            base_x=q_prev.base_x,
            base_y=q_prev.base_y,
            base_z=q_prev.base_z,
            base_qw=q_prev.base_qw,
            base_qx=q_prev.base_qx,
            base_qy=q_prev.base_qy,
            base_qz=q_prev.base_qz,
            joint_names=self.joint_names,
            joint_positions=q_prev.joint_positions.copy(),
            ik_error=q_prev.ik_error,
            orientation_error=q_prev.orientation_error,
            success=q_prev.success,
            status=q_prev.status,
        )

    def default_targets_for_configuration(self, q):
        return {
            "torso": (q.base_x, q.base_y, q.base_z + 0.35),
            "left_foot": (q.base_x - 0.10, q.base_y + 0.08, 0.0),
            "right_foot": (q.base_x + 0.10, q.base_y - 0.08, 0.0),
            "left_hand": (q.base_x - 0.35, q.base_y + 0.18, q.base_z + 0.05),
            "right_hand": (q.base_x + 0.35, q.base_y - 0.18, q.base_z + 0.05),
        }

    def estimate_joint_positions(self, q, active_targets):
        """
        First-pass task-target to 29-joint estimate.

        This is a lightweight analytic approximation for CSV generation, not a
        replacement for full humanoid IK. It keeps every named joint column
        populated from pelvis, torso, hand, and foot targets.
        """
        joints = [0.0 for _ in self.joint_names]
        defaults = self.default_targets_for_configuration(q)

        def set_joint(name, value):
            index = JOINT_INDEX.get(name)
            if index is not None:
                joints[index] = clamp(value, -2.8, 2.8)

        torso_target = active_targets.get("torso")
        if torso_target is not None:
            tx, ty, tz = target_point(torso_target, default_y=q.base_y)
            dx = tx - q.base_x
            dy = ty - q.base_y
            dz = max(0.05, tz - q.base_z)

            set_joint("waist_yaw_joint", torso_target.yaw)
            set_joint("waist_roll_joint", torso_target.roll + math.atan2(dy, dz))
            set_joint("waist_pitch_joint", torso_target.pitch + math.atan2(dx, dz))

        self.estimate_leg_joints(
            side="left",
            q=q,
            target=active_targets.get("left_foot"),
            default_point=defaults["left_foot"],
            set_joint=set_joint,
        )
        self.estimate_leg_joints(
            side="right",
            q=q,
            target=active_targets.get("right_foot"),
            default_point=defaults["right_foot"],
            set_joint=set_joint,
        )
        self.estimate_arm_joints(
            side="left",
            q=q,
            target=active_targets.get("left_hand"),
            default_point=defaults["left_hand"],
            set_joint=set_joint,
        )
        self.estimate_arm_joints(
            side="right",
            q=q,
            target=active_targets.get("right_hand"),
            default_point=defaults["right_hand"],
            set_joint=set_joint,
        )

        return joints

    def estimate_leg_joints(self, side, q, target, default_point, set_joint):
        sign = 1.0 if side == "left" else -1.0
        hip_y = q.base_y + sign * 0.08
        hip_x = q.base_x
        hip_z = q.base_z - 0.10

        if target is None:
            foot_x, foot_y, foot_z = default_point
            roll = pitch = yaw = 0.0
        else:
            foot_x, foot_y, foot_z = target_point(target, default_y=hip_y)
            roll = target.roll
            pitch = target.pitch
            yaw = target.yaw

        dx = foot_x - hip_x
        dy = foot_y - hip_y
        dz = foot_z - hip_z

        upper_len = 0.42
        lower_len = 0.42
        distance = math.sqrt(dx * dx + dz * dz)
        knee = two_link_flexion(distance, upper_len, lower_len)
        hip_pitch = math.atan2(dx, max(1e-6, -dz)) - 0.5 * knee
        ankle_pitch = -hip_pitch - knee

        set_joint(f"{side}_hip_pitch_joint", hip_pitch)
        set_joint(f"{side}_hip_roll_joint", sign * math.atan2(dy, max(0.10, abs(dz))))
        set_joint(f"{side}_hip_yaw_joint", yaw)
        set_joint(f"{side}_knee_joint", knee)
        set_joint(f"{side}_ankle_pitch_joint", ankle_pitch + pitch)
        set_joint(
            f"{side}_ankle_roll_joint",
            -sign * math.atan2(dy, max(0.10, abs(dz))) + roll,
        )

    def estimate_arm_joints(self, side, q, target, default_point, set_joint):
        sign = 1.0 if side == "left" else -1.0
        shoulder_y = q.base_y + sign * 0.18
        shoulder_x = q.base_x
        shoulder_z = q.base_z + 0.35

        if target is None:
            hand_x, hand_y, hand_z = default_point
            roll = pitch = yaw = 0.0
        else:
            hand_x, hand_y, hand_z = target_point(target, default_y=shoulder_y)
            roll = target.roll
            pitch = target.pitch
            yaw = target.yaw

        dx = hand_x - shoulder_x
        dy = hand_y - shoulder_y
        dz = hand_z - shoulder_z

        upper_len = 0.28
        lower_len = 0.28
        planar_distance = math.sqrt(dx * dx + dz * dz)
        elbow = two_link_flexion(planar_distance, upper_len, lower_len)
        shoulder_pitch = -math.atan2(dz, dx if abs(dx) > 1e-6 else 1e-6)

        if side == "right":
            shoulder_pitch = math.atan2(dz, -dx if abs(dx) > 1e-6 else 1e-6)

        set_joint(f"{side}_shoulder_pitch_joint", shoulder_pitch)
        set_joint(f"{side}_shoulder_roll_joint", sign * math.atan2(dy, max(0.10, planar_distance)))
        set_joint(f"{side}_shoulder_yaw_joint", yaw + math.atan2(dy, max(0.10, abs(dx))))
        set_joint(f"{side}_elbow_joint", elbow)
        set_joint(f"{side}_wrist_roll_joint", roll)
        set_joint(f"{side}_wrist_pitch_joint", pitch)
        set_joint(f"{side}_wrist_yaw_joint", yaw)

    def solve_trajectory(self, trajectory):
        if getattr(trajectory, "samples", None) is not None:
            return self.solve_grouped_trajectory(trajectory.samples)

        self.last_solution = []
        active_targets = {}

        sorted_frames = sorted(trajectory.frames, key=lambda f: f.time)

        for frame in sorted_frames:
            q = self.make_default_configuration(time=frame.time)
            active_targets[frame.frame_name] = frame

            if frame.frame_name in ["pelvis", "base", "root"]:
                q.base_x = frame.x
                q.base_y = frame.y
                q.base_z = frame.z

                (
                    q.base_qw,
                    q.base_qx,
                    q.base_qy,
                    q.base_qz,
                ) = rpy_to_quaternion(
                    frame.roll,
                    frame.pitch,
                    frame.yaw,
                )

                q.status = (
                    f"Approximate analytic solve: mapped {frame.frame_name} "
                    f"target to base pose at t={frame.time:.2f}s"
                )

            else:
                q.status = (
                    f"Approximate analytic solve: estimated joints from "
                    f"{frame.frame_name} target"
                )

            q.joint_positions = self.estimate_joint_positions(q, active_targets)
            self.last_solution.append(q)

        return self.last_solution

    def solve_grouped_trajectory(self, samples):
        self.last_solution = []
        q_prev = self.make_default_configuration()
        active_targets = {}

        for sample in samples:
            targets = sample["targets"]
            active_targets.update(targets)
            pelvis_target = None

            for name in ["pelvis", "base", "root"]:
                if name in targets:
                    pelvis_target = targets[name]
                    break

            q = self.copy_configuration_at_time(q_prev, sample["time"])

            ignored_names = sorted(
                name
                for name in targets.keys()
                if name not in ["pelvis", "base", "root"]
            )

            if pelvis_target is not None:
                q.base_x = pelvis_target.x
                q.base_y = pelvis_target.y
                q.base_z = pelvis_target.z

                (
                    q.base_qw,
                    q.base_qx,
                    q.base_qy,
                    q.base_qz,
                ) = rpy_to_quaternion(
                    pelvis_target.roll,
                    pelvis_target.pitch,
                    pelvis_target.yaw,
                )

                q.status = (
                    f"Approximate analytic solve: mapped {pelvis_target.frame_name} "
                    f"target to base pose at t={sample['time']:.2f}s"
                )
            else:
                q.status = (
                    "Approximate analytic solve: no pelvis/base/root target at "
                    f"t={sample['time']:.2f}s; held previous base pose"
                )

            if ignored_names:
                q.status += (
                    "; estimated joint targets from: "
                    + ", ".join(ignored_names)
                )

            q.joint_positions = self.estimate_joint_positions(q, active_targets)
            self.last_solution.append(q)
            q_prev = q

        return self.last_solution

    def export_last_solution_csv(self, csv_path):
        if not self.last_solution:
            raise RuntimeError("No solved trajectory to export.")
        csv_path = prepare_csv_save_path(csv_path)

        header = [
            "time",
            "base_x",
            "base_y",
            "base_z",
            "base_qw",
            "base_qx",
            "base_qy",
            "base_qz",
        ] + self.joint_names

        with atomic_text_writer(csv_path, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for q in self.last_solution:
                row = [
                    q.time,
                    q.base_x,
                    q.base_y,
                    q.base_z,
                    q.base_qw,
                    q.base_qx,
                    q.base_qy,
                    q.base_qz,
                ] + q.joint_positions

                writer.writerow(row)


class MujocoIKBackend(PythonTrajectoryBackend):
    """
    MuJoCo pose IK backend.

    Pelvis/base/root targets set the floating base directly. Other active
    target frames are solved against real MuJoCo bodies/sites using position
    and rotation Jacobians with damped least squares.
    """

    def __init__(self, model_path=MODEL_PATH, mj_model=None, adapter=None):
        super().__init__()
        self.adapter = adapter
        if not MUJOCO_IK_AVAILABLE:
            raise RuntimeError("mujoco or numpy is not available")
        if adapter is None:
            raise ValueError(
                "MuJoCo IK requires a model adapter so interactive and batch "
                "solves share one robotics contract"
            )
        self.joint_names = list(adapter.actuated_joints)
        self.default_joint_positions = [
            float(adapter.home_qpos[adapter.joints[name].qpos_address])
            for name in self.joint_names
        ]

        self.model_path = Path(model_path)

        if mj_model is None and not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        self.model = mj_model or adapter.mj_model
        self.state = adapter.create_state()
        self.data = self.state.mj_data

        self.joint_qpos_addresses = {}
        self.joint_dof_addresses = {}
        self.joint_limits = {}
        self.task_weights = {}

        self.build_joint_maps()
        self.build_task_bindings()
        self.reset_model_state()

    def reset_model_state(self):
        self.state.reset_to_default()
        self.data = self.state.mj_data

    def build_joint_maps(self):
        for joint_id in range(self.model.njnt):
            mujoco_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if not mujoco_name:
                continue

            plain_name = mujoco_name
            if plain_name.startswith("robot/"):
                plain_name = plain_name[len("robot/"):]

            if plain_name not in self.joint_names:
                continue

            self.joint_qpos_addresses[plain_name] = int(
                self.model.jnt_qposadr[joint_id]
            )
            self.joint_dof_addresses[plain_name] = int(
                self.model.jnt_dofadr[joint_id]
            )

            if self.model.jnt_limited[joint_id]:
                lo, hi = self.model.jnt_range[joint_id]
                self.joint_limits[plain_name] = (float(lo), float(hi))

    def build_task_bindings(self):
        if self.adapter is None or self.adapter.info.key == "g1":
            configured = IK_TASKS
        else:
            configured = {
                name: (kind, mujoco_name, 1.0)
                for name, (kind, mujoco_name)
                in self.adapter.logical_frame_bindings.items()
            }
        for frame_name, (kind, mujoco_name, weight) in configured.items():
            if kind == "site":
                object_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    mujoco_name,
                )
            else:
                object_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    mujoco_name,
                )

            if object_id >= 0:
                self.task_weights[frame_name] = float(weight)

    def backend_label(self):
        return "MuJoCo pose IK backend"

    def qpos_to_configuration(self, time=0.0, status="MuJoCo IK"):
        q = PythonRobotConfiguration(
            time=time,
            joint_names=self.joint_names,
            joint_positions=[],
            status=status,
            qpos=self.data.qpos.copy(),
        )

        free_joints = (
            list(self.adapter.free_joints_by_body.values())
            if self.adapter is not None else []
        )
        if free_joints:
            address = free_joints[0].qpos_address
            q.base_x, q.base_y, q.base_z = map(
                float, self.data.qpos[address:address + 3]
            )
            q.base_qw, q.base_qx, q.base_qy, q.base_qz = map(
                float, self.data.qpos[address + 3:address + 7]
            )

        for joint_name in self.joint_names:
            qpos_address = self.joint_qpos_addresses.get(joint_name)
            if qpos_address is None:
                q.joint_positions.append(0.0)
            else:
                q.joint_positions.append(float(self.data.qpos[qpos_address]))

        return q

    def set_base_from_target(self, target):
        free_joints = (
            list(self.adapter.free_joints_by_body.values())
            if self.adapter is not None else []
        )
        address = free_joints[0].qpos_address if free_joints else 0
        if self.adapter is not None and not free_joints:
            return False
        self.data.qpos[address] = target.x
        self.data.qpos[address + 1] = target.y
        self.data.qpos[address + 2] = target.z
        (
            self.data.qpos[address + 3],
            self.data.qpos[address + 4],
            self.data.qpos[address + 5],
            self.data.qpos[address + 6],
        ) = rpy_to_quaternion(target.roll, target.pitch, target.yaw)
        return True

    def solve_pose_ik(
        self,
        active_targets,
        max_iterations=80,
        tolerance=0.005,
        orientation_tolerance=0.03,
        orientation_weight=0.25,
        damping=0.04,
        step_size=0.7,
        max_step=0.08,
    ):
        settings = IKSolverSettings(
            max_iterations=max_iterations,
            position_tolerance=tolerance,
            orientation_tolerance=orientation_tolerance,
            orientation_weight=orientation_weight,
            damping=damping,
            step_size=step_size,
            max_step=max_step,
        )
        result = solve_pose_targets(
            self.state,
            active_targets,
            self.adapter.logical_frame_bindings,
            frame_weights=self.task_weights,
            settings=settings,
        )
        self.data = self.state.mj_data
        self.last_orientation_error = result.orientation_error
        success = (
            result.ik_result.success
            and result.position_error <= tolerance * 2.0
            and (
                orientation_weight <= 0.0
                or result.orientation_error <= orientation_tolerance * 2.0
            )
        )
        return (
            result.position_error,
            success,
            result.ik_result.iterations,
        )

    def solve_position_ik(
        self,
        active_targets,
        max_iterations=80,
        tolerance=0.005,
        damping=0.04,
        step_size=0.7,
        max_step=0.08,
    ):
        """Compatibility entry point for callers that request position only."""
        return self.solve_pose_ik(
            active_targets,
            max_iterations=max_iterations,
            tolerance=tolerance,
            orientation_tolerance=float("inf"),
            orientation_weight=0.0,
            damping=damping,
            step_size=step_size,
            max_step=max_step,
        )

    def solve_trajectory(self, trajectory):
        if getattr(trajectory, "samples", None) is not None:
            return self.solve_grouped_trajectory(trajectory.samples)

        samples = [
            {
                "time": frame.time,
                "targets": {frame.frame_name: frame},
            }
            for frame in sorted(trajectory.frames, key=lambda f: f.time)
        ]
        return self.solve_grouped_trajectory(samples)

    def solve_grouped_trajectory(self, samples):
        self.last_solution = []
        self.reset_model_state()
        active_targets = {}

        for sample in samples:
            targets = sample["targets"]
            active_targets.update(targets)
            pelvis_target = None

            for name in ["pelvis", "base", "root"]:
                if name in targets:
                    pelvis_target = targets[name]
                    break

            if pelvis_target is not None:
                self.set_base_from_target(pelvis_target)

            error, success, iterations = self.solve_pose_ik(active_targets)
            q = self.qpos_to_configuration(
                time=sample["time"],
                status=(
                    "MuJoCo pose IK: "
                    f"position_error={error:.4f}, "
                    f"orientation_error={self.last_orientation_error:.4f}, "
                    f"iterations={iterations}"
                ),
            )
            q.ik_error = error
            q.orientation_error = self.last_orientation_error
            q.success = success

            if pelvis_target is None:
                q.status += "; held previous base pose"

            self.last_solution.append(q)

        return self.last_solution


class BackendInterface:
    def __init__(
        self,
        mj_model=None,
        adapter=None,
        *,
        preferred_backend=BackendKind.MUJOCO,
        fallback_policy=FallbackPolicy.ALLOW_APPROXIMATE,
    ):
        try:
            preferred_backend = BackendKind(preferred_backend)
        except ValueError as exc:
            raise ValueError(
                f"unknown backend: {preferred_backend}"
            ) from exc
        try:
            fallback_policy = FallbackPolicy(fallback_policy)
        except ValueError as exc:
            raise ValueError(
                f"unknown fallback policy: {fallback_policy}"
            ) from exc

        self.grouped_fallback_backend = PythonTrajectoryBackend()
        self.ik_backend = None
        self.ik_error = None
        self.last_backend = None
        self.last_solve_degraded_reason = None
        self.fallback_policy = fallback_policy
        self.selection = None

        if preferred_backend is BackendKind.ANALYTIC:
            self.backend = self.grouped_fallback_backend
            self.using_cpp_backend = False
            self.using_mujoco_ik_backend = False
            self.selection = BackendSelection(
                requested=preferred_backend,
                selected=BackendKind.ANALYTIC,
                capabilities=ANALYTIC_CAPABILITIES,
                degraded=False,
                reason="approximate backend explicitly selected",
            )
            return

        if preferred_backend is BackendKind.CPP:
            if CPP_BACKEND_AVAILABLE:
                self.backend = robot_backend.RobotBackend()
                self.backend.set_joint_names(LAB_JOINT_NAMES)
                self.backend.set_default_joint_positions(DEFAULT_JOINT_POSITIONS)
                self.using_cpp_backend = True
                self.using_mujoco_ik_backend = False
                self.selection = BackendSelection(
                    requested=preferred_backend,
                    selected=BackendKind.CPP,
                    capabilities=CPP_CAPABILITIES,
                )
                return
            self.ik_error = "compiled robot_backend is unavailable"
        elif MUJOCO_IK_AVAILABLE:
            try:
                self.backend = MujocoIKBackend(mj_model=mj_model, adapter=adapter)
                self.ik_backend = self.backend
                self.using_cpp_backend = False
                self.using_mujoco_ik_backend = True
                self.selection = BackendSelection(
                    requested=preferred_backend,
                    selected=BackendKind.MUJOCO,
                    capabilities=MUJOCO_CAPABILITIES,
                )
                return
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                self.ik_error = str(exc)
        else:
            self.ik_error = "mujoco or numpy is unavailable"

        if fallback_policy is FallbackPolicy.ERROR:
            raise BackendUnavailableError(
                f"{preferred_backend.value} backend is unavailable: "
                f"{self.ik_error}"
            )
        reason = (
            f"{preferred_backend.value} backend unavailable: {self.ik_error}"
        )
        warnings.warn(
            reason + "; using approximate analytic trajectory generation",
            RuntimeWarning,
            stacklevel=2,
        )
        self.backend = self.grouped_fallback_backend
        self.using_cpp_backend = False
        self.using_mujoco_ik_backend = False
        self.selection = BackendSelection(
            requested=preferred_backend,
            selected=BackendKind.ANALYTIC,
            capabilities=ANALYTIC_CAPABILITIES,
            degraded=True,
            reason=reason,
        )

    def backend_name(self):
        labels = {
            BackendKind.MUJOCO: "MuJoCo weighted pose IK backend",
            BackendKind.CPP: "C++ pelvis-target backend (limited capabilities)",
            BackendKind.ANALYTIC: "Approximate analytic trajectory backend",
        }
        label = labels[self.selection.selected]
        if self.selection.degraded and self.selection.reason:
            return f"{label} ({self.selection.reason})"
        return label

    def last_backend_name(self):
        backend = self.last_backend or self.backend
        if hasattr(backend, "backend_label"):
            return backend.backend_label()
        if backend is self.grouped_fallback_backend:
            return self.grouped_fallback_backend.backend_label()
        return self.backend_name()

    def solve_trajectory(self, trajectory):
        if getattr(trajectory, "samples", None) is not None and self.using_cpp_backend:
            if self.fallback_policy is FallbackPolicy.ERROR:
                raise BackendUnavailableError(
                    "C++ backend cannot solve grouped whole-body targets"
                )
            self.last_solve_degraded_reason = (
                "C++ backend cannot solve grouped targets; used explicit "
                "approximate analytic fallback"
            )
            warnings.warn(
                self.last_solve_degraded_reason,
                RuntimeWarning,
                stacklevel=2,
            )
            self.last_backend = self.grouped_fallback_backend
            return self.grouped_fallback_backend.solve_trajectory(trajectory)

        self.last_solve_degraded_reason = (
            self.selection.reason if self.selection.degraded else None
        )
        self.last_backend = self.backend
        return self.backend.solve_trajectory(trajectory)

    def export_last_solution_csv(self, csv_path):
        backend = self.last_backend or self.backend
        backend.export_last_solution_csv(csv_path)

    def has_last_solution(self):
        backend = self.last_backend
        if backend is None:
            return False
        solution = getattr(backend, "last_solution", None)
        if solution is None:
            # Some compiled backends keep their solved trajectory internally.
            return True
        return bool(solution)

    def clear_last_solution(self):
        backends = {
            id(backend): backend
            for backend in (
                self.backend,
                self.grouped_fallback_backend,
                self.last_backend,
            )
            if backend is not None
        }
        for backend in backends.values():
            solution = getattr(backend, "last_solution", None)
            if hasattr(solution, "clear"):
                solution.clear()
            elif solution is not None:
                try:
                    backend.last_solution = []
                except (AttributeError, TypeError):
                    pass
        self.last_backend = None
