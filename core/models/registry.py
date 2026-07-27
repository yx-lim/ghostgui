"""Robot model registrations and logical-frame naming hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.resources import bundled_resource_root


# Compatibility name retained for callers that treated the checkout as the
# resource root. In an installed wheel this points at ``share/ghostgui``.
PROJECT_ROOT = bundled_resource_root()


@dataclass(frozen=True)
class RobotModelInfo:
    key: str
    display_name: str
    model_type: str
    model_path: Path
    root_body_candidates: tuple[str, ...]
    root_joint_candidates: tuple[str, ...] = ()
    logical_frames: dict[str, tuple[str, ...]] = field(default_factory=dict)
    package_map: dict[str, Path] = field(default_factory=dict)
    ignored_body_tokens: tuple[str, ...] = (
        "camera", "imu", "radar", "rotor", "logo", "sensor", "contour",
        "constraint", "support",
    )
    home_joints: dict[str, float] = field(default_factory=dict)


ROBOT_MODELS = {
    "g1": RobotModelInfo(
        key="g1",
        display_name="Unitree G1",
        model_type="humanoid",
        model_path=PROJECT_ROOT / "models" / "g1_29dof.xml",
        root_body_candidates=("robot/pelvis", "pelvis", "torso", "base"),
        root_joint_candidates=("robot/root", "root", "floating_base"),
        logical_frames={
            "pelvis": ("robot/pelvis", "pelvis", "base", "base_link"),
            "torso": ("robot/torso_link", "torso_link", "torso", "trunk"),
            "left_hand": ("robot/left_palm", "left_palm", "left_hand"),
            "right_hand": ("robot/right_palm", "right_palm", "right_hand"),
            "left_foot": ("robot/left_foot", "left_foot", "left_ankle_roll_link"),
            "right_foot": ("robot/right_foot", "right_foot", "right_ankle_roll_link"),
        },
    ),
    "go2": RobotModelInfo(
        key="go2",
        display_name="Unitree Go2",
        model_type="quadruped",
        model_path=PROJECT_ROOT / "models" / "go2_description.urdf",
        root_body_candidates=("base", "trunk", "base_link", "world"),
        root_joint_candidates=("floating_base", "root", "freejoint"),
        logical_frames={
            "base": ("base", "trunk", "base_link", "world"),
            "trunk": ("base", "trunk", "base_link", "world"),
            "FL_foot": ("FL_foot", "FL_calf"),
            "FR_foot": ("FR_foot", "FR_calf"),
            "RL_foot": ("RL_foot", "RL_calf"),
            "RR_foot": ("RR_foot", "RR_calf"),
        },
        package_map={
            # The vendored ROS package stores its DAE files in a flattened
            # model-specific asset directory instead of go2_description/dae.
            "go2_description": PROJECT_ROOT / "models" / "assets-go2",
        },
        home_joints={
            **{f"{leg}_thigh_joint": 0.8 for leg in ("FL", "FR", "RL", "RR")},
            **{f"{leg}_calf_joint": -1.5 for leg in ("FL", "FR", "RL", "RR")},
        },
    ),
    "h2": RobotModelInfo(
        key="h2",
        display_name="Unitree H2",
        model_type="humanoid",
        model_path=PROJECT_ROOT / "models" / "h2.urdf",
        root_body_candidates=("pelvis", "torso_link", "base", "base_link"),
        root_joint_candidates=("floating_base", "root", "freejoint"),
        logical_frames={
            "pelvis": ("pelvis", "base", "base_link"),
            "torso": ("torso_link", "torso"),
            "left_hand": ("left_hand_link", "left_wrist_yaw_link"),
            "right_hand": ("right_hand_link", "right_wrist_yaw_link"),
            "left_foot": (
                "left_ankle_pitch_link", "left_ankle_roll_link", "left_foot",
            ),
            "right_foot": (
                "right_ankle_pitch_link", "right_ankle_roll_link", "right_foot",
            ),
        },
    ),
    "z1": RobotModelInfo(
        key="z1",
        display_name="Unitree Z1",
        model_type="manipulator",
        model_path=PROJECT_ROOT / "models" / "z1.urdf",
        root_body_candidates=("link00", "base", "base_link", "world"),
        root_joint_candidates=("floating_base", "root", "freejoint"),
        logical_frames={
            "base": ("link00", "base", "base_link"),
            "tool": ("link06", "tool", "ee_link"),
            "wrist": ("link05", "link06"),
        },
    ),
}


def get_model_info(model: str | RobotModelInfo = "g1") -> RobotModelInfo:
    if isinstance(model, RobotModelInfo):
        return model
    try:
        return ROBOT_MODELS[model]
    except KeyError as exc:
        raise KeyError(
            f"Unknown robot model {model!r}; choose one of {', '.join(ROBOT_MODELS)}"
        ) from exc
