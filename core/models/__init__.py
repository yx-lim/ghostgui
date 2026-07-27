"""MuJoCo-backed robot model logic."""

from .adapter import MuJoCoRobotAdapter
from .assets import (
    CONVERTIBLE_MESH_FORMATS,
    DIRECT_MUJOCO_MESH_FORMATS,
    COLLADA_NS,
    ConvertedMeshPart,
    ResolvedMeshAsset,
    prepare_urdf_visual_meshes,
    resolve_mesh_path,
    validate_model_assets,
)
from .model import (
    DEFAULT_MODEL_PATH,
    FreeJointInfo,
    IKResult,
    JointInfo,
    RobotModel3D,
    RobotState3D,
    RobotStateTimeline,
    TrajectoryGhostRenderer,
    interpolate_qpos,
)
from .registry import PROJECT_ROOT, ROBOT_MODELS, RobotModelInfo, get_model_info
from .reference import MujocoReferenceFrames

__all__ = [
    "COLLADA_NS",
    "CONVERTIBLE_MESH_FORMATS",
    "DEFAULT_MODEL_PATH",
    "DIRECT_MUJOCO_MESH_FORMATS",
    "ConvertedMeshPart",
    "FreeJointInfo",
    "IKResult",
    "JointInfo",
    "MuJoCoRobotAdapter",
    "MujocoReferenceFrames",
    "PROJECT_ROOT",
    "ROBOT_MODELS",
    "ResolvedMeshAsset",
    "RobotModel3D",
    "RobotModelInfo",
    "RobotState3D",
    "RobotStateTimeline",
    "TrajectoryGhostRenderer",
    "get_model_info",
    "interpolate_qpos",
    "prepare_urdf_visual_meshes",
    "resolve_mesh_path",
    "validate_model_assets",
]
