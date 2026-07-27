"""
Reference-frame bindings for the real MuJoCo robot model.

The GUI frame names are user-facing targets. This module maps them to concrete
MuJoCo bodies/sites so new targets can start from the real robot geometry
and remain consistent with the 3D editor.
"""

from pathlib import Path

from .registry import PROJECT_ROOT

try:
    import mujoco
    import numpy as np
except ImportError:
    mujoco = None
    np = None


MODEL_PATH = PROJECT_ROOT / "models" / "g1_29dof.xml"


REFERENCE_FRAME_BINDINGS = {
    "pelvis": ("body", "robot/pelvis"),
    "torso": ("body", "robot/torso_link"),
    "left_foot": ("site", "robot/left_foot"),
    "right_foot": ("site", "robot/right_foot"),
    "left_hand": ("site", "robot/left_palm"),
    "right_hand": ("site", "robot/right_palm"),
}


class MujocoReferenceFrames:
    def __init__(self, model_path=MODEL_PATH, mj_model=None, adapter=None):
        self.adapter = adapter
        self.model_path = Path(model_path)
        self.model = adapter.mj_model if adapter is not None else mj_model
        self.data = None
        self.error = None
        self.load()

    def load(self):
        if mujoco is None:
            self.error = "mujoco Python package is not installed."
            return

        if self.model is None and not self.model_path.exists():
            self.error = f"Robot model not found: {self.model_path}"
            return

        try:
            if self.model is None:
                self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)

            if self.adapter is not None:
                self.data.qpos[:] = self.adapter.home_qpos
            elif self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            mujoco.mj_forward(self.model, self.data)
            self.error = None
        except Exception as exc:
            self.model = None
            self.data = None
            self.error = str(exc)

    def position_for_frame(self, frame_name):
        pose = self.pose_for_frame(frame_name)
        return pose[0] if pose is not None else None

    def pose_for_frame(self, frame_name):
        if self.model is None or self.data is None:
            return None

        binding = (
            self.adapter.resolve_logical_frame(frame_name)
            if self.adapter is not None
            else REFERENCE_FRAME_BINDINGS.get(frame_name)
        )
        if binding is None:
            return None

        kind, mujoco_name = binding

        if kind == "site":
            object_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                mujoco_name,
            )
            if object_id < 0:
                return None
            pos = self.data.site_xpos[object_id]
            rotation = self.data.site_xmat[object_id]
        else:
            object_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                mujoco_name,
            )
            if object_id < 0:
                return None
            pos = self.data.xpos[object_id]
            rotation = self.data.xmat[object_id]

        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, rotation)
        return (
            (float(pos[0]), float(pos[1]), float(pos[2])),
            tuple(map(float, quaternion)),
        )
