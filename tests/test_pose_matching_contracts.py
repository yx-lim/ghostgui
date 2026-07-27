import math
import os
import tempfile
import unittest

import mujoco
import numpy as np

from application.backend_interface import MujocoIKBackend, rpy_to_quaternion
from application.model_importer import default_model_library_root, discover_imported_models
from core.models import MuJoCoRobotAdapter, ROBOT_MODELS
from core.trajectory import TargetFrame, quat_to_rpy, rpy_to_quat


POSE_TARGET_PREFERENCES = {
    "g1": ("right_hand",),
    "go2": ("FL_foot",),
    "h2": ("right_hand", "left_hand", "right_wrist_yaw_link", "left_wrist_yaw_link"),
    "h2-5": ("right_wrist_yaw_link", "left_wrist_yaw_link"),
    "z1": ("tool", "link06"),
    "z1-3": ("link06",),
}


def quaternion_angle_error(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return 2.0 * math.acos(
        float(np.clip(abs(np.dot(actual, expected)), -1.0, 1.0))
    )


class PoseMatchingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.old_cache_dir = os.environ.get("GHOSTGUI_CACHE_DIR")
        os.environ["GHOSTGUI_CACHE_DIR"] = cls.cache_dir.name
        cls.model_infos = dict(ROBOT_MODELS)
        cls.model_infos.update(discover_imported_models(default_model_library_root()))

    @classmethod
    def tearDownClass(cls):
        if cls.old_cache_dir is None:
            os.environ.pop("GHOSTGUI_CACHE_DIR", None)
        else:
            os.environ["GHOSTGUI_CACHE_DIR"] = cls.old_cache_dir
        cls.cache_dir.cleanup()

    def load_adapter(self, model_key):
        return MuJoCoRobotAdapter(self.model_infos[model_key])

    def model_keys_matching(self, prefix):
        return [
            key for key in sorted(self.model_infos)
            if key == prefix or key.startswith(f"{prefix}-")
        ]

    def assert_model_backend_joint_contract(self, model_key):
        adapter = self.load_adapter(model_key)
        backend = MujocoIKBackend(mj_model=adapter.mj_model, adapter=adapter)

        self.assertEqual(backend.joint_names, adapter.get_joint_names())
        self.assertEqual(backend.joint_names, adapter.actuated_joints)

        for name in adapter.get_joint_names():
            with self.subTest(model=model_key, joint=name):
                joint = adapter.joints[name]
                self.assertEqual(backend.joint_qpos_addresses[name], joint.qpos_address)
                self.assertEqual(backend.joint_dof_addresses[name], joint.dof_address)

    def pose_target_for_model(self, model_key, adapter):
        preferences = POSE_TARGET_PREFERENCES.get(model_key, ())
        for logical in preferences:
            binding = adapter.logical_frame_bindings.get(logical)
            if binding is not None:
                return logical, *binding

        candidates = []
        for logical, (kind, name) in adapter.logical_frame_bindings.items():
            if logical in ("pelvis", "base", "root", "trunk"):
                continue
            if kind == "body":
                body_id = mujoco.mj_name2id(
                    adapter.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                if adapter.free_joint_for_body(body_id) is not None:
                    continue
            lower = logical.lower()
            priority = 1
            if any(token in lower for token in ("hand", "wrist", "foot", "ankle")):
                priority = 0
            candidates.append((priority, logical, kind, name))

        if not candidates:
            self.fail(f"No non-root pose target found for {model_key}")
        _, logical, kind, name = sorted(candidates)[0]
        return logical, kind, name

    def assert_batch_pose_match(
        self,
        model_key,
        *,
        translation=(0.0, 0.0, 0.01),
        rpy_delta=(0.0, 0.0, 0.0),
    ):
        adapter = self.load_adapter(model_key)
        logical, kind, name = self.pose_target_for_model(model_key, adapter)
        home_qpos = adapter.home_qpos.copy()
        home = adapter.create_state()
        position, quaternion = home.get_body_pose(name, kind)
        roll, pitch, yaw = quat_to_rpy(quaternion)
        target_rpy = (
            roll + rpy_delta[0],
            pitch + rpy_delta[1],
            yaw + rpy_delta[2],
        )
        target_position = np.asarray(position, dtype=float) + np.asarray(
            translation,
            dtype=float,
        )
        target = TargetFrame(
            time=0.0,
            frame_name=logical,
            x=float(target_position[0]),
            y=float(target_position[1]),
            z=float(target_position[2]),
            roll=target_rpy[0],
            pitch=target_rpy[1],
            yaw=target_rpy[2],
        )
        backend = MujocoIKBackend(mj_model=adapter.mj_model, adapter=adapter)

        result = backend.solve_grouped_trajectory([{
            "time": target.time,
            "targets": {target.frame_name: target},
        }])[0]

        solved = adapter.create_state()
        solved.set_qpos(result.qpos)
        solved_position, solved_quaternion = solved.get_body_pose(name, kind)
        target_quaternion = rpy_to_quat(*target_rpy)

        self.assertTrue(result.success, result.status)
        self.assertGreater(float(np.linalg.norm(result.qpos - home_qpos)), 1e-6)
        np.testing.assert_allclose(solved_position, target_position, atol=0.005)
        self.assertLess(
            quaternion_angle_error(solved_quaternion, target_quaternion),
            0.03,
        )

    def test_h2_model_is_discovered_for_contract_coverage(self):
        self.assertTrue(
            self.model_keys_matching("h2"),
            f"No h2 model found in {default_model_library_root()}",
        )

    def test_z2_model_is_discovered_when_present(self):
        if not self.model_keys_matching("z2"):
            self.skipTest(
                f"No z2 model found in {default_model_library_root()}; "
                "available imported models are "
                f"{sorted(key for key in self.model_infos if key not in ROBOT_MODELS)}"
            )

    def test_backend_joint_order_and_addresses_match_live_model(self):
        for model_key in sorted(self.model_infos):
            with self.subTest(model=model_key):
                self.assert_model_backend_joint_contract(model_key)

    def test_free_root_qpos_uses_mujoco_xyz_wxyz_radian_convention(self):
        for model_key in sorted(self.model_infos):
            with self.subTest(model=model_key):
                adapter = self.load_adapter(model_key)
                if not adapter.free_joints_by_body:
                    self.skipTest(f"{model_key} has no floating root")
                backend = MujocoIKBackend(mj_model=adapter.mj_model, adapter=adapter)
                free_joint = next(iter(adapter.free_joints_by_body.values()))
                address = free_joint.qpos_address
                body_name = mujoco.mj_id2name(
                    adapter.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    free_joint.body_id,
                )
                target = TargetFrame(
                    frame_name="root",
                    x=0.12,
                    y=-0.04,
                    z=0.88,
                    roll=0.25,
                    pitch=-0.15,
                    yaw=0.40,
                )

                self.assertTrue(backend.set_base_from_target(target))
                qpos = backend.data.qpos
                np.testing.assert_allclose(
                    qpos[address:address + 3],
                    [target.x, target.y, target.z],
                )

                expected_quaternion = np.asarray(rpy_to_quaternion(
                    target.roll,
                    target.pitch,
                    target.yaw,
                ))
                np.testing.assert_allclose(
                    qpos[address + 3:address + 7],
                    expected_quaternion,
                )

                state = adapter.create_state()
                state.set_qpos(qpos.copy())
                position, quaternion = state.get_body_pose(body_name, "body")
                np.testing.assert_allclose(
                    position,
                    [target.x, target.y, target.z],
                    atol=1e-9,
                )
                self.assertLess(
                    quaternion_angle_error(quaternion, expected_quaternion),
                    1e-9,
                )

    def test_gui_and_backend_quaternion_helpers_share_wxyz_radian_convention(self):
        rpy = (0.31, -0.22, 0.47)
        gui_quaternion = np.asarray(rpy_to_quat(*rpy), dtype=float)
        backend_quaternion = np.asarray(rpy_to_quaternion(*rpy), dtype=float)
        np.testing.assert_allclose(gui_quaternion, backend_quaternion, atol=1e-12)
        np.testing.assert_allclose(quat_to_rpy(gui_quaternion), rpy, atol=1e-12)

        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, gui_quaternion)
        roundtrip_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(roundtrip_quaternion, rotation)
        self.assertLess(
            quaternion_angle_error(roundtrip_quaternion, gui_quaternion),
            1e-12,
        )

    def test_batch_pose_matching_hits_requested_frame_for_all_models(self):
        for model_key in sorted(self.model_infos):
            with self.subTest(model=model_key):
                self.assert_batch_pose_match(model_key)

    def test_batch_pose_matching_rotates_g1_and_h2_targets(self):
        model_keys = ["g1", *self.model_keys_matching("h2")]
        model_keys.extend(self.model_keys_matching("z2"))
        for model_key in model_keys:
            with self.subTest(model=model_key):
                self.assert_batch_pose_match(
                    model_key,
                    translation=(0.006, 0.0, 0.0),
                    rpy_delta=(0.04, -0.02, 0.01),
                )


if __name__ == "__main__":
    unittest.main()
