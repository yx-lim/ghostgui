import unittest

import numpy as np

from core.ik import Collision, CollisionAwareIKSolver, CollisionChecker
from core.models import IKResult, RobotModel3D


class FakeCandidateState:
    def __init__(self, fail=False):
        self.qpos = np.zeros(1)
        self.fail = fail

    def set_qpos(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float).copy()

    def get_qpos(self):
        return self.qpos.copy()

    def solve_ik(self, name, position, quaternion, kind=None, **kwargs):
        if self.fail:
            return IKResult(False, 1.0, 1, "failed")
        self.qpos[0] = position[0]
        return IKResult(True, 0.0, 1, "converged")


class ThresholdCollisionChecker:
    def get_collisions(self, state):
        return [Collision("a", "b", "one", "two", -0.01, "self")] \
            if state.qpos[0] >= 0.6 else []


class CollisionCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel3D()

    def test_actual_mujoco_self_collision_is_reported(self):
        state = self.model.create_state()
        qpos = np.array([
            0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0, -1.3199862104,
            1.2988490986, 0.7785858799, 2.6991265477, -0.0600187980,
            -0.1215624368, 2.4998459890, -1.2506345490, 0.9695769303,
            1.3251723420, -0.5697072985, 0.1008202506, 1.4170212095,
            -0.3215799545, -0.0416873467, -1.0052899810, -0.9326487123,
            -1.4590027090, 1.9766132834, 1.5155447738, -0.3800632141,
            0.7732255758, -2.8474808931, 1.2620451356, 0.1946265770,
            1.5297533315, -0.8837980993, -0.3999788646, -0.4906455578,
        ])
        state.set_qpos(qpos)
        collisions = CollisionChecker(self.model).get_collisions(state)
        self.assertTrue(collisions)
        self.assertTrue(any(item.kind == "self" for item in collisions))
        self.assertTrue(
            all(
                item.geom1_id is not None
                and item.geom2_id is not None
                and item.body1_id is not None
                and item.body2_id is not None
                for item in collisions
            )
        )

    def test_actual_ground_collision_is_reported(self):
        state = self.model.create_state()
        qpos = state.get_qpos()
        free_joint = next(iter(self.model.free_joints_by_body.values()))
        qpos[free_joint.qpos_address + 2] -= 0.2
        state.set_qpos(qpos)

        collisions = CollisionChecker(self.model).get_collisions(state)

        self.assertTrue(collisions)
        self.assertTrue(any(item.kind == "environment" for item in collisions))

    def _fake_solver(self, fail=False):
        solver = CollisionAwareIKSolver.__new__(CollisionAwareIKSolver)
        solver.candidate_state = FakeCandidateState(fail=fail)
        solver.collision_checker = ThresholdCollisionChecker()
        solver.collision_drag_substeps = 4
        solver.ik_tolerance = 0.001
        solver.orientation_weight = 0.25
        return solver

    def test_collision_warns_without_clamping_preview_drag(self):
        result = self._fake_solver().solve_drag(
            np.zeros(1), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            object_name="target",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.accepted_fraction, 1.0)
        self.assertAlmostEqual(result.qpos[0], 1.0)
        self.assertIn("Collision warning", result.status)
        self.assertTrue(result.collisions)

    def test_failed_ik_preserves_last_valid_qpos(self):
        start = np.array([0.25])
        result = self._fake_solver(fail=True).solve_drag(
            start, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]),
            object_name="target",
        )
        self.assertFalse(result.success)
        np.testing.assert_allclose(result.qpos, start)
        self.assertEqual(result.accepted_fraction, 0.0)

    def test_ground_collision_warns_without_blocking_free_root_drag(self):
        state = self.model.create_state()
        start_qpos = state.get_qpos()
        start_position, start_quaternion = state.get_body_pose("robot/pelvis", "body")
        solver = CollisionAwareIKSolver(
            self.model,
            CollisionChecker(self.model),
            collision_drag_substeps=8,
            orientation_weight=0.0,
        )

        result = solver.solve_drag(
            start_qpos,
            start_position,
            start_quaternion,
            start_position + np.array([0.0, 0.0, -0.2]),
            start_quaternion,
            object_name="robot/pelvis",
            kind="body",
            tcp_orientation_weight=0.0,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.accepted_fraction, 1.0)
        self.assertFalse(np.allclose(result.qpos, start_qpos))
        np.testing.assert_allclose(
            result.position,
            start_position + np.array([0.0, 0.0, -0.2]),
        )
        self.assertIn("Collision warning", result.status)
        self.assertTrue(any(item.kind == "environment" for item in result.collisions))


if __name__ == "__main__":
    unittest.main()
