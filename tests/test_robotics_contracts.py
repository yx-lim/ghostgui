"""Shared coordinate, qpos, IK, collision, and backend-selection contracts."""

from __future__ import annotations

import math
from unittest.mock import patch
import unittest
import warnings

import numpy as np

from application import backend_interface
from application.backend_interface import (
    BackendInterface,
    BackendKind,
    BackendUnavailableError,
    FallbackPolicy,
    MujocoIKBackend,
)
from core.ik import CollisionChecker, IKSolverSettings
from core.math3d import (
    QUATERNION_ORDER,
    normalize_quaternion,
    quaternion_slerp,
    quaternion_to_rpy,
    rpy_to_quaternion,
)
from core.models import MuJoCoRobotAdapter
from core.robotics import QposContract, validate_trajectory_arrays
from core.trajectory import TargetFrame, Trajectory, rpy_to_quat


class CoordinateContractTests(unittest.TestCase):
    def test_quaternion_contract_is_wxyz_radians_everywhere(self):
        rpy = (0.2, -0.3, 0.4)
        quaternion = rpy_to_quaternion(*rpy)
        self.assertEqual(QUATERNION_ORDER, ("w", "x", "y", "z"))
        np.testing.assert_allclose(quaternion, rpy_to_quat(*rpy))
        np.testing.assert_allclose(quaternion_to_rpy(quaternion), rpy)

    def test_invalid_quaternion_is_rejected_instead_of_becoming_identity(self):
        with self.assertRaisesRegex(ValueError, "norm"):
            normalize_quaternion([0.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "four"):
            quaternion_slerp([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5)


class TrajectoryContractTests(unittest.TestCase):
    def test_target_frame_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            TargetFrame(time=-0.1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            TargetFrame(x=math.nan)
        with self.assertRaisesRegex(ValueError, "frame_name"):
            TargetFrame(frame_name=" ")

    def test_sampling_requires_positive_finite_dt(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame())
        for value in (0.0, -0.1, math.inf, math.nan):
            with self.subTest(dt=value):
                with self.assertRaises(ValueError):
                    trajectory.sample_tracks_uniform_dt(dt=value)

    def test_qpos_and_time_series_share_one_validator(self):
        contract = QposContract(2)
        np.testing.assert_allclose(contract.validate([1.0, 2.0]), [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            contract.validate([1.0, math.inf])
        with self.assertRaisesRegex(ValueError, "negative"):
            validate_trajectory_arrays([-0.1], [[0.0, 0.0]], 2)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            validate_trajectory_arrays(
                [0.2, 0.1],
                [[0.0, 0.0], [0.0, 0.0]],
                2,
            )


class BackendSelectionTests(unittest.TestCase):
    def test_normal_runtime_selects_exact_mujoco_backend(self):
        adapter = MuJoCoRobotAdapter("g1")
        interface = BackendInterface(
            mj_model=adapter.mj_model,
            adapter=adapter,
            fallback_policy=FallbackPolicy.ERROR,
        )
        self.assertEqual(interface.selection.selected, BackendKind.MUJOCO)
        self.assertFalse(interface.selection.degraded)
        self.assertTrue(interface.selection.capabilities.exact_qpos)

    def test_unavailable_backend_requires_explicit_approximation_policy(self):
        with patch.object(backend_interface, "MUJOCO_IK_AVAILABLE", False):
            with self.assertRaises(BackendUnavailableError):
                BackendInterface(fallback_policy=FallbackPolicy.ERROR)

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                interface = BackendInterface(
                    fallback_policy=FallbackPolicy.ALLOW_APPROXIMATE
                )

        self.assertEqual(interface.selection.selected, BackendKind.ANALYTIC)
        self.assertTrue(interface.selection.degraded)
        self.assertTrue(interface.selection.capabilities.approximate)
        self.assertTrue(captured)

    def test_unexpected_backend_programming_errors_are_not_hidden(self):
        with (
            patch.object(backend_interface, "MUJOCO_IK_AVAILABLE", True),
            patch.object(
                backend_interface,
                "MujocoIKBackend",
                side_effect=KeyError("programming defect"),
            ),
        ):
            with self.assertRaises(KeyError):
                BackendInterface(
                    fallback_policy=FallbackPolicy.ALLOW_APPROXIMATE
                )

    def test_solver_settings_reject_unsafe_values(self):
        with self.assertRaises(ValueError):
            IKSolverSettings(damping=0.0)
        with self.assertRaises(ValueError):
            IKSolverSettings(max_iterations=0)


class SupportContactPolicyTests(unittest.TestCase):
    def test_go2_shallow_foot_support_is_not_a_blocking_collision(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        checker = CollisionChecker(adapter)

        collisions = checker.get_collisions(state)

        support_ids = checker.policy.support_body_ids
        self.assertTrue(support_ids)
        self.assertFalse(
            any(
                collision.kind == "environment"
                and (
                    collision.body1_id in support_ids
                    or collision.body2_id in support_ids
                )
                for collision in collisions
            )
        )

    def test_deep_go2_ground_penetration_is_still_reported(self):
        adapter = MuJoCoRobotAdapter("go2")
        state = adapter.create_state()
        qpos = state.get_qpos()
        free_joint = next(iter(adapter.free_joints_by_body.values()))
        qpos[free_joint.qpos_address + 2] -= 0.08
        state.set_qpos(qpos)

        collisions = CollisionChecker(adapter).get_collisions(state)

        self.assertTrue(
            any(collision.kind == "environment" for collision in collisions)
        )


if __name__ == "__main__":
    unittest.main()
