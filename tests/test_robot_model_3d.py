import unittest

try:
    import numpy as np
    from core.models import (
        DEFAULT_MODEL_PATH,
        RobotModel3D,
        RobotStateTimeline,
        TrajectoryGhostRenderer,
        interpolate_qpos,
    )
except ImportError:
    np = None
    RobotModel3D = None


@unittest.skipIf(RobotModel3D is None or np is None, "MuJoCo/NumPy unavailable")
class RobotModel3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel3D(DEFAULT_MODEL_PATH)

    def setUp(self):
        self.state = self.model.create_state()

    def test_model_and_joints_load(self):
        self.assertGreater(len(self.model.get_joint_names()), 0)
        self.assertGreater(self.model.mj_model.ngeom, 0)

    def test_visual_geoms_resolve_assigned_material_colors(self):
        colors = {
            tuple(round(float(channel), 3) for channel in self.model.get_geom_rgba(geom_id))
            for geom_id in range(self.model.mj_model.ngeom)
            if int(self.model.mj_model.geom_group[geom_id]) == 2
        }
        self.assertIn((0.7, 0.7, 0.7, 1.0), colors)
        self.assertIn((0.2, 0.2, 0.2, 1.0), colors)
        self.assertNotIn((0.5, 0.5, 0.5, 1.0), colors)

    def test_model_color_lookup_does_not_mutate_material(self):
        before = self.model.mj_model.mat_rgba.copy()
        geom_id = next(
            index for index in range(self.model.mj_model.ngeom)
            if int(self.model.mj_model.geom_group[index]) == 2
        )
        color = self.model.get_geom_rgba(geom_id)
        color[:] = 0.0
        np.testing.assert_allclose(self.model.mj_model.mat_rgba, before)

    def test_joint_update_changes_qpos_and_runs_fk(self):
        name = self.model.get_joint_names()[0]
        address = self.model.joints[name].qpos_address
        before = self.state.get_qpos()
        lo, hi = self.state.get_joint_limits(name)
        value = min(hi, max(lo, float(before[address]) + 0.05))
        self.state.set_joint_value(name, value)
        self.assertAlmostEqual(self.state.get_qpos()[address], value)
        self.assertTrue(np.isfinite(self.state.mj_data.xpos).all())

    def test_body_pose_has_position_and_quaternion(self):
        position, quaternion = self.state.get_body_pose("robot/pelvis", "body")
        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=6)

    def test_demo_trajectory_shape(self):
        start = self.state.get_qpos()
        target = start.copy()
        target[-1] += 0.1
        trajectory = interpolate_qpos(start, target, 12)
        self.assertEqual(len(trajectory), 12)
        self.assertEqual(trajectory[0].shape, (self.model.mj_model.nq,))
        np.testing.assert_allclose(trajectory[0], start)
        np.testing.assert_allclose(trajectory[-1], target)

    def test_small_site_ik_target_converges(self):
        position, _ = self.state.get_body_pose("robot/left_palm", "site")
        result = self.state.solve_ik(
            "robot/left_palm",
            position + np.array([0.003, 0.0, 0.0]),
            kind="site",
            max_iterations=20,
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.error, 0.01)

    def test_free_root_pelvis_translation(self):
        position, quaternion = self.state.get_body_pose("robot/pelvis", "body")
        result = self.state.solve_ik(
            "robot/pelvis",
            position + np.array([0.01, 0.0, 0.0]),
            quaternion,
            kind="body",
            tolerance=0.001,
        )
        self.assertTrue(result.success, result.message)
        moved, _ = self.state.get_body_pose("robot/pelvis", "body")
        self.assertAlmostEqual(moved[0], position[0] + 0.01, places=6)

    def test_reset_restores_model_home_qpos(self):
        changed = self.state.get_qpos()
        changed[-1] += 0.1
        self.state.set_qpos(changed)
        self.state.reset_to_default()
        np.testing.assert_allclose(self.state.get_qpos(), self.model.home_qpos)

    def test_timeline_keeps_times_independent(self):
        timeline = RobotStateTimeline(self.model)
        at_zero = timeline.get_state(0.0)
        at_later = timeline.ensure_state(0.2)
        np.testing.assert_allclose(at_zero, at_later)
        at_later[-1] += 0.15
        timeline.set_state(0.2, at_later)
        self.assertNotEqual(timeline.get_state(0.0)[-1], timeline.get_state(0.2)[-1])

    def test_timeline_interpolates_free_joint_on_manifold(self):
        timeline = RobotStateTimeline(self.model)
        end = timeline.get_state(0.0)
        end[0] += 0.4
        end[3:7] = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        timeline.set_state(0.4, end)
        middle = timeline.sample_state(0.2)
        self.assertAlmostEqual(middle[0], 0.2, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(middle[3:7])), 1.0, places=6)
        self.assertEqual(timeline.times(), [0.0, 0.4])

        timeline.ensure_state(0.2)
        self.assertEqual(timeline.times(), [0.0, 0.2, 0.4])

    def test_ghost_cache_reuses_unchanged_trajectory(self):
        start = self.state.get_qpos()
        trajectory = interpolate_qpos(start, start, 10)
        renderer = TrajectoryGhostRenderer(self.model)
        self.assertTrue(renderer.update(trajectory, stride=3))
        original_ids = [id(item) for item in renderer.transforms]
        self.assertFalse(renderer.update(trajectory, stride=3))
        self.assertEqual(original_ids, [id(item) for item in renderer.transforms])
        self.assertEqual(len(renderer.transforms), 4)


if __name__ == "__main__":
    unittest.main()
