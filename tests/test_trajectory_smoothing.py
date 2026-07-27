import math
import unittest

from core.trajectory import TargetFrame, Trajectory, quat_to_rpy, rpy_to_quat, slerp


class TrajectorySmoothingTests(unittest.TestCase):
    def make_corner_trajectory(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.0,
            y=0.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="pelvis",
            x=1.0,
            y=1.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=2.0,
            frame_name="pelvis",
            x=2.0,
            y=0.0,
            z=0.0,
        ))
        return trajectory

    def sample_pelvis_at(self, trajectory, time_value, smoothing):
        samples = trajectory.sample_tracks_uniform_dt(
            dt=0.5,
            smoothing=smoothing,
        )
        for sample in samples:
            if abs(sample["time"] - time_value) <= 1e-9:
                return sample["targets"]["pelvis"]
        raise AssertionError(f"missing sample at t={time_value}")

    def sample_at(self, trajectory, time_value, frame_name, smoothing, dt=0.5):
        samples = trajectory.sample_tracks_uniform_dt(
            dt=dt,
            smoothing=smoothing,
        )
        for sample in samples:
            if abs(sample["time"] - time_value) <= 1e-9:
                return sample["targets"][frame_name]
        raise AssertionError(f"missing {frame_name} sample at t={time_value}")

    def test_zero_smoothing_preserves_linear_position_interpolation(self):
        trajectory = self.make_corner_trajectory()

        midpoint = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.0)

        self.assertAlmostEqual(midpoint.x, 0.5)
        self.assertAlmostEqual(midpoint.y, 0.5)
        self.assertAlmostEqual(midpoint.z, 0.0)

    def test_full_smoothing_rounds_corner_without_moving_keyframes(self):
        trajectory = self.make_corner_trajectory()

        midpoint = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)
        keyframe = self.sample_pelvis_at(trajectory, 1.0, smoothing=1.0)

        self.assertAlmostEqual(midpoint.x, 0.5)
        self.assertGreater(midpoint.y, 0.5)
        self.assertAlmostEqual(keyframe.x, 1.0)
        self.assertAlmostEqual(keyframe.y, 1.0)

    def test_partial_smoothing_blends_linear_and_smooth_positions(self):
        trajectory = self.make_corner_trajectory()

        linear = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.0)
        smooth = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)
        halfway = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.5)

        self.assertAlmostEqual(halfway.x, (linear.x + smooth.x) * 0.5)
        self.assertAlmostEqual(halfway.y, (linear.y + smooth.y) * 0.5)
        self.assertAlmostEqual(halfway.z, (linear.z + smooth.z) * 0.5)

    def test_smoothing_clamps_to_zero_and_one(self):
        trajectory = self.make_corner_trajectory()

        below_zero = self.sample_pelvis_at(trajectory, 0.5, smoothing=-1.0)
        zero = self.sample_pelvis_at(trajectory, 0.5, smoothing=0.0)
        above_one = self.sample_pelvis_at(trajectory, 0.5, smoothing=2.0)
        one = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)

        self.assertAlmostEqual(below_zero.x, zero.x)
        self.assertAlmostEqual(below_zero.y, zero.y)
        self.assertAlmostEqual(above_one.x, one.x)
        self.assertAlmostEqual(above_one.y, one.y)

    def test_single_keyframe_track_does_not_crash_with_smoothing(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.25,
            y=-0.5,
            z=0.9,
        ))

        sample = self.sample_pelvis_at(trajectory, 0.0, smoothing=1.0)

        self.assertAlmostEqual(sample.x, 0.25)
        self.assertAlmostEqual(sample.y, -0.5)
        self.assertAlmostEqual(sample.z, 0.9)

    def test_two_keyframes_remain_linear_with_smoothing_enabled(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.0,
            y=0.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="pelvis",
            x=1.0,
            y=2.0,
            z=0.5,
        ))

        sample = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)

        self.assertAlmostEqual(sample.x, 0.5)
        self.assertAlmostEqual(sample.y, 1.0)
        self.assertAlmostEqual(sample.z, 0.25)

    def test_uneven_keyframe_times_use_time_based_tangents(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.0,
            y=0.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=0.2,
            frame_name="pelvis",
            x=1.0,
            y=1.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="pelvis",
            x=2.0,
            y=0.0,
            z=0.0,
        ))

        sample = self.sample_at(
            trajectory,
            time_value=0.1,
            frame_name="pelvis",
            smoothing=1.0,
            dt=0.1,
        )

        self.assertAlmostEqual(sample.x, 0.575)
        self.assertAlmostEqual(sample.y, 0.625)
        self.assertAlmostEqual(sample.z, 0.0)

    def test_near_duplicate_times_do_not_produce_non_finite_samples(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.0,
            y=0.0,
            z=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1e-12,
            frame_name="pelvis",
            x=10.0,
            y=10.0,
            z=10.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="pelvis",
            x=1.0,
            y=0.0,
            z=0.0,
        ))

        samples = trajectory.sample_tracks_uniform_dt(dt=0.5, smoothing=1.0)

        for sample in samples:
            pelvis = sample["targets"]["pelvis"]
            with self.subTest(time=sample["time"]):
                self.assertTrue(math.isfinite(pelvis.x))
                self.assertTrue(math.isfinite(pelvis.y))
                self.assertTrue(math.isfinite(pelvis.z))

    def test_tracks_are_smoothed_independently(self):
        trajectory = self.make_corner_trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="left_hand",
            x=0.0,
            y=0.0,
            z=1.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="left_hand",
            x=0.0,
            y=0.0,
            z=2.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=2.0,
            frame_name="left_hand",
            x=0.0,
            y=0.0,
            z=3.0,
        ))

        pelvis = self.sample_at(trajectory, 0.5, "pelvis", smoothing=1.0)
        hand = self.sample_at(trajectory, 0.5, "left_hand", smoothing=1.0)

        self.assertGreater(pelvis.y, 0.5)
        self.assertAlmostEqual(hand.x, 0.0)
        self.assertAlmostEqual(hand.y, 0.0)
        self.assertAlmostEqual(hand.z, 1.5)

    def test_orientation_still_uses_segment_slerp_not_position_smoothing(self):
        trajectory = Trajectory()
        trajectory.add_frame(TargetFrame(
            time=0.0,
            frame_name="pelvis",
            x=0.0,
            yaw=0.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=1.0,
            frame_name="pelvis",
            x=1.0,
            yaw=1.0,
        ))
        trajectory.add_frame(TargetFrame(
            time=2.0,
            frame_name="pelvis",
            x=2.0,
            yaw=-0.5,
        ))

        sample = self.sample_pelvis_at(trajectory, 0.5, smoothing=1.0)
        expected = quat_to_rpy(slerp(
            rpy_to_quat(0.0, 0.0, 0.0),
            rpy_to_quat(0.0, 0.0, 1.0),
            0.5,
        ))

        self.assertAlmostEqual(sample.roll, expected[0])
        self.assertAlmostEqual(sample.pitch, expected[1])
        self.assertAlmostEqual(sample.yaw, expected[2])


if __name__ == "__main__":
    unittest.main()
