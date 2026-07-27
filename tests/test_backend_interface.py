import unittest

from application.backend_interface import PythonTrajectoryBackend
from core.trajectory import SampledTrajectory, TargetFrame


class PythonTrajectoryBackendTests(unittest.TestCase):
    def solve_left_foot(self, *, roll=0.0, pitch=0.0, yaw=0.0):
        backend = PythonTrajectoryBackend()
        target = TargetFrame(
            time=0.0,
            frame_name="left_foot",
            x=-0.10,
            y=0.08,
            z=0.0,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
        result = backend.solve_trajectory(SampledTrajectory(samples=[{
            "time": target.time,
            "targets": {target.frame_name: target},
        }]))[0]
        return dict(zip(result.joint_names, result.joint_positions))

    def test_python_fallback_uses_full_foot_rpy(self):
        neutral = self.solve_left_foot()
        oriented = self.solve_left_foot(roll=0.20, pitch=-0.15, yaw=0.30)

        self.assertAlmostEqual(
            oriented["left_ankle_roll_joint"] - neutral["left_ankle_roll_joint"],
            0.20,
        )
        self.assertAlmostEqual(
            oriented["left_ankle_pitch_joint"] - neutral["left_ankle_pitch_joint"],
            -0.15,
        )
        self.assertAlmostEqual(oriented["left_hip_yaw_joint"], 0.30)


if __name__ == "__main__":
    unittest.main()
