import unittest

from gui.viewers.trajectory_colors import gl_color_for_frame, qt_color_for_frame


class TrajectoryColorTests(unittest.TestCase):
    def assert_colors_close(self, actual, expected):
        for actual_channel, expected_channel in zip(actual, expected):
            self.assertAlmostEqual(actual_channel, expected_channel, places=2)

    def test_known_semantic_frame_colors_are_preserved(self):
        self.assertEqual(gl_color_for_frame("left_foot"), (0.18, 0.42, 1.00))
        self.assertEqual(gl_color_for_frame("right_hand"), (0.95, 0.15, 0.12))

    def test_generated_frame_colors_are_stable(self):
        first = gl_color_for_frame("FL_foot")
        second = gl_color_for_frame("FL_foot")
        self.assertEqual(first, second)
        self.assertNotEqual(first, gl_color_for_frame("FR_foot"))

    def test_generated_qt_and_gl_colors_match(self):
        gl_color = gl_color_for_frame("tool")
        qt_color = qt_color_for_frame("tool").getRgbF()[:3]
        self.assert_colors_close(qt_color, gl_color)

    def test_go2_and_z1_frames_do_not_share_one_fallback_color(self):
        frame_names = [
            "FL_foot", "FR_foot", "RL_foot", "RR_foot",
            "base", "wrist", "tool",
        ]
        colors = {name: gl_color_for_frame(name) for name in frame_names}

        self.assertEqual(len(set(colors.values())), len(frame_names))


if __name__ == "__main__":
    unittest.main()
