import unittest

from gui.widgets.status import status_event_from_text


class StatusEventTests(unittest.TestCase):
    def test_completed_action_is_one_compact_success(self):
        event = status_event_from_text("Loaded 82 robot trajectory states.")

        self.assertEqual(event.severity, "success")
        self.assertEqual(event.icon, "✓")
        self.assertEqual(event.title, "Loaded 82 robot trajectory states.")
        self.assertEqual(event.message, "")

    def test_verbose_ik_status_moves_metrics_into_structured_details(self):
        event = status_event_from_text(
            "axis X; MuJoCo IK converged in 4 iterations; "
            "state is collision-free; accepted=100%; IK error=0.0032; "
            "tasks=3; frame=left_hand; model=G1; preview not committed"
        )

        self.assertEqual(event.severity, "info")
        self.assertEqual(event.title, "Preview updated")
        self.assertEqual(event.message, "Left hand pose is collision-free.")
        self.assertIn("Frame: left hand", event.details)
        self.assertIn("IK error: 0.0032 m", event.details)
        self.assertIn("Active tasks: 3", event.details)
        self.assertNotIn(";", event.details)

    def test_collision_cause_is_visible_without_opening_details(self):
        event = status_event_from_text(
            "Collision blocked: Box_0 ↔ panda_link5; accepted=35%; "
            "IK error=0.0210; tasks=2; frame=trunk; model=Panda; "
            "preview not committed"
        )

        self.assertEqual(event.severity, "warning")
        self.assertEqual(event.title, "Preview blocked")
        self.assertEqual(
            event.message,
            "Collision blocked: Box_0 ↔ panda_link5.",
        )

    def test_nonblocking_collision_is_presented_as_preview_warning(self):
        event = status_event_from_text(
            "MuJoCo IK converged; Collision warning: left_hand ↔ torso; "
            "accepted=100%; IK error=0.0020; tasks=2; frame=left_hand; "
            "model=G1; preview not committed"
        )

        self.assertEqual(event.severity, "warning")
        self.assertEqual(event.title, "Preview warning")
        self.assertEqual(
            event.message,
            "Collision warning: left_hand ↔ torso.",
        )

    def test_failure_uses_error_severity(self):
        event = status_event_from_text("Could not load Go2: invalid model.")

        self.assertEqual(event.severity, "error")
        self.assertEqual(event.icon, "✕")


if __name__ == "__main__":
    unittest.main()
