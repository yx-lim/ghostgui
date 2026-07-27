"""Contracts for the RViz-inspired visualization runtime."""

from __future__ import annotations

import unittest

from application.project_document import ProjectDocument
from application.visualization import (
    CallbackDisplay,
    CallbackPanel,
    CallbackTool,
    Display,
    FramePoseError,
    LifecycleState,
    RobotFramePoseProvider,
    ToolEvent,
    VisualizationContext,
    VisualizationManager,
    VisualizationUpdate,
)


class FakeFramePoses:
    def frame_names(self):
        return ()

    def pose(self, _name):
        raise FramePoseError("no frames")

    def snapshot(self):
        return {}


class RecordingDisplay(Display):
    def __init__(self, name="Scene"):
        super().__init__(name)
        self.operations = []

    def on_initialize(self):
        self.operations.append("initialize")

    def on_enable(self):
        self.operations.append("enable")

    def on_update(self, update):
        self.operations.append(("update", update.scene))

    def on_disable(self):
        self.operations.append("disable")

    def on_reset(self):
        self.operations.append("reset")

    def on_shutdown(self):
        self.operations.append("shutdown")


def context(status=None):
    return VisualizationContext(
        document_provider=lambda: ProjectDocument("g1"),
        frame_poses=FakeFramePoses(),
        status_sink=status or (lambda _message: None),
    )


class VisualizationLifecycleTests(unittest.TestCase):
    def test_display_has_deterministic_idempotent_lifecycle(self):
        display = RecordingDisplay()
        manager = VisualizationManager(context())
        manager.register_display(display)

        manager.initialize()
        manager.initialize()
        manager.update(VisualizationUpdate(scene="frame", revision=4))
        manager.reset()
        manager.shutdown()
        manager.shutdown()

        self.assertEqual(
            display.operations,
            [
                "initialize",
                "enable",
                ("update", "frame"),
                "reset",
                "disable",
                "shutdown",
            ],
        )
        self.assertEqual(display.state, LifecycleState.SHUTDOWN)

    def test_failed_display_does_not_prevent_other_components_updating(self):
        updates = []
        statuses = []

        def fail(_update):
            raise RuntimeError("renderer unavailable")

        manager = VisualizationManager(context(statuses.append))
        broken = manager.register_display(CallbackDisplay("Broken", fail))
        manager.register_panel(CallbackPanel("Status", updates.append))
        manager.initialize()

        failures = manager.update(VisualizationUpdate(scene="snapshot"))

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].component_name, "Broken")
        self.assertEqual(broken.state, LifecycleState.FAILED)
        self.assertEqual(len(updates), 1)
        self.assertIn("renderer unavailable", statuses[0])

    def test_only_selected_tool_is_enabled_and_receives_events(self):
        operations = []
        events = []
        manager = VisualizationManager(context())
        manager.register_tool(
            CallbackTool(
                "Move",
                activated=lambda: operations.append("move on"),
                deactivated=lambda: operations.append("move off"),
                event_handler=lambda event: events.append(event) or True,
            )
        )
        manager.register_tool(
            CallbackTool(
                "Rotate",
                activated=lambda: operations.append("rotate on"),
                deactivated=lambda: operations.append("rotate off"),
            )
        )
        manager.initialize()

        self.assertTrue(manager.select_tool("Move"))
        self.assertTrue(manager.handle_tool_event(ToolEvent("drag", (1, 2))))
        self.assertTrue(manager.select_tool("Rotate"))

        self.assertEqual(operations, ["move on", "move off", "rotate on"])
        self.assertEqual(events, [ToolEvent("drag", (1, 2))])
        self.assertEqual(manager.active_tool_name, "Rotate")

    def test_registration_rejects_duplicate_component_names(self):
        manager = VisualizationManager(context())
        manager.register_display(RecordingDisplay())
        with self.assertRaises(ValueError):
            manager.register_display(RecordingDisplay())


class RobotFramePoseProviderTests(unittest.TestCase):
    class Adapter:
        logical_frame_bindings = {
            "left_hand": ("site", "left_tcp"),
            "pelvis": ("body", "pelvis"),
        }

    class State:
        def __init__(self):
            self.calls = []

        def get_body_pose(self, name, kind=None):
            self.calls.append((kind, name))
            if name == "left_tcp":
                return [1.0, 2.0, 3.0], [2.0, 0.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]

    def test_resolves_logical_names_and_normalizes_wxyz_quaternion(self):
        state = self.State()
        provider = RobotFramePoseProvider(lambda: self.Adapter(), lambda: state)

        pose = provider.pose("left_hand")

        self.assertEqual(provider.frame_names(), ("left_hand", "pelvis"))
        self.assertEqual(pose.position, (1.0, 2.0, 3.0))
        self.assertEqual(pose.quaternion_wxyz, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(state.calls, [("site", "left_tcp")])
        self.assertEqual(set(provider.snapshot()), {"left_hand", "pelvis"})

    def test_unknown_or_unavailable_frame_is_explicit(self):
        provider = RobotFramePoseProvider(lambda: self.Adapter(), lambda: None)
        with self.assertRaises(FramePoseError):
            provider.pose("missing")
        with self.assertRaises(FramePoseError):
            provider.pose("pelvis")


if __name__ == "__main__":
    unittest.main()
