"""Production adapters connecting the main window to visualization contracts."""

from __future__ import annotations

from dataclasses import dataclass

from application.visualization import (
    CallbackDisplay,
    CallbackPanel,
    CallbackTool,
    RobotFramePoseProvider,
    VisualizationContext,
    VisualizationManager,
)


@dataclass(frozen=True)
class SceneSnapshot:
    trajectory: object
    active_frame: object
    show_trajectory_lines: bool
    trajectory_smoothing: float
    show_keyframes: bool
    defined_timeslices: tuple[float, ...]


def build_main_window_visualization(window) -> VisualizationManager:
    """Build the compatibility runtime around the current Qt widgets."""

    context = VisualizationContext(
        document_provider=lambda: window.document,
        frame_poses=RobotFramePoseProvider(
            adapter_provider=lambda: window.robot_model_3d,
            state_provider=lambda: (
                window.viewer_3d.preview_state
                if window.viewer_3d.preview_active
                else window.viewer_3d.committed_state
            ),
        ),
        request_render=lambda: window.viewer_3d.canvas.request_render(),
        status_sink=window.show_status_message,
    )
    context.register_service("active_viewer", lambda: window.viewer_3d)

    manager = VisualizationManager(context)
    manager.register_display(
        CallbackDisplay("Robot Scene", window._render_robot_scene)
    )
    manager.register_display(
        CallbackDisplay("Timeline Markers", window._render_timeline_markers)
    )
    manager.register_panel(
        CallbackPanel("Editor Status", window._refresh_status_panel)
    )
    manager.register_tool(
        CallbackTool(
            "Move",
            activated=lambda: window.viewer_3d.canvas.set_gizmo_mode(
                "translate"
            ),
        )
    )
    manager.register_tool(
        CallbackTool(
            "Rotate",
            activated=lambda: window.viewer_3d.canvas.set_gizmo_mode("rotate"),
        )
    )
    manager.initialize()
    manager.select_tool("Move")
    return manager
