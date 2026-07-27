"""GUI snapshot schema used by the application history stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuiHistorySnapshot:
    trajectory_frames: tuple
    trajectory_track_names: tuple
    active_index: int
    control_frame: dict
    selected_row: int
    current_time: float
    timeline_states: tuple
    committed_qpos: object
    preview_qpos: object
    preview_active: bool
    robot_trajectory: tuple
    robot_trajectory_times: tuple
    ghost_trajectory: tuple
    ghost_source: str | None
    show_ghosts: bool
    timeline_duration: float
