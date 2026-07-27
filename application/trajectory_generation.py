"""Plain-Python helper for generating sampled robot trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from application.paths import (
    mujoco_playback_cache_path,
    prepare_csv_save_path,
)
from core.trajectory import SampledTrajectory


@dataclass(frozen=True)
class TrajectoryGenerationResult:
    csv_path: str
    result_states: list
    status_text: str


def generate_trajectory_status(
    trajectory,
    backend_interface,
    *,
    smoothing,
    export_dt=0.01,
    csv_path=None,
):
    if csv_path is None:
        csv_path = mujoco_playback_cache_path()
    csv_path = prepare_csv_save_path(csv_path)
    sampled_tracks = trajectory.sample_tracks_uniform_dt(
        dt=export_dt,
        smoothing=smoothing,
    )
    sampled_trajectory = SampledTrajectory(samples=sampled_tracks)

    result_states = backend_interface.solve_trajectory(sampled_trajectory)
    backend_interface.export_last_solution_csv(csv_path)

    lines = []
    lines.append("Generated uniformly sampled per-frame target tracks.")
    lines.append(f"Backend: {backend_interface.last_backend_name()}")
    lines.append(f"Export dt: {export_dt:.4f} s")
    lines.append(f"Corner smoothing: {smoothing * 100.0:.0f}%")
    lines.append(f"Number of GUI keyframes: {len(trajectory.frames)}")
    lines.append(f"Number of sampled time steps: {len(sampled_tracks)}")
    lines.append(f"Number of backend states: {len(result_states)}")
    if result_states:
        max_ik_error = max(state.ik_error for state in result_states)
        lines.append(f"Max IK position error: {max_ik_error:.4f} m")
        max_orientation_error = max(
            state.orientation_error for state in result_states
        )
        lines.append(
            "Max IK orientation error: "
            f"{max_orientation_error:.4f} rad"
        )
    lines.append(f"Exported CSV to: {csv_path}")
    lines.append("")
    lines.append("First few sampled time groups:")
    lines.append("")

    for sample in sampled_tracks[:10]:
        frame_names = ", ".join(sorted(sample["targets"].keys()))
        lines.append(f"t={sample['time']:.3f}s | targets={frame_names}")

    return TrajectoryGenerationResult(
        csv_path=str(csv_path),
        result_states=result_states,
        status_text="\n".join(lines),
    )
