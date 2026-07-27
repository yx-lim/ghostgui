"""Plain-Python helpers for target-frame time-slice workflows."""

from __future__ import annotations

from dataclasses import dataclass

from core.trajectory import TargetFrame, quat_to_rpy


@dataclass(frozen=True)
class TimesliceSnapshotResult:
    count: int
    active_index: int
    selected_frame: TargetFrame | None


@dataclass(frozen=True)
class LoadedTrajectoryTargets:
    frames: tuple[TargetFrame, ...]
    imported_times: tuple[float, ...]


def editable_logical_frame_names(
    robot_model,
    control_frame_names,
    frame_bindings,
):
    """Merge logical frame names without depending on GUI widget types."""
    frame_names = []
    for name in getattr(robot_model, "trajectory_frames", []):
        if name not in frame_names:
            frame_names.append(name)
    for name in control_frame_names:
        if name not in frame_names:
            frame_names.append(name)
    for name in frame_bindings:
        if name not in frame_names:
            frame_names.append(name)
    return frame_names


def capture_timeslice_from_committed_pose(
    state,
    *,
    time,
    phase,
    frame_names,
    frame_bindings,
):
    if state is None:
        return ()

    frames = []

    for frame_name in frame_names:
        binding = frame_bindings.get(frame_name)
        if binding is None:
            continue
        kind, object_name = binding
        try:
            position, quaternion = state.get_body_pose(object_name, kind)
        except KeyError:
            continue
        roll, pitch, yaw = quat_to_rpy(quaternion)
        frames.append(
            TargetFrame(
                time=time,
                phase=phase,
                frame_name=frame_name,
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                roll=roll,
                pitch=pitch,
                yaw=yaw,
            )
        )
    return tuple(frames)


def define_timeslice_from_committed_pose(
    trajectory,
    state,
    *,
    time,
    phase,
    selected_frame_name,
    frame_names,
    frame_bindings,
):
    """Compatibility facade that captures and applies one logical timeslice."""
    frames = capture_timeslice_from_committed_pose(
        state,
        time=time,
        phase=phase,
        frame_names=frame_names,
        frame_bindings=frame_bindings,
    )
    selected_frame = None
    selected_index = -1
    last_index = -1
    for frame in frames:
        last_index = trajectory.upsert_frame(frame)
        if frame.frame_name == selected_frame_name:
            selected_frame = frame
            selected_index = last_index

    active_index = selected_index if selected_index >= 0 else last_index
    return TimesliceSnapshotResult(len(frames), active_index, selected_frame)


def selected_loaded_trajectory_import_samples(times, qposes, interval):
    if not times or not qposes:
        return []

    interval = max(0.0, float(interval))
    samples = []
    last_import_time = None
    for time, qpos in zip(times, qposes):
        time = float(time)
        if (
            last_import_time is None
            or interval <= 1e-9
            or time >= last_import_time + interval - 1e-9
        ):
            samples.append((time, qpos))
            last_import_time = time

    final_time = float(times[-1])
    if abs(samples[-1][0] - final_time) > 1e-9:
        samples.append((final_time, qposes[-1]))

    return samples


def build_loaded_trajectory_target_frames(
    robot_model,
    times,
    qposes,
    *,
    interval,
    phase,
    frame_names,
    frame_bindings,
):
    """Compute FK-derived target frames without touching GUI-owned state."""
    samples = selected_loaded_trajectory_import_samples(
        times,
        qposes,
        interval,
    )
    state = robot_model.create_state()
    frames = []
    for time, qpos in samples:
        state.set_qpos(qpos)
        for frame_name in frame_names:
            binding = frame_bindings.get(frame_name)
            if binding is None:
                continue
            kind, object_name = binding
            try:
                position, quaternion = state.get_body_pose(object_name, kind)
            except KeyError:
                continue
            roll, pitch, yaw = quat_to_rpy(quaternion)
            frames.append(
                TargetFrame(
                    time=float(time),
                    phase=phase,
                    frame_name=frame_name,
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                )
            )
    return LoadedTrajectoryTargets(
        frames=tuple(frames),
        imported_times=tuple(float(time) for time, _qpos in samples),
    )


def delete_timeslice_at_time(trajectory, time, tolerance=1e-6):
    count = 0
    for track in trajectory.tracks.values():
        kept = []
        for frame in track:
            if abs(frame.time - time) <= tolerance:
                count += 1
            else:
                kept.append(frame)
        track[:] = kept
    return count
