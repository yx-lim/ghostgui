"""
trajectory.py

Purpose:
    Stores target reference-frame keyframes.

Updated project idea:
    The GUI does not directly control the robot.
    The GUI creates a trajectory array of target reference frames.

Each TargetFrame says:
    At this time, this robot frame should be at this pose.

Updated:
    - Position is linearly interpolated by default.
    - Generated samples can optionally blend toward a smoothed Hermite path.
    - Orientation is converted from roll/pitch/yaw to quaternion.
    - Quaternion orientation is interpolated using SLERP.
    - Result is converted back to roll/pitch/yaw for the existing backend/export format.
"""

from dataclasses import dataclass, asdict
import json
import math

from core.math3d import (
    normalize_quaternion,
    quaternion_slerp,
    quaternion_to_rpy,
    rpy_to_quaternion,
)


# ============================================================
# Quaternion helpers
# Convention:
#     Quaternion is [w, x, y, z]
#     Roll  = rotation around x-axis
#     Pitch = rotation around y-axis
#     Yaw   = rotation around z-axis
#     Euler convention is standard robotics RPY:
#         R = Rz(yaw) * Ry(pitch) * Rx(roll)
# ============================================================

def normalize_quat(q):
    """Compatibility tuple wrapper around the shared wxyz contract."""
    return tuple(float(value) for value in normalize_quaternion(q))


def rpy_to_quat(roll, pitch, yaw):
    """
    Convert roll/pitch/yaw to quaternion [w, x, y, z].
    Angles are in radians.
    """
    return tuple(float(value) for value in rpy_to_quaternion(roll, pitch, yaw))


def quat_to_rpy(q):
    """
    Convert quaternion [w, x, y, z] back to roll/pitch/yaw.
    Angles are returned in radians.
    """
    return quaternion_to_rpy(q)


def slerp(q0, q1, alpha):
    """
    Spherical linear interpolation between two quaternions.

    q0, q1:
        Quaternions in [w, x, y, z] format.

    alpha:
        Interpolation value from 0.0 to 1.0.
    """
    return tuple(
        float(value) for value in quaternion_slerp(q0, q1, float(alpha))
    )


# ============================================================
# Data classes
# ============================================================

DEFAULT_TRACK_NAMES = [
    "pelvis",
    "torso",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
]


@dataclass
class TargetFrame:
    time: float = 0.0
    phase: str = "crouch"
    frame_name: str = "pelvis"

    # Position of target reference frame
    x: float = 0.0
    y: float = 0.0
    z: float = 0.9

    # Orientation of target reference frame, radians
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self):
        self.time = float(self.time)
        self.phase = str(self.phase)
        self.frame_name = str(self.frame_name).strip()
        if not self.frame_name:
            raise ValueError("target frame_name cannot be empty")
        values = (
            self.time,
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("target frame contains a non-finite value")
        if self.time < 0.0:
            raise ValueError("target frame time cannot be negative")
        self.x, self.y, self.z = map(float, (self.x, self.y, self.z))
        self.roll, self.pitch, self.yaw = map(
            float,
            (self.roll, self.pitch, self.yaw),
        )

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return TargetFrame(**data)


class Trajectory:
    def __init__(self):
        # A trajectory is no longer one mixed path. It is multiple named
        # timelines, one per robot frame/body part.
        self.tracks = {name: [] for name in DEFAULT_TRACK_NAMES}

    @property
    def frames(self):
        """
        Compatibility view for existing viewers and table code.

        The internal source of truth is self.tracks; this property returns all
        keyframes flattened and sorted for display/drawing only.
        """
        return self.all_frames()

    def ensure_track(self, frame_name):
        if frame_name not in self.tracks:
            self.tracks[frame_name] = []

    def all_frames(self):
        frames = []

        for track in self.tracks.values():
            frames.extend(track)

        return sorted(frames, key=lambda f: (f.time, f.frame_name))

    def _sort_track(self, frame_name):
        self.tracks[frame_name].sort(key=lambda f: f.time)

    def _locate_flat_index(self, index):
        frames = self.all_frames()

        if index < 0 or index >= len(frames):
            return None, None

        frame = frames[index]
        track = self.tracks.get(frame.frame_name, [])

        for track_index, candidate in enumerate(track):
            if candidate is frame:
                return frame.frame_name, track_index

        return None, None

    def index_of_frame(self, frame):
        for index, candidate in enumerate(self.all_frames()):
            if candidate is frame:
                return index

        return -1

    def add_frame(self, frame):
        """
        Add one keyframe to the matching frame-name track.
        """
        self.ensure_track(frame.frame_name)
        self.tracks[frame.frame_name].append(frame)
        self._sort_track(frame.frame_name)
        return self.index_of_frame(frame)

    def upsert_frame(self, frame, tolerance=1e-6):
        """Insert or replace one logical target keyframe at the same time."""
        self.ensure_track(frame.frame_name)
        track = self.tracks[frame.frame_name]
        for index, existing in enumerate(track):
            if abs(existing.time - frame.time) <= tolerance:
                track[index] = frame
                self._sort_track(frame.frame_name)
                return self.index_of_frame(frame)
        return self.add_frame(frame)

    def update_frame(self, index, frame):
        """
        Replace an existing keyframe selected from the flattened display.

        If the frame name changed, the keyframe moves to the new named track.
        """
        old_frame_name, track_index = self._locate_flat_index(index)

        if old_frame_name is None:
            return

        del self.tracks[old_frame_name][track_index]
        self.add_frame(frame)

    def delete_frame(self, index):
        """
        Delete a keyframe selected from the flattened display.
        """
        frame_name, track_index = self._locate_flat_index(index)

        if frame_name is None:
            return

        del self.tracks[frame_name][track_index]

    def clear(self):
        for track in self.tracks.values():
            track.clear()

    def as_list(self):
        """
        Flattened serialization for compatibility with older saved files.
        """
        return [frame.to_dict() for frame in self.all_frames()]

    def to_project_dict(self):
        """Serialize named target-frame tracks for GhostGUI project files."""
        return {
            "schema_version": 1,
            "tracks": {
                name: [frame.to_dict() for frame in track]
                for name, track in self.tracks.items()
            },
        }

    def load_project_dict(self, data):
        """
        Load named target-frame tracks from a project file.

        Older flattened trajectory lists are accepted so regular trajectory JSON
        files can still be promoted into project data.
        """
        self.tracks = {name: [] for name in DEFAULT_TRACK_NAMES}

        if isinstance(data, list):
            items = data
        else:
            if not isinstance(data, dict):
                raise ValueError(
                    "trajectory project data must be an object or legacy list"
                )
            schema_version = int(data.get("schema_version", 1))
            if schema_version != 1:
                raise ValueError(
                    f"Unsupported target trajectory schema: {schema_version}"
                )
            tracks = data.get("tracks", {})
            if not isinstance(tracks, dict):
                raise ValueError("trajectory tracks must be an object")
            for name in tracks:
                self.ensure_track(name)
            items = []
            for name, track_items in tracks.items():
                if not isinstance(track_items, list):
                    raise ValueError(f"trajectory track {name!r} must be a list")
                for item in track_items:
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"trajectory track {name!r} contains a non-object"
                        )
                    if str(item.get("frame_name", name)) != str(name):
                        raise ValueError(
                            f"trajectory frame_name does not match track {name!r}"
                        )
                    normalized = dict(item)
                    normalized.setdefault("frame_name", name)
                    items.append(normalized)

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("trajectory contains a non-object keyframe")
            self.add_frame(TargetFrame.from_dict(item))

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.as_list(), f, indent=4)

    def load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.load_project_dict(data)

    # ============================================================
    # Uniform-dt sampling
    # ============================================================

    def sample_uniform_dt(self, dt=0.01, smoothing=0.0):
        """
        Backward-compatible flattened sampling.

        Prefer sample_tracks_uniform_dt() for whole-body trajectories. This
        method samples each named track independently, then flattens the result
        for older code that expects a list of TargetFrame objects.
        """
        sampled_frames = []

        for sample in self.sample_tracks_uniform_dt(dt=dt, smoothing=smoothing):
            sampled_frames.extend(sample["targets"].values())

        return sorted(sampled_frames, key=lambda f: (f.time, f.frame_name))

    def sample_tracks_uniform_dt(self, dt=0.01, smoothing=0.0):
        """
        Sample each robot frame/body-part track independently.

        Pelvis keyframes interpolate only with pelvis keyframes, left_foot
        only with left_foot, and so on. The returned list is grouped by sampled
        time:

            {"time": t, "targets": {"pelvis": TargetFrame(...), ...}}
        """
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("trajectory sample dt must be a positive finite value")
        non_empty_tracks = {
            name: sorted(track, key=lambda f: f.time)
            for name, track in self.tracks.items()
            if track
        }

        if not non_empty_tracks:
            return []

        smoothing = max(0.0, min(1.0, float(smoothing)))

        t_start = min(track[0].time for track in non_empty_tracks.values())
        t_end = max(track[-1].time for track in non_empty_tracks.values())
        num_steps = int(round((t_end - t_start) / dt))
        samples = []

        for k in range(num_steps + 1):
            t = t_start + k * dt
            t = round(t, 10)

            if t > t_end:
                t = t_end

            targets = {}

            for frame_name, track in non_empty_tracks.items():
                if t < track[0].time - 1e-9 or t > track[-1].time + 1e-9:
                    continue

                targets[frame_name] = self.interpolate_track(track, t, smoothing)

            samples.append({
                "time": t,
                "targets": targets,
            })

        return samples

    def interpolate_track(self, sorted_frames, t, smoothing=0.0):
        """
        Interpolate one sorted frame-name track at time t.

        smoothing blends from the current linear position path at 0.0 to an
        auto-tangent cubic Hermite position path at 1.0. Orientation keeps the
        existing per-segment SLERP behavior.
        """
        f0, f1 = self.find_surrounding_keyframes(sorted_frames, t)
        frame = self.interpolate_frames(f0, f1, t)

        smoothing = max(0.0, min(1.0, float(smoothing)))
        if smoothing <= 1e-9 or len(sorted_frames) < 3:
            return frame

        smooth_x, smooth_y, smooth_z = self.smooth_position(sorted_frames, t)
        frame.x = frame.x + smoothing * (smooth_x - frame.x)
        frame.y = frame.y + smoothing * (smooth_y - frame.y)
        frame.z = frame.z + smoothing * (smooth_z - frame.z)
        return frame

    def targets_at_time(self, t):
        """
        Return interpolated targets grouped by frame name at one time.

        This is the single-time version of sample_tracks_uniform_dt(). Each
        robot frame/body part is interpolated only within its own named track;
        pelvis never interpolates with hands, feet, torso, etc.
        """
        targets = {}

        for frame_name, track in self.tracks.items():
            if not track:
                continue

            sorted_track = sorted(track, key=lambda f: f.time)

            if t < sorted_track[0].time - 1e-9:
                continue

            if t > sorted_track[-1].time + 1e-9:
                continue

            f0, f1 = self.find_surrounding_keyframes(sorted_track, t)
            targets[frame_name] = self.interpolate_frames(f0, f1, t)

        return targets

    def find_surrounding_keyframes(self, sorted_frames, t):
        """
        Find keyframes f0 and f1 such that:

            f0.time <= t <= f1.time
        """

        if t <= sorted_frames[0].time:
            return sorted_frames[0], sorted_frames[0]

        if t >= sorted_frames[-1].time:
            return sorted_frames[-1], sorted_frames[-1]

        for i in range(len(sorted_frames) - 1):
            f0 = sorted_frames[i]
            f1 = sorted_frames[i + 1]

            if f0.time <= t <= f1.time:
                return f0, f1

        return sorted_frames[-1], sorted_frames[-1]

    def find_segment_index(self, sorted_frames, t):
        if len(sorted_frames) < 2:
            return 0

        if t <= sorted_frames[0].time:
            return 0

        if t >= sorted_frames[-1].time:
            return len(sorted_frames) - 2

        for i in range(len(sorted_frames) - 1):
            if sorted_frames[i].time <= t <= sorted_frames[i + 1].time:
                return i

        return len(sorted_frames) - 2

    def smooth_position(self, sorted_frames, t):
        """
        Cubic Hermite position interpolation using automatic track tangents.
        """
        segment_index = self.find_segment_index(sorted_frames, t)
        f0 = sorted_frames[segment_index]
        f1 = sorted_frames[segment_index + 1]
        duration = f1.time - f0.time

        if abs(duration) < 1e-9:
            return f0.x, f0.y, f0.z

        alpha = max(0.0, min(1.0, (t - f0.time) / duration))
        m0 = self.position_tangent(sorted_frames, segment_index)
        m1 = self.position_tangent(sorted_frames, segment_index + 1)

        h00 = 2.0 * alpha ** 3 - 3.0 * alpha ** 2 + 1.0
        h10 = alpha ** 3 - 2.0 * alpha ** 2 + alpha
        h01 = -2.0 * alpha ** 3 + 3.0 * alpha ** 2
        h11 = alpha ** 3 - alpha ** 2

        return tuple(
            h00 * p0 + h10 * duration * tangent0
            + h01 * p1 + h11 * duration * tangent1
            for p0, p1, tangent0, tangent1 in zip(
                self.position_tuple(f0),
                self.position_tuple(f1),
                m0,
                m1,
            )
        )

    def position_tangent(self, sorted_frames, index):
        if len(sorted_frames) < 2:
            return 0.0, 0.0, 0.0

        if index <= 0:
            return self.position_slope(sorted_frames[0], sorted_frames[1])

        if index >= len(sorted_frames) - 1:
            return self.position_slope(sorted_frames[-2], sorted_frames[-1])

        return self.position_slope(
            sorted_frames[index - 1],
            sorted_frames[index + 1],
        )

    def position_slope(self, f0, f1):
        duration = f1.time - f0.time
        if abs(duration) < 1e-9:
            return 0.0, 0.0, 0.0

        return tuple(
            (b - a) / duration
            for a, b in zip(self.position_tuple(f0), self.position_tuple(f1))
        )

    def position_tuple(self, frame):
        return frame.x, frame.y, frame.z

    def interpolate_frames(self, f0, f1, t):
        """
        Interpolate between two TargetFrames.

        Position:
            Linear interpolation.

        Orientation:
            roll/pitch/yaw -> quaternion
            SLERP quaternion
            quaternion -> roll/pitch/yaw
        """

        if abs(f1.time - f0.time) < 1e-9:
            alpha = 0.0
        else:
            alpha = (t - f0.time) / (f1.time - f0.time)

        alpha = max(0.0, min(1.0, alpha))

        def lerp(a, b):
            return a + alpha * (b - a)

        # Position interpolation
        x = lerp(f0.x, f1.x)
        y = lerp(f0.y, f1.y)
        z = lerp(f0.z, f1.z)

        # Orientation interpolation using SLERP
        q0 = rpy_to_quat(f0.roll, f0.pitch, f0.yaw)
        q1 = rpy_to_quat(f1.roll, f1.pitch, f1.yaw)
        q_interp = slerp(q0, q1, alpha)

        roll, pitch, yaw = quat_to_rpy(q_interp)

        return TargetFrame(
            time=t,

            # For phase/frame_name, use previous keyframe.
            # Later, you can make this phase-aware.
            phase=f0.phase,
            frame_name=f0.frame_name,

            x=x,
            y=y,
            z=z,

            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )


class SampledTrajectory:
    def __init__(self, frames=None, samples=None):
        self.samples = samples
        self.frames = frames if frames is not None else self.flatten_samples(samples)

    def flatten_samples(self, samples):
        if samples is None:
            return []

        frames = []

        for sample in samples:
            frames.extend(sample["targets"].values())

        return sorted(frames, key=lambda f: (f.time, f.frame_name))

    def as_list(self):
        return [frame.to_dict() for frame in self.frames]
