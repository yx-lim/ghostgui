"""Authoritative, GUI-independent state for an editor document."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from core.trajectory import TargetFrame, Trajectory


@dataclass(frozen=True)
class ProjectDocumentSnapshot:
    model_key: str
    trajectory: dict
    active_index: int
    current_time: float
    timeline_duration: float
    revision: int
    dirty: bool


@dataclass
class ProjectDocument:
    """Own target keyframes and references to model-specific timeline state."""

    model_key: str
    trajectory: Trajectory = field(default_factory=Trajectory)
    active_index: int = -1
    current_time: float = 0.0
    timeline_duration: float = 5.0
    qpos_timeline: object | None = None
    revision: int = 0
    dirty: bool = False
    document_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        self.model_key = str(self.model_key)
        self.active_index = int(self.active_index)
        self.current_time = self._validated_time(self.current_time)
        self.timeline_duration = self._validated_duration(self.timeline_duration)

    @staticmethod
    def _validated_time(value: float) -> float:
        value = float(value)
        if value < 0.0:
            raise ValueError("current time cannot be negative")
        return value

    @staticmethod
    def _validated_duration(value: float) -> float:
        value = float(value)
        if value <= 0.0:
            raise ValueError("timeline duration must be positive")
        return value

    def attach_qpos_timeline(self, timeline: object | None) -> None:
        self.qpos_timeline = timeline

    def set_current_time(self, value: float) -> bool:
        value = min(self._validated_time(value), self.timeline_duration)
        if abs(value - self.current_time) <= 1e-9:
            return False
        self.current_time = value
        return True

    def set_timeline_duration(self, value: float) -> bool:
        value = self._validated_duration(value)
        if abs(value - self.timeline_duration) <= 1e-9:
            return False
        self.timeline_duration = value
        self.current_time = min(self.current_time, value)
        return True

    def mark_changed(self) -> None:
        self.revision += 1
        self.dirty = True

    def mark_saved(self) -> bool:
        if not self.dirty:
            return False
        self.dirty = False
        return True

    def snapshot(self) -> ProjectDocumentSnapshot:
        return ProjectDocumentSnapshot(
            model_key=self.model_key,
            trajectory=self.trajectory.to_project_dict(),
            active_index=self.active_index,
            current_time=self.current_time,
            timeline_duration=self.timeline_duration,
            revision=self.revision,
            dirty=self.dirty,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ProjectDocumentSnapshot,
        *,
        qpos_timeline: object | None = None,
    ) -> "ProjectDocument":
        trajectory = Trajectory()
        trajectory.load_project_dict(snapshot.trajectory)
        return cls(
            model_key=snapshot.model_key,
            trajectory=trajectory,
            active_index=snapshot.active_index,
            current_time=snapshot.current_time,
            timeline_duration=snapshot.timeline_duration,
            qpos_timeline=qpos_timeline,
            revision=snapshot.revision,
            dirty=snapshot.dirty,
        )

    def frames_at_time(
        self,
        time: float,
        *,
        tolerance: float = 1e-6,
    ) -> tuple[TargetFrame, ...]:
        time = float(time)
        return tuple(
            frame
            for frame in self.trajectory.frames
            if abs(frame.time - time) <= tolerance
        )
