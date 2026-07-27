"""Commands that mutate a :class:`ProjectDocument`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.trajectory import TargetFrame

from .project_document import ProjectDocument


@dataclass(frozen=True)
class CommandResult:
    changed: bool
    operation: str
    active_index: int
    affected_count: int = 0


class EditorCommand(Protocol):
    operation: str

    def execute(self, document: ProjectDocument) -> CommandResult:
        ...


@dataclass(frozen=True)
class AddKeyframe:
    frame: TargetFrame
    operation: str = "add_keyframe"

    def execute(self, document: ProjectDocument) -> CommandResult:
        document.active_index = document.trajectory.add_frame(self.frame)
        return CommandResult(True, self.operation, document.active_index, 1)


@dataclass(frozen=True)
class UpsertKeyframe:
    frame: TargetFrame
    operation: str = "upsert_keyframe"

    def execute(self, document: ProjectDocument) -> CommandResult:
        document.active_index = document.trajectory.upsert_frame(self.frame)
        return CommandResult(True, self.operation, document.active_index, 1)


@dataclass(frozen=True)
class UpsertKeyframes:
    frames: tuple[TargetFrame, ...]
    selected_frame_name: str | None = None
    operation: str = "upsert_keyframes"

    def execute(self, document: ProjectDocument) -> CommandResult:
        active_index = document.active_index
        count = 0
        for frame in self.frames:
            index = document.trajectory.upsert_frame(frame)
            if (
                self.selected_frame_name is None
                or frame.frame_name == self.selected_frame_name
            ):
                active_index = index
            count += 1
        if count:
            document.active_index = active_index
        return CommandResult(
            bool(count),
            self.operation,
            document.active_index,
            count,
        )


@dataclass(frozen=True)
class UpdateKeyframe:
    index: int
    frame: TargetFrame
    operation: str = "update_keyframe"

    def execute(self, document: ProjectDocument) -> CommandResult:
        index = int(self.index)
        if index < 0 or index >= len(document.trajectory.frames):
            return CommandResult(False, self.operation, document.active_index)
        document.trajectory.update_frame(index, self.frame)
        document.active_index = document.trajectory.index_of_frame(self.frame)
        return CommandResult(True, self.operation, document.active_index, 1)


@dataclass(frozen=True)
class DeleteKeyframe:
    index: int
    operation: str = "delete_keyframe"

    def execute(self, document: ProjectDocument) -> CommandResult:
        index = int(self.index)
        if index < 0 or index >= len(document.trajectory.frames):
            return CommandResult(False, self.operation, document.active_index)
        document.trajectory.delete_frame(index)
        document.active_index = -1
        return CommandResult(True, self.operation, -1, 1)


@dataclass(frozen=True)
class DeleteTimeslice:
    time: float
    tolerance: float = 1e-6
    operation: str = "delete_timeslice"

    def execute(self, document: ProjectDocument) -> CommandResult:
        deleted = 0
        for track in document.trajectory.tracks.values():
            kept = []
            for frame in track:
                if abs(frame.time - self.time) <= self.tolerance:
                    deleted += 1
                else:
                    kept.append(frame)
            track[:] = kept
        if deleted:
            document.active_index = -1
        return CommandResult(
            bool(deleted),
            self.operation,
            document.active_index,
            deleted,
        )


@dataclass(frozen=True)
class ClearTrajectory:
    operation: str = "clear_trajectory"

    def execute(self, document: ProjectDocument) -> CommandResult:
        count = len(document.trajectory.frames)
        if not count:
            return CommandResult(False, self.operation, document.active_index)
        document.trajectory.clear()
        document.active_index = -1
        return CommandResult(True, self.operation, -1, count)


@dataclass(frozen=True)
class ReplaceTrajectoryFrames:
    frames: tuple[TargetFrame, ...]
    selected_frame_name: str | None = None
    selected_time: float | None = None
    operation: str = "replace_trajectory"

    def execute(self, document: ProjectDocument) -> CommandResult:
        previous = document.trajectory.to_project_dict()
        document.trajectory.clear()
        active_index = -1
        for frame in self.frames:
            index = document.trajectory.add_frame(frame)
            if (
                self.selected_frame_name is not None
                and frame.frame_name == self.selected_frame_name
                and (
                    self.selected_time is None
                    or abs(frame.time - self.selected_time) <= 1e-6
                )
            ):
                active_index = index
        if active_index < 0 and self.frames:
            active_index = document.trajectory.index_of_frame(self.frames[-1])
        document.active_index = active_index
        changed = previous != document.trajectory.to_project_dict()
        return CommandResult(
            changed,
            self.operation,
            active_index,
            len(self.frames),
        )


@dataclass(frozen=True)
class SetActiveIndex:
    index: int
    operation: str = "set_active_index"

    def execute(self, document: ProjectDocument) -> CommandResult:
        index = int(self.index)
        if index < -1 or index >= len(document.trajectory.frames):
            raise IndexError(f"keyframe index is out of range: {index}")
        changed = index != document.active_index
        document.active_index = index
        return CommandResult(changed, self.operation, index)
