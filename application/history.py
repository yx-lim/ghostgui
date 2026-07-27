"""Bounded, presentation-independent undo/redo stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


Snapshot = TypeVar("Snapshot")


@dataclass(frozen=True)
class HistoryEntry(Generic[Snapshot]):
    description: str
    snapshot: Snapshot


@dataclass(frozen=True)
class HistoryTransition(Generic[Snapshot]):
    description: str
    target: Snapshot
    direction: str


class HistoryStack(Generic[Snapshot]):
    """Store bounded snapshots while leaving capture/restore to the caller."""

    def __init__(self, max_depth: int = 100):
        if int(max_depth) < 1:
            raise ValueError("history max_depth must be positive")
        self.max_depth = int(max_depth)
        self.undo_entries: list[HistoryEntry[Snapshot]] = []
        self.redo_entries: list[HistoryEntry[Snapshot]] = []
        self.baseline: Snapshot | None = None

    def clear(self) -> None:
        self.undo_entries.clear()
        self.redo_entries.clear()
        self.baseline = None

    def set_baseline(self, snapshot: Snapshot) -> None:
        self.baseline = snapshot

    def record(
        self,
        description: str,
        *,
        before: Snapshot,
        after: Snapshot,
    ) -> None:
        self.undo_entries.append(HistoryEntry(str(description), before))
        if len(self.undo_entries) > self.max_depth:
            del self.undo_entries[: len(self.undo_entries) - self.max_depth]
        self.redo_entries.clear()
        self.baseline = after

    def undo(self, current: Snapshot) -> HistoryTransition[Snapshot] | None:
        if not self.undo_entries:
            return None
        entry = self.undo_entries.pop()
        self.redo_entries.append(HistoryEntry(entry.description, current))
        self.baseline = entry.snapshot
        return HistoryTransition(entry.description, entry.snapshot, "undo")

    def redo(self, current: Snapshot) -> HistoryTransition[Snapshot] | None:
        if not self.redo_entries:
            return None
        entry = self.redo_entries.pop()
        self.undo_entries.append(HistoryEntry(entry.description, current))
        self.baseline = entry.snapshot
        return HistoryTransition(entry.description, entry.snapshot, "redo")
