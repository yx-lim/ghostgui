"""Model-specific editor runtime and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .project_document import ProjectDocument


class EditorSessionState(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    CLOSED = "closed"


@dataclass
class EditorSession:
    model_key: str
    adapter: object
    backend: object
    reference: object
    viewer_3d: object
    document: ProjectDocument
    state: EditorSessionState = EditorSessionState.INACTIVE

    def __post_init__(self):
        self.model_key = str(self.model_key)
        if self.document.model_key != self.model_key:
            raise ValueError("session and document model keys must match")
        timeline = getattr(self.viewer_3d, "state_timeline", None)
        self.document.attach_qpos_timeline(timeline)

    @property
    def trajectory(self):
        """Compatibility view for callers migrating to ``document``."""
        return self.document.trajectory

    @trajectory.setter
    def trajectory(self, value):
        self.document.trajectory = value

    @property
    def active_index(self):
        return self.document.active_index

    @active_index.setter
    def active_index(self, value):
        self.document.active_index = int(value)

    def activate(self) -> None:
        if self.state is EditorSessionState.CLOSED:
            raise RuntimeError("cannot activate a closed editor session")
        self.state = EditorSessionState.ACTIVE

    def deactivate(self) -> None:
        if self.state is not EditorSessionState.CLOSED:
            self.state = EditorSessionState.INACTIVE

    def close(self) -> None:
        if self.state is EditorSessionState.CLOSED:
            return
        shutdown = getattr(self.viewer_3d, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.document.attach_qpos_timeline(None)
        self.state = EditorSessionState.CLOSED
