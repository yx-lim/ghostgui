"""Compatibility helpers for the model-specific :class:`EditorSession`."""

from __future__ import annotations

from .editor_session import EditorSession, EditorSessionState
from .project_document import ProjectDocument


class RobotModelSession(EditorSession):
    """Backward-compatible constructor for the former session data class."""

    def __init__(
        self,
        adapter,
        backend,
        reference,
        viewer_3d,
        trajectory=None,
        active_index=-1,
        *,
        document=None,
        model_key=None,
    ):
        model_key = str(
            model_key
            or getattr(adapter, "model_key", None)
            or getattr(adapter, "key", None)
            or getattr(adapter, "model_name", "unknown")
        )
        if document is None:
            document = ProjectDocument(
                model_key=model_key,
                trajectory=trajectory,
                active_index=active_index,
            ) if trajectory is not None else ProjectDocument(
                model_key=model_key,
                active_index=active_index,
            )
        super().__init__(
            model_key=model_key,
            adapter=adapter,
            backend=backend,
            reference=reference,
            viewer_3d=viewer_3d,
            document=document,
        )


def remember_current_session(model_sessions, model_key, trajectory, active_index):
    current = model_sessions.get(model_key)
    if current is not None:
        current.trajectory = trajectory
        current.active_index = active_index
        current.deactivate()


def activated_session_state(model_key, session):
    session.activate()
    return {
        "model_key": model_key,
        "robot_model_3d": session.adapter,
        "robot_model_error": session.adapter.load_warning,
        "backend_interface": session.backend,
        "model_reference": session.reference,
        "viewer_3d": session.viewer_3d,
        "document": session.document,
    }


__all__ = [
    "EditorSession",
    "EditorSessionState",
    "RobotModelSession",
    "activated_session_state",
    "remember_current_session",
]
