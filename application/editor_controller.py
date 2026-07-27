"""Application controller for editor commands and typed state events."""

from __future__ import annotations

from collections import deque

from .editor_commands import CommandResult, EditorCommand
from .editor_events import (
    ActiveIndexChanged,
    CurrentTimeChanged,
    DocumentActivated,
    DocumentChanged,
    DocumentDirtyChanged,
    EditorEventBus,
    EventDispatchFailure,
)
from .project_document import ProjectDocument


class EditorController:
    """Single mutation gateway shared by GUI actions and future front ends."""

    def __init__(
        self,
        document: ProjectDocument,
        events: EditorEventBus | None = None,
        *,
        dispatch_failure_limit: int = 20,
    ):
        self.document = document
        self.events = events or EditorEventBus()
        self._dispatch_failures = deque(maxlen=max(1, dispatch_failure_limit))

    @property
    def dispatch_failures(self) -> tuple[EventDispatchFailure, ...]:
        return tuple(self._dispatch_failures)

    def activate_document(self, document: ProjectDocument) -> None:
        self.document = document
        self._publish(
            DocumentActivated(
                document.document_id,
                document.model_key,
                document.revision,
            )
        )

    def execute(self, command: EditorCommand) -> CommandResult:
        previous_index = self.document.active_index
        was_dirty = self.document.dirty
        result = command.execute(self.document)
        if not isinstance(result, CommandResult):
            raise TypeError("editor commands must return CommandResult")
        if not result.changed:
            return result

        self.document.mark_changed()
        self._publish(
            DocumentChanged(
                document_id=self.document.document_id,
                model_key=self.document.model_key,
                revision=self.document.revision,
                operation=result.operation,
                active_index=self.document.active_index,
            )
        )
        if previous_index != self.document.active_index:
            self._publish(
                ActiveIndexChanged(
                    self.document.document_id,
                    previous_index,
                    self.document.active_index,
                )
            )
        if not was_dirty:
            self._publish(
                DocumentDirtyChanged(self.document.document_id, True)
            )
        return result

    def set_current_time(self, value: float) -> bool:
        previous = self.document.current_time
        if not self.document.set_current_time(value):
            return False
        self._publish(
            CurrentTimeChanged(
                self.document.document_id,
                previous,
                self.document.current_time,
            )
        )
        return True

    def mark_saved(self) -> bool:
        if not self.document.mark_saved():
            return False
        self._publish(
            DocumentDirtyChanged(self.document.document_id, False)
        )
        return True

    def _publish(self, event) -> None:
        self._dispatch_failures.extend(self.events.publish(event))
