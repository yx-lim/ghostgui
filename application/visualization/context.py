"""Services shared by displays, tools, and panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Callable

from .frames import FramePoseProvider


def _noop() -> None:
    return None


def _ignore_status(_message: str) -> None:
    return None


@dataclass
class VisualizationContext:
    """Runtime-owned service locator with a deliberately small stable surface.

    Components receive this context during initialization instead of reaching
    into the main window. Providers are callables so switching editor sessions
    does not require rebuilding every display.
    """

    document_provider: Callable[[], object]
    frame_poses: FramePoseProvider
    request_render: Callable[[], None] = _noop
    status_sink: Callable[[str], None] = _ignore_status
    _services: dict[str, object] = field(default_factory=dict, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def document(self):
        return self.document_provider()

    def register_service(self, name: str, service: object) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("visualization service name cannot be empty")
        with self._lock:
            if key in self._services:
                raise ValueError(f"visualization service already registered: {key}")
            self._services[key] = service

    def service(self, name: str):
        with self._lock:
            try:
                return self._services[name]
            except KeyError as exc:
                raise KeyError(f"unknown visualization service: {name}") from exc

    def report_status(self, message: str) -> bool:
        try:
            self.status_sink(str(message))
        except Exception:
            return False
        return True
