"""Typed, Qt-free events for editor coordination."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, TypeVar


@dataclass(frozen=True)
class EditorEvent:
    """Base type for events emitted by application services."""


@dataclass(frozen=True)
class DocumentActivated(EditorEvent):
    document_id: str
    model_key: str
    revision: int


@dataclass(frozen=True)
class DocumentChanged(EditorEvent):
    document_id: str
    model_key: str
    revision: int
    operation: str
    active_index: int


@dataclass(frozen=True)
class ActiveIndexChanged(EditorEvent):
    document_id: str
    previous_index: int
    active_index: int


@dataclass(frozen=True)
class CurrentTimeChanged(EditorEvent):
    document_id: str
    previous_time: float
    current_time: float


@dataclass(frozen=True)
class DocumentDirtyChanged(EditorEvent):
    document_id: str
    dirty: bool


EventType = TypeVar("EventType", bound=EditorEvent)
EventCallback = Callable[[EventType], None]


@dataclass(frozen=True)
class EventDispatchFailure:
    event: EditorEvent
    callback: Callable
    error: Exception


class Subscription(Generic[EventType]):
    """Idempotent handle used to detach an event subscriber."""

    def __init__(
        self,
        bus: "EditorEventBus",
        event_type: type[EventType],
        callback: EventCallback[EventType],
    ):
        self._bus = bus
        self._event_type = event_type
        self._callback = callback
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def unsubscribe(self) -> None:
        if not self._active:
            return
        self._active = False
        self._bus._unsubscribe(self._event_type, self._callback)


class EditorEventBus:
    """Synchronous event bus that isolates faulty presentation subscribers."""

    def __init__(self):
        self._subscribers: dict[type[EditorEvent], list[Callable]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(
        self,
        event_type: type[EventType],
        callback: EventCallback[EventType],
    ) -> Subscription[EventType]:
        if not isinstance(event_type, type) or not issubclass(
            event_type, EditorEvent
        ):
            raise TypeError("event_type must inherit EditorEvent")
        if not callable(callback):
            raise TypeError("event callback must be callable")
        with self._lock:
            self._subscribers[event_type].append(callback)
        return Subscription(self, event_type, callback)

    def publish(self, event: EditorEvent) -> tuple[EventDispatchFailure, ...]:
        if not isinstance(event, EditorEvent):
            raise TypeError("only EditorEvent instances can be published")
        with self._lock:
            callbacks = [
                callback
                for event_type, subscribers in self._subscribers.items()
                if isinstance(event, event_type)
                for callback in tuple(subscribers)
            ]

        failures = []
        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:
                failures.append(EventDispatchFailure(event, callback, exc))
        return tuple(failures)

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()

    def _unsubscribe(self, event_type: type[EditorEvent], callback: Callable) -> None:
        with self._lock:
            subscribers = self._subscribers.get(event_type)
            if not subscribers:
                return
            try:
                subscribers.remove(callback)
            except ValueError:
                return
            if not subscribers:
                self._subscribers.pop(event_type, None)
