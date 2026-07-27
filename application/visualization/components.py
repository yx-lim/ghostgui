"""Lifecycle contracts for RViz-style displays, tools, and panels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .context import VisualizationContext


class LifecycleState(str, Enum):
    NEW = "new"
    INITIALIZED = "initialized"
    ENABLED = "enabled"
    DISABLED = "disabled"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class ToolEvent:
    kind: str
    payload: object = None


class VisualizationComponent:
    """Deterministic, idempotent component lifecycle.

    Subclasses implement the ``on_*`` hooks. State is advanced only after a
    hook succeeds, making failures observable to the manager.
    """

    component_kind = "component"

    def __init__(self, name: str):
        self.name = str(name).strip()
        if not self.name:
            raise ValueError("visualization component name cannot be empty")
        self.context: VisualizationContext | None = None
        self.state = LifecycleState.NEW

    def initialize(self, context: VisualizationContext) -> None:
        if self.state is not LifecycleState.NEW:
            return
        self.context = context
        try:
            self.on_initialize()
        except Exception:
            self.state = LifecycleState.FAILED
            raise
        self.state = LifecycleState.INITIALIZED

    def enable(self) -> None:
        if self.state is LifecycleState.ENABLED:
            return
        if self.state not in (
            LifecycleState.INITIALIZED,
            LifecycleState.DISABLED,
        ):
            raise RuntimeError(
                f"cannot enable {self.name} from state {self.state.value}"
            )
        try:
            self.on_enable()
        except Exception:
            self.state = LifecycleState.FAILED
            raise
        self.state = LifecycleState.ENABLED

    def disable(self) -> None:
        if self.state in (
            LifecycleState.NEW,
            LifecycleState.INITIALIZED,
            LifecycleState.DISABLED,
            LifecycleState.FAILED,
            LifecycleState.SHUTDOWN,
        ):
            return
        try:
            self.on_disable()
        except Exception:
            self.state = LifecycleState.FAILED
            raise
        self.state = LifecycleState.DISABLED

    def reset(self) -> None:
        if self.state in (LifecycleState.NEW, LifecycleState.SHUTDOWN):
            return
        self.on_reset()

    def shutdown(self) -> None:
        if self.state is LifecycleState.SHUTDOWN:
            return
        error = None
        try:
            if self.state is LifecycleState.ENABLED:
                try:
                    self.on_disable()
                except Exception as exc:
                    error = exc
            try:
                self.on_shutdown()
            except Exception as exc:
                if error is None:
                    error = exc
        finally:
            self.state = LifecycleState.SHUTDOWN
            self.context = None
        if error is not None:
            raise error

    def on_initialize(self) -> None:
        pass

    def on_enable(self) -> None:
        pass

    def on_disable(self) -> None:
        pass

    def on_reset(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass


class Display(VisualizationComponent):
    component_kind = "display"

    def update(self, update) -> None:
        if self.state is LifecycleState.ENABLED:
            self.on_update(update)

    def on_update(self, update) -> None:
        pass


class Tool(VisualizationComponent):
    component_kind = "tool"

    def handle_event(self, event: ToolEvent) -> bool:
        if self.state is not LifecycleState.ENABLED:
            return False
        return bool(self.on_event(event))

    def on_event(self, event: ToolEvent) -> bool:
        return False


class Panel(VisualizationComponent):
    component_kind = "panel"

    def update(self, update) -> None:
        if self.state is LifecycleState.ENABLED:
            self.on_update(update)

    def on_update(self, update) -> None:
        pass


class CallbackDisplay(Display):
    def __init__(self, name: str, callback: Callable[[object], None]):
        super().__init__(name)
        self._callback = callback

    def on_update(self, update) -> None:
        self._callback(update)


class CallbackPanel(Panel):
    def __init__(self, name: str, callback: Callable[[object], None]):
        super().__init__(name)
        self._callback = callback

    def on_update(self, update) -> None:
        self._callback(update)


class CallbackTool(Tool):
    def __init__(
        self,
        name: str,
        *,
        activated: Callable[[], None],
        deactivated: Callable[[], None] | None = None,
        event_handler: Callable[[ToolEvent], bool] | None = None,
    ):
        super().__init__(name)
        self._activated = activated
        self._deactivated = deactivated
        self._event_handler = event_handler

    def on_enable(self) -> None:
        self._activated()

    def on_disable(self) -> None:
        if self._deactivated is not None:
            self._deactivated()

    def on_event(self, event: ToolEvent) -> bool:
        if self._event_handler is None:
            return False
        return bool(self._event_handler(event))
