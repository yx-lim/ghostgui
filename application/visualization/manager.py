"""Runtime manager for visualization displays, tools, and panels."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from .components import Display, LifecycleState, Panel, Tool, ToolEvent
from .context import VisualizationContext


@dataclass(frozen=True)
class VisualizationUpdate:
    """Immutable envelope passed through one visualization refresh."""

    scene: object
    revision: int = 0
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class VisualizationFailure:
    component_kind: str
    component_name: str
    operation: str
    error: Exception


class VisualizationManager:
    """Own component registration, activation, updates, and teardown."""

    def __init__(self, context: VisualizationContext):
        self.context = context
        self.displays: dict[str, Display] = {}
        self.tools: dict[str, Tool] = {}
        self.panels: dict[str, Panel] = {}
        self.active_tool_name: str | None = None
        self.failures: list[VisualizationFailure] = []
        self.initialized = False
        self.closed = False

    def register_display(self, display: Display, *, enabled: bool = True) -> Display:
        self._register(self.displays, display)
        if self.initialized:
            self._initialize_component(display, enabled=enabled)
        else:
            display._enable_after_initialize = bool(enabled)
        return display

    def register_tool(self, tool: Tool) -> Tool:
        self._register(self.tools, tool)
        if self.initialized:
            self._initialize_component(tool, enabled=False)
        return tool

    def register_panel(self, panel: Panel, *, enabled: bool = True) -> Panel:
        self._register(self.panels, panel)
        if self.initialized:
            self._initialize_component(panel, enabled=enabled)
        else:
            panel._enable_after_initialize = bool(enabled)
        return panel

    def _register(self, registry: dict, component) -> None:
        if self.closed:
            raise RuntimeError("visualization manager is shut down")
        if component.name in registry:
            raise ValueError(
                f"{component.component_kind} already registered: {component.name}"
            )
        registry[component.name] = component

    def initialize(self) -> None:
        if self.closed:
            raise RuntimeError("visualization manager is shut down")
        if self.initialized:
            return
        self.initialized = True
        for component in (*self.displays.values(), *self.panels.values()):
            enabled = bool(
                getattr(component, "_enable_after_initialize", True)
            )
            self._initialize_component(component, enabled=enabled)
        for tool in self.tools.values():
            self._initialize_component(tool, enabled=False)

    def set_display_enabled(self, name: str, enabled: bool) -> bool:
        display = self.displays[name]
        if not self.initialized:
            display._enable_after_initialize = bool(enabled)
            return True
        return self._set_enabled(display, enabled)

    def set_panel_enabled(self, name: str, enabled: bool) -> bool:
        panel = self.panels[name]
        if not self.initialized:
            panel._enable_after_initialize = bool(enabled)
            return True
        return self._set_enabled(panel, enabled)

    def select_tool(self, name: str | None) -> bool:
        if not self.initialized:
            raise RuntimeError("visualization manager is not initialized")
        if name == self.active_tool_name:
            return True
        previous = self.tools.get(self.active_tool_name)
        if previous is not None and not self._set_enabled(previous, False):
            return False
        self.active_tool_name = None
        if name is None:
            return True
        tool = self.tools[name]
        if not self._set_enabled(tool, True):
            return False
        self.active_tool_name = name
        return True

    def handle_tool_event(self, event: ToolEvent) -> bool:
        tool = self.tools.get(self.active_tool_name)
        if tool is None:
            return False
        try:
            return tool.handle_event(event)
        except Exception as exc:
            self._record_failure(tool, "event", exc)
            self.active_tool_name = None
            return False

    def update(self, update: VisualizationUpdate) -> tuple[VisualizationFailure, ...]:
        if self.closed:
            raise RuntimeError("visualization manager is shut down")
        if not self.initialized:
            raise RuntimeError("visualization manager is not initialized")
        start = len(self.failures)
        for component in (*self.displays.values(), *self.panels.values()):
            if component.state is not LifecycleState.ENABLED:
                continue
            try:
                component.update(update)
            except Exception as exc:
                self._record_failure(component, "update", exc)
        return tuple(self.failures[start:])

    def reset(self) -> tuple[VisualizationFailure, ...]:
        start = len(self.failures)
        for component in self._components():
            try:
                component.reset()
            except Exception as exc:
                self._record_failure(component, "reset", exc)
        return tuple(self.failures[start:])

    def shutdown(self) -> tuple[VisualizationFailure, ...]:
        if self.closed:
            return ()
        start = len(self.failures)
        for component in reversed(self._components()):
            try:
                component.shutdown()
            except Exception as exc:
                self._record_failure(component, "shutdown", exc)
        self.active_tool_name = None
        self.closed = True
        return tuple(self.failures[start:])

    def _components(self):
        return [
            *self.displays.values(),
            *self.tools.values(),
            *self.panels.values(),
        ]

    def _initialize_component(self, component, *, enabled: bool) -> bool:
        try:
            component.initialize(self.context)
            if enabled:
                component.enable()
            return True
        except Exception as exc:
            self._record_failure(component, "initialize", exc)
            return False

    def _set_enabled(self, component, enabled: bool) -> bool:
        try:
            if enabled:
                component.enable()
            else:
                component.disable()
            return True
        except Exception as exc:
            operation = "enable" if enabled else "disable"
            self._record_failure(component, operation, exc)
            return False

    def _record_failure(self, component, operation: str, error: Exception) -> None:
        if component.state is not LifecycleState.SHUTDOWN:
            component.state = LifecycleState.FAILED
        failure = VisualizationFailure(
            component_kind=component.component_kind,
            component_name=component.name,
            operation=operation,
            error=error,
        )
        self.failures.append(failure)
        self.context.report_status(
            f"{component.component_kind.title()} '{component.name}' "
            f"failed during {operation}: {error}"
        )
