"""Status widgets and compact viewer-status presentation helpers."""

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel


STATUS_ICONS = {
    "info": "●",
    "success": "✓",
    "warning": "⚠",
    "error": "✕",
}

_DETAIL_FIELD_LABELS = (
    ("frame", "Frame"),
    ("accepted", "Accepted movement"),
    ("ik error", "IK error"),
    ("tasks", "Active tasks"),
    ("model", "Model"),
)
_DETAIL_FIELD_KEYS = {key for key, _label in _DETAIL_FIELD_LABELS}


@dataclass(frozen=True)
class StatusEvent:
    """One user-facing status event with optional technical details."""

    severity: str
    title: str
    message: str = ""
    details: str = ""

    @property
    def icon(self):
        return STATUS_ICONS.get(self.severity, STATUS_ICONS["info"])

    @property
    def signature(self):
        """Stable summary identity used to coalesce live status updates."""
        return self.severity, self.title, self.message


def _status_parts(text):
    parts = []
    for line in str(text).splitlines():
        parts.extend(part.strip() for part in line.split(";") if part.strip())
    return parts


def _status_severity(text):
    lower = str(text).lower()
    risk_text = lower.replace("collision-free", "")
    if any(
        token in risk_text
        for token in (
            "cannot ",
            "could not",
            "failed",
            "failure",
            "unknown robot",
            "non-finite",
            "outside limits",
        )
    ):
        return "error"
    if any(
        token in risk_text
        for token in (
            "blocked",
            "collision",
            " is empty",
            "near singularity",
            "not selectable",
            "no editable",
        )
    ) or lower.startswith("no "):
        return "warning"
    if lower.startswith(
        (
            "accepted",
            "applied",
            "cleared",
            "deleted",
            "discarded",
            "generated",
            "imported",
            "loaded",
            "opened",
            "planned",
            "saved",
        )
    ) or " ready" in lower:
        return "success"
    return "info"


def _normal_status_event(text):
    raw = str(text).strip() or "Ready."
    parts = _status_parts(raw)
    title = parts[0] if parts else raw
    message = ""
    if len(parts) == 2 and len(parts[1]) <= 64:
        message = parts[1]

    details = raw
    return StatusEvent(
        severity=_status_severity(raw),
        title=title,
        message=message,
        details=details,
    )


def _verbose_ik_status_event(raw, parts, fields):
    lower = raw.lower()
    frame = fields.get("frame", "selected frame").replace("_", " ")
    accepted = fields.get("accepted", "")
    notes = []
    for part in parts:
        key = part.split("=", 1)[0].strip().lower() if "=" in part else ""
        if key not in _DETAIL_FIELD_KEYS:
            notes.append(part)

    if "collision warning" in lower:
        severity = "warning"
        title = "Preview warning"
        cause = next(
            (note for note in notes if "collision warning" in note.lower()),
            "The preview pose contains a collision.",
        )
        message = cause.rstrip(".") + "."
    elif "collision blocked" in lower:
        severity = "warning"
        title = "Preview blocked"
        cause = next(
            (note for note in notes if "collision blocked" in note.lower()),
            "Movement was stopped by a collision.",
        )
        message = cause.rstrip(".") + "."
    elif "ik blocked" in lower:
        severity = "warning"
        title = "Preview blocked"
        cause = next(
            (note for note in notes if "ik blocked" in note.lower()),
            "The IK solver could not apply the requested movement.",
        )
        message = cause.rstrip(".") + "."
    elif "near singularity" in lower:
        severity = "warning"
        title = "Preview warning"
        message = f"Near singularity while updating {frame}."
    else:
        severity = "info"
        title = "Preview updated"
        if accepted and accepted != "100%":
            message = f"Movement partially applied ({accepted}) for {frame}."
        elif "collision-free" in lower:
            message = f"{frame.capitalize()} pose is collision-free."
        else:
            message = f"Updated {frame} pose."

    detail_lines = []
    for note in notes:
        note_lower = note.lower()
        if note_lower.startswith(("axis ", "rot ", "move ")):
            label = "Edit"
        elif "converged" in note_lower or "ik blocked" in note_lower:
            label = "Solver"
        elif "collision" in note_lower:
            label = "Collision"
        elif note_lower.startswith("tcp "):
            label = "Mode"
        elif "near singularity" in note_lower:
            label = "Warning"
        elif "preview not committed" in note_lower:
            label = "State"
        elif note_lower.startswith("ready to "):
            label = "Next"
        else:
            label = "Result"
        detail_lines.append(f"{label}: {note}")

    for key, label in _DETAIL_FIELD_LABELS:
        value = fields.get(key)
        if not value:
            continue
        if key == "frame":
            value = value.replace("_", " ")
        if key == "ik error":
            value = f"{value} m"
        detail_lines.append(f"{label}: {value}")

    return StatusEvent(
        severity=severity,
        title=title,
        message=message,
        details="\n".join(detail_lines),
    )


def status_event_from_text(text):
    """Convert existing status strings into a compact, structured event."""
    raw = str(text).strip()
    parts = _status_parts(raw)
    fields = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key in _DETAIL_FIELD_KEYS:
            fields[key] = value.strip()

    verbose_fields = {"accepted", "ik error", "frame"}
    if verbose_fields.issubset(fields):
        return _verbose_ik_status_event(raw, parts, fields)
    return _normal_status_event(raw)


class StatusValueLabel(QLabel):
    text_changed = Signal(str)

    def setText(self, text):
        previous = self.text()
        super().setText(text)
        if text != previous:
            self.text_changed.emit(text)
