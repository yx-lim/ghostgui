"""Non-blocking launcher for GhostGUI's synchronous file-selector process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QProcess, QTimer, Signal


class SynchronousFileSelectionStage(QObject):
    """Run a blocking Qt file picker outside the main GUI process."""

    active_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._selected_callback = None
        self._failed_callback = None
        self._cancelled_callback = None
        self._callback_pending = False
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.cancel)

    def is_active(self):
        return self._process is not None or self._callback_pending

    def select_file(
        self,
        *,
        mode,
        title,
        directory,
        name_filter,
        selected,
        failed=None,
        cancelled=None,
        filename=None,
    ):
        if self.is_active():
            return False

        arguments = [
            "-m",
            "application.file_dialog_helper",
            "--mode",
            mode,
            "--title",
            title,
            "--directory",
            str(directory),
            "--name-filter",
            name_filter,
        ]
        if filename:
            arguments.extend(("--filename", filename))

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(arguments)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        process.finished.connect(
            lambda exit_code, exit_status, candidate=process: (
                self._on_finished(candidate, exit_code, exit_status)
            )
        )
        process.errorOccurred.connect(
            lambda error, candidate=process: (
                self._on_process_error(candidate, error)
            )
        )

        self._process = process
        self._selected_callback = selected
        self._failed_callback = failed
        self._cancelled_callback = cancelled
        self.active_changed.emit(True)
        process.start()
        return True

    def cancel(self):
        process = self._process
        self._selected_callback = None
        self._failed_callback = None
        self._cancelled_callback = None
        self._callback_pending = False
        if process is None:
            self.active_changed.emit(False)
            return
        self._process = None
        process.finished.connect(process.deleteLater)
        # This process owns only a picker. Kill it immediately during app
        # shutdown so a stuck desktop portal cannot outlive or delay GhostGUI.
        process.kill()
        process.waitForFinished(1000)
        self.active_changed.emit(False)

    def _on_finished(self, process, exit_code, _exit_status):
        if process is not self._process:
            return

        stdout = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        stderr = bytes(process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        ).strip()
        process.deleteLater()
        self._process = None

        selected_path = ""
        error_message = ""
        try:
            payload = json.loads(stdout.strip() or "{}")
            selected_path = str(payload.get("selected") or "")
            error_message = str(payload.get("error") or "")
        except (TypeError, ValueError) as exc:
            error_message = f"Invalid file-selector response: {exc}"

        if exit_code != 0 and not error_message:
            error_message = stderr or f"File selector exited with code {exit_code}."

        selected_callback = self._selected_callback
        failed_callback = self._failed_callback
        cancelled_callback = self._cancelled_callback
        self._selected_callback = None
        self._failed_callback = None
        self._cancelled_callback = None

        if selected_path and selected_callback is not None:
            self._callback_pending = True
            QTimer.singleShot(
                0,
                lambda: self._deliver_selection(
                    selected_callback, selected_path
                ),
            )
            return
        if error_message and failed_callback is not None:
            failed_callback(error_message)
        elif not error_message and cancelled_callback is not None:
            cancelled_callback()
        self.active_changed.emit(False)

    def _on_process_error(self, process, _error):
        if process is not self._process:
            return
        if process.state() != QProcess.ProcessState.NotRunning:
            return
        message = process.errorString() or "Could not start the file selector."
        failed_callback = self._failed_callback
        self._process = None
        self._selected_callback = None
        self._failed_callback = None
        self._cancelled_callback = None
        process.deleteLater()
        if failed_callback is not None:
            failed_callback(message)
        self.active_changed.emit(False)

    def _deliver_selection(self, callback, path):
        if not self._callback_pending:
            return
        try:
            callback(path)
        finally:
            self._callback_pending = False
            self.active_changed.emit(False)
