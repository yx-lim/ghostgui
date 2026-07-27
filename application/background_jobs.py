"""Serialized, cancellable background work with GUI-thread result delivery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread
from typing import Callable

from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal


class JobCancelled(RuntimeError):
    """Raised by cooperative work when its cancellation token is set."""


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELIVERED = "delivered"


class CancellationToken:
    def __init__(self):
        self._cancelled = Event()

    @property
    def cancellation_requested(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self) -> None:
        if self.cancellation_requested:
            raise JobCancelled("background job was cancelled")


class JobHandle:
    """Thread-safe observation and cancellation handle returned by submit."""

    def __init__(self, identifier: int, name: str, token: CancellationToken):
        self.identifier = int(identifier)
        self.name = str(name)
        self.token = token
        self._state = JobState.QUEUED
        self._lock = Lock()

    def __bool__(self):
        return True

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    @property
    def done(self) -> bool:
        return self.state in {
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.DELIVERED,
        }

    def cancel(self) -> bool:
        with self._lock:
            if self._state in {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.DELIVERED,
            }:
                return False
            self._state = JobState.CANCEL_REQUESTED
            self.token.cancel()
            return True

    def _set_state(self, state: JobState) -> None:
        with self._lock:
            self._state = state


@dataclass(frozen=True)
class _BackgroundJob:
    handle: JobHandle
    work: Callable
    pass_token: bool


@dataclass(frozen=True)
class _Callbacks:
    succeeded: Callable
    failed: Callable
    cancelled: Callable | None = None


class SerializedBackgroundJobs(QObject):
    """Run jobs one at a time and deliver their results on the Qt GUI thread."""

    busy_changed = Signal(bool)
    callback_failed = Signal(str)
    _result_ready = Signal(int, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs = Queue()
        self._callbacks: dict[int, _Callbacks] = {}
        self._handles: dict[int, JobHandle] = {}
        self._state_lock = Lock()
        self._next_identifier = 1
        self._pending_count = 0
        self._closing = False
        self._active_handle: JobHandle | None = None
        self._result_ready.connect(
            self._deliver_result,
            Qt.ConnectionType.QueuedConnection,
        )
        self._thread = Thread(
            target=self._processing_loop,
            name="ghostgui-background-jobs",
            daemon=True,
        )
        self._thread.start()
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @property
    def closing(self) -> bool:
        with self._state_lock:
            return self._closing

    def is_busy(self):
        with self._state_lock:
            return self._pending_count > 0

    def submit(self, name, work, succeeded, failed):
        """Submit existing work and preserve the historical boolean result."""
        return bool(
            self.submit_handle(name, work, succeeded, failed)
        )

    def submit_handle(self, name, work, succeeded, failed):
        """Submit zero-argument work and return its cancellation handle."""
        return self._submit(
            name,
            work,
            succeeded,
            failed,
            cancelled=None,
            pass_token=False,
        )

    def submit_cancellable(
        self,
        name,
        work,
        succeeded,
        failed,
        cancelled=None,
    ):
        """Submit work accepting one :class:`CancellationToken` argument."""
        return self._submit(
            name,
            work,
            succeeded,
            failed,
            cancelled=cancelled,
            pass_token=True,
        )

    def _submit(
        self,
        name,
        work,
        succeeded,
        failed,
        *,
        cancelled,
        pass_token,
    ):
        if not all(callable(callback) for callback in (work, succeeded, failed)):
            raise TypeError("background work and result callbacks must be callable")
        with self._state_lock:
            if self._closing:
                return None
            identifier = self._next_identifier
            self._next_identifier += 1
            token = CancellationToken()
            handle = JobHandle(identifier, str(name), token)
            self._callbacks[identifier] = _Callbacks(
                succeeded,
                failed,
                cancelled,
            )
            self._handles[identifier] = handle
            was_idle = self._pending_count == 0
            self._pending_count += 1
        self._jobs.put(_BackgroundJob(handle, work, pass_token))
        if was_idle:
            self._safe_emit_busy(True)
        return handle

    def clear_pending(self):
        removed = []
        while True:
            try:
                job = self._jobs.get_nowait()
            except Empty:
                break
            if job is None:
                self._jobs.task_done()
                self._jobs.put(None)
                break
            job.handle.cancel()
            job.handle._set_state(JobState.CANCELLED)
            removed.append(job.handle.identifier)
            self._jobs.task_done()

        if not removed:
            return 0
        with self._state_lock:
            for identifier in removed:
                self._callbacks.pop(identifier, None)
                self._handles.pop(identifier, None)
            self._pending_count = max(0, self._pending_count - len(removed))
            became_idle = self._pending_count == 0
        if became_idle:
            self._safe_emit_busy(False)
        return len(removed)

    def cancel_all(self) -> None:
        with self._state_lock:
            active = self._active_handle
        if active is not None:
            active.cancel()
        self.clear_pending()

    def shutdown(self, timeout: float = 2.0):
        """Cancel outstanding work and wait briefly for the worker to exit."""
        with self._state_lock:
            already_closing = self._closing
            self._closing = True
            active = self._active_handle
        if active is not None:
            active.cancel()
        self.clear_pending()
        with self._state_lock:
            self._callbacks.clear()
            self._handles.clear()
            self._pending_count = 0
        if not already_closing:
            self._jobs.put(None)
        if self._thread is not current_thread() and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))
        self._safe_emit_busy(False)
        return not self._thread.is_alive()

    def _processing_loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                return
            handle = job.handle
            with self._state_lock:
                self._active_handle = handle
                closing = self._closing
            if closing:
                handle.cancel()
            result = None
            error = None
            try:
                handle.token.raise_if_cancelled()
                handle._set_state(JobState.RUNNING)
                result = (
                    job.work(handle.token)
                    if job.pass_token
                    else job.work()
                )
                handle.token.raise_if_cancelled()
                handle._set_state(JobState.SUCCEEDED)
            except JobCancelled as exc:
                error = exc
                handle._set_state(JobState.CANCELLED)
            except Exception as exc:
                error = exc
                handle._set_state(JobState.FAILED)
            finally:
                with self._state_lock:
                    if self._active_handle is handle:
                        self._active_handle = None
                self._jobs.task_done()
            if self.closing:
                continue
            try:
                self._result_ready.emit(handle.identifier, result, error)
            except RuntimeError:
                return

    def _deliver_result(self, identifier, result, error):
        with self._state_lock:
            callbacks = self._callbacks.pop(identifier, None)
            handle = self._handles.pop(identifier, None)
            if callbacks is None or handle is None:
                return
            self._pending_count = max(0, self._pending_count - 1)
            became_idle = self._pending_count == 0
        if became_idle:
            self._safe_emit_busy(False)

        try:
            if isinstance(error, JobCancelled):
                if callbacks.cancelled is not None:
                    callbacks.cancelled()
            elif error is None:
                callbacks.succeeded(result)
            else:
                callbacks.failed(error)
        except Exception as exc:
            try:
                self.callback_failed.emit(
                    f"Background job '{handle.name}' callback failed: {exc}"
                )
            except RuntimeError:
                pass
        finally:
            if handle.state is not JobState.CANCELLED:
                handle._set_state(JobState.DELIVERED)

    def _safe_emit_busy(self, busy: bool) -> None:
        try:
            self.busy_changed.emit(bool(busy))
        except RuntimeError:
            pass
