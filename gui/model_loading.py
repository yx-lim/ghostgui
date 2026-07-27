"""Background construction of immutable robot-model adapters."""

from PySide6.QtCore import QThread, Signal

from core.models import MuJoCoRobotAdapter


_DETACHED_LOADERS = set()


class ModelLoadThread(QThread):
    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, model_key, model_info=None, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.model_info = model_info

    def run(self):
        try:
            adapter = MuJoCoRobotAdapter(self.model_info or self.model_key)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(self.model_key, str(exc))
            return
        if not self.isInterruptionRequested():
            self.loaded.emit(self.model_key, adapter)

    def cancel_and_wait(self, timeout_ms=2000):
        """Request cancellation and retain slow native loads until they exit."""
        self.requestInterruption()
        if self.wait(max(0, int(timeout_ms))):
            return True
        self.setParent(None)
        _DETACHED_LOADERS.add(self)
        self.finished.connect(lambda: _DETACHED_LOADERS.discard(self))
        self.finished.connect(self.deleteLater)
        return False
