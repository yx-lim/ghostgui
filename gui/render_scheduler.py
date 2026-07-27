"""Coalesce high-frequency render invalidations onto the Qt event loop."""

from PySide6.QtCore import QObject, QTimer


class RenderRequestCoalescer(QObject):
    def __init__(self, render_callback, parent=None):
        super().__init__(parent)
        if not callable(render_callback):
            raise TypeError("render callback must be callable")
        self._render_callback = render_callback
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.flush)

    @property
    def pending(self) -> bool:
        return self._timer.isActive()

    def request(self) -> bool:
        if self._closed:
            return False
        if not self._timer.isActive():
            self._timer.start()
        return True

    def flush(self) -> bool:
        if self._closed:
            return False
        self._timer.stop()
        self._render_callback()
        return True

    def shutdown(self) -> None:
        self._closed = True
        self._timer.stop()
