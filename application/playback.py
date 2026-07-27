"""Elapsed-time playback calculations independent of Qt and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Callable


@dataclass
class PlaybackClock:
    """Track monotonic elapsed time and wrap a bounded playback interval."""

    now: Callable[[], float] = monotonic
    last_tick: float | None = None

    def start(self) -> None:
        self.last_tick = float(self.now())

    def stop(self) -> None:
        self.last_tick = None

    def elapsed(self, fallback: float, supplied: float | None = None) -> float:
        if supplied is not None:
            elapsed = float(supplied)
        else:
            current = float(self.now())
            elapsed = (
                float(fallback)
                if self.last_tick is None
                else current - self.last_tick
            )
            self.last_tick = current
        if not isfinite(elapsed):
            raise ValueError("playback elapsed time must be finite")
        return max(0.0, elapsed)

    @staticmethod
    def advance(
        current_time: float,
        start_time: float,
        end_time: float,
        elapsed: float,
        speed: float = 1.0,
    ) -> float:
        values = tuple(
            float(value)
            for value in (current_time, start_time, end_time, elapsed, speed)
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("playback inputs must be finite")
        current_time, start_time, end_time, elapsed, speed = values
        if end_time <= start_time:
            raise ValueError("playback end time must be after start time")
        if elapsed < 0.0 or speed <= 0.0:
            raise ValueError("playback elapsed time and speed must be positive")
        next_time = current_time + elapsed * speed
        duration = end_time - start_time
        if next_time > end_time:
            next_time = start_time + ((next_time - start_time) % duration)
        return next_time
