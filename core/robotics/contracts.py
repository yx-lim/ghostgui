"""Model-independent coordinate, qpos, and time-series contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.math3d import ANGLE_UNIT, QUATERNION_ORDER


POSITION_UNIT = "meters"
TIME_UNIT = "seconds"


@dataclass(frozen=True)
class QposContract:
    width: int

    def __post_init__(self):
        if int(self.width) <= 0:
            raise ValueError("qpos width must be positive")
        object.__setattr__(self, "width", int(self.width))

    def validate(self, values, *, context="qpos") -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape != (self.width,):
            raise ValueError(
                f"{context} must contain {self.width} values, "
                f"found shape {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{context} contains a non-finite value")
        return array.copy()


def validate_trajectory_arrays(times, qposes, qpos_width):
    contract = QposContract(qpos_width)
    normalized_times = tuple(float(value) for value in times)
    if not np.all(np.isfinite(normalized_times)):
        raise ValueError("trajectory times contain a non-finite value")
    if any(value < 0.0 for value in normalized_times):
        raise ValueError("trajectory times cannot be negative")
    if any(
        earlier > later
        for earlier, later in zip(normalized_times, normalized_times[1:])
    ):
        raise ValueError("trajectory times must be nondecreasing")
    normalized_qposes = tuple(
        contract.validate(qpos, context=f"trajectory qpos row {index + 1}")
        for index, qpos in enumerate(qposes)
    )
    if len(normalized_times) != len(normalized_qposes):
        raise ValueError("trajectory times and qpos rows must have equal length")
    return normalized_times, normalized_qposes
