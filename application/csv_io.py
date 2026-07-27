"""Pure CSV parsing and writing used by GUI and background jobs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from application.paths import atomic_text_writer, prepare_csv_save_path
from core.robotics import QposContract, validate_trajectory_arrays


@dataclass(frozen=True)
class LoadedQpos:
    path: Path
    qpos: np.ndarray


@dataclass(frozen=True)
class LoadedTrajectory:
    path: Path
    times: tuple[float, ...]
    qposes: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TrajectoryExport:
    expected_qpos_count: int
    times: tuple[float, ...]
    qposes: tuple[np.ndarray, ...]
    source_name: str
    preview_active: bool


def read_qpos_csv(csv_path, expected_qpos_count):
    path = Path(csv_path).expanduser().resolve()
    with path.open("r", newline="") as handle:
        rows = [
            row
            for row in csv.reader(handle)
            if any(cell.strip() for cell in row)
        ]
    if not rows:
        raise ValueError("the file is empty")
    try:
        qpos = np.asarray(
            [float(cell.strip()) for cell in rows[0]],
            dtype=float,
        )
    except ValueError as exc:
        raise ValueError(
            "expected a headerless row containing only qpos numbers"
        ) from exc
    expected = int(expected_qpos_count)
    try:
        qpos = QposContract(expected).validate(qpos)
    except ValueError as exc:
        if qpos.shape != (expected,):
            raise ValueError(
                f"expected {expected} qpos values for this model, found {qpos.size}"
            ) from exc
        raise
    return LoadedQpos(path=path, qpos=qpos)


def read_trajectory_csv(csv_path, expected_qpos_count):
    path = Path(csv_path).expanduser().resolve()
    with path.open("r", newline="") as handle:
        rows = [
            row
            for row in csv.reader(handle)
            if any(cell.strip() for cell in row)
        ]
    if not rows:
        raise ValueError("the file is empty")

    expected = int(expected_qpos_count)
    qposes = []
    times = []
    for row_index, row in enumerate(rows, start=1):
        try:
            values = [float(cell.strip()) for cell in row]
        except ValueError as exc:
            raise ValueError(
                "expected headerless numeric rows containing time plus qpos"
            ) from exc
        if len(values) != expected + 1:
            raise ValueError(
                f"expected {expected + 1} values per trajectory row "
                f"(time plus {expected} qpos values), found "
                f"{len(values)} on row {row_index}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite value on row {row_index}")
        times.append(values[0])
        qposes.append(np.asarray(values[1:], dtype=float))

    times, qposes = validate_trajectory_arrays(times, qposes, expected)

    return LoadedTrajectory(
        path=path,
        times=tuple(times),
        qposes=tuple(qposes),
    )


def write_qpos_csv(csv_path, qpos):
    path = prepare_csv_save_path(csv_path)
    raw = np.asarray(qpos, dtype=float)
    if raw.ndim != 1:
        raise ValueError("qpos export must be a one-dimensional array")
    values = QposContract(raw.size).validate(raw)
    with atomic_text_writer(path, newline="") as handle:
        csv.writer(handle).writerow(f"{value:.18e}" for value in values)
    return path


def write_trajectory_csv(csv_path, export):
    path = prepare_csv_save_path(csv_path)
    expected = int(export.expected_qpos_count)
    times, qposes = validate_trajectory_arrays(
        export.times,
        export.qposes,
        expected,
    )
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.writer(handle)
        for time_value, qpos in zip(times, qposes):
            writer.writerow(
                [f"{time_value:.6f}"]
                + [f"{value:.18e}" for value in qpos]
            )
    return path
