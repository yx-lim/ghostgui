"""
3D viewer panel.

For now, this launches the MuJoCo viewer in a separate process.

Reason:
    Embedding MuJoCo directly inside PySide6 is possible,
    but launching it separately is much simpler and more stable
    for the first version.
"""

from pathlib import Path
import csv
import shutil
import sys

from application.paths import mujoco_playback_cache_path
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QSlider,
    QDoubleSpinBox,
)


class Mujoco3DViewerPanel(QWidget):
    def __init__(self, adapter=None):
        super().__init__()

        self.process = None
        self._trajectory_regenerator = None
        self._missing_trajectory_notice_shown = False

        self.project_root = Path(__file__).resolve().parents[2]
        self.viewer_script = self.project_root / "scripts" / "view_g1_mujoco.py"
        self.adapter = adapter
        self.model_path = (
            adapter.runtime_model_path if adapter is not None
            else self.project_root / "models" / "g1_29dof.xml"
        )
        self.trajectory_csv_path = mujoco_playback_cache_path()
        self.trajectory_times = []
        self._syncing_timeline = False
        self.stdout_buffer = ""

        self.build_ui()
        self.load_timeline_metadata(self.trajectory_csv_path)

    def build_ui(self):
        layout = QVBoxLayout()

        display_name = self.adapter.model_name if self.adapter else "Unitree G1"
        self.title = QLabel(f"3D MuJoCo Viewer: {display_name}")
        shown_path = self.adapter.model_path if self.adapter else self.model_path
        self.model_label = QLabel(f"Model: {shown_path}")

        self.open_button = QPushButton("Open MuJoCo Viewer")
        self.close_button = QPushButton("Close MuJoCo Viewer")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.refresh_button = QPushButton("Refresh Viewer")

        self.time_label = QLabel("Time: 0.000 / 0.000 s")
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)

        self.speed_box = QDoubleSpinBox()
        self.speed_box.setDecimals(2)
        self.speed_box.setRange(0.05, 5.0)
        self.speed_box.setSingleStep(0.25)
        self.speed_box.setValue(1.0)
        self.speed_box.setSuffix("x")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.open_button.clicked.connect(self.open_viewer)
        self.close_button.clicked.connect(self.close_viewer)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.refresh_button.clicked.connect(self.refresh_viewer)
        self.timeline_slider.sliderMoved.connect(self.on_timeline_scrubbed)
        self.timeline_slider.sliderReleased.connect(self.on_timeline_released)
        self.speed_box.valueChanged.connect(self.on_speed_changed)

        window_row = QHBoxLayout()
        window_row.addWidget(self.open_button)
        window_row.addWidget(self.close_button)

        playback_row = QHBoxLayout()
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.pause_button)
        playback_row.addWidget(self.refresh_button)
        playback_row.addWidget(QLabel("Speed"))
        playback_row.addWidget(self.speed_box)

        layout.addWidget(self.title)
        layout.addWidget(self.model_label)
        self.trajectory_label = QLabel(
            f"Trajectory CSV: {self.trajectory_csv_path}"
        )
        layout.addWidget(self.trajectory_label)
        layout.addLayout(window_row)
        layout.addWidget(self.time_label)
        layout.addWidget(self.timeline_slider)
        layout.addLayout(playback_row)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def set_model_adapter(self, adapter):
        if self.process is not None:
            self.close_viewer()
        self.adapter = adapter
        self.model_path = adapter.runtime_model_path
        self.title.setText(f"3D MuJoCo Viewer: {adapter.model_name}")
        self.model_label.setText(f"Model: {adapter.model_path}")
        if adapter.load_warning:
            self.log_box.append(adapter.load_warning)

    def set_trajectory_regenerator(self, callback):
        self._trajectory_regenerator = callback

    def load_timeline_metadata(self, csv_path):
        self.trajectory_csv_path = Path(csv_path).expanduser().resolve()
        self.trajectory_label.setText(
            f"Trajectory CSV: {self.trajectory_csv_path}"
        )
        self.trajectory_times = []

        if not self.trajectory_csv_path.exists():
            self.timeline_slider.setMaximum(0)
            self.timeline_slider.setValue(0)
            self.time_label.setText("Time: 0.000 / 0.000 s")
            self._update_playback_availability()
            return

        try:
            self.trajectory_times = self._read_timeline_times(
                self.trajectory_csv_path
            )
        except Exception as exc:
            self.log_box.append(f"Could not read trajectory CSV: {exc}")
            self.trajectory_times = []

        max_index = max(0, len(self.trajectory_times) - 1)
        self.timeline_slider.setMaximum(max_index)
        self.timeline_slider.setValue(0)
        self.update_time_label(0)
        self._missing_trajectory_notice_shown = False
        self._update_playback_availability()

    def _active_qpos_width(self):
        if self.adapter is None:
            return None
        home_qpos = getattr(self.adapter, "home_qpos", None)
        if home_qpos is None:
            return None
        return len(home_qpos)

    def _read_timeline_times(self, csv_path):
        with open(csv_path, "r", newline="") as f:
            raw_rows = [
                row
                for row in csv.reader(f)
                if any(cell.strip() for cell in row)
            ]

        if not raw_rows:
            return []

        header = [cell.strip() for cell in raw_rows[0]]
        if "time" in header:
            time_column = header.index("time")
            return [
                float(row[time_column])
                for row in raw_rows[1:]
                if len(row) > time_column and row[time_column].strip()
            ]

        numeric_rows = [
            [float(cell.strip()) for cell in row]
            for row in raw_rows
        ]
        qpos_width = self._active_qpos_width()
        row_widths = {len(row) for row in numeric_rows}

        if qpos_width is not None and row_widths == {qpos_width + 1}:
            return [row[0] for row in numeric_rows]
        if qpos_width is not None and row_widths == {qpos_width}:
            return [float(index) for index in range(len(numeric_rows))]
        if self._first_numeric_column_looks_like_time(numeric_rows):
            return [row[0] for row in numeric_rows]
        return [float(index) for index in range(len(numeric_rows))]

    def _first_numeric_column_looks_like_time(self, numeric_rows):
        if any(len(row) < 2 for row in numeric_rows):
            return False
        times = [row[0] for row in numeric_rows]
        return all(
            earlier <= later
            for earlier, later in zip(times, times[1:])
        )

    def set_trajectory_csv(self, csv_path):
        """
        Called by the main window after Generate exports a fresh trajectory.
        """
        self.load_timeline_metadata(csv_path)

        if self.process is not None:
            self.send_command(f"load {Path(csv_path).resolve()}")

    def set_trajectory_metadata(self, csv_path, times):
        """Apply already parsed metadata without reading the CSV again."""
        self.trajectory_csv_path = Path(csv_path).expanduser().resolve()
        self.trajectory_label.setText(
            f"Trajectory CSV: {self.trajectory_csv_path}"
        )
        self.trajectory_times = [float(time) for time in times]
        max_index = max(0, len(self.trajectory_times) - 1)
        self.timeline_slider.setMaximum(max_index)
        self.timeline_slider.setValue(0)
        self.update_time_label(0)
        self._missing_trajectory_notice_shown = False
        self._update_playback_availability()
        if self.process is not None and self.trajectory_csv_path.exists():
            self.send_command(f"load {self.trajectory_csv_path}")

    def clear_trajectory(self):
        playback_cache = mujoco_playback_cache_path().resolve()
        try:
            playback_cache.unlink(missing_ok=True)
        except OSError as exc:
            self.log_box.append(f"Could not clear playback cache: {exc}")
        self.trajectory_csv_path = playback_cache
        self.trajectory_times = []
        self.trajectory_label.setText(
            f"Trajectory CSV: {self.trajectory_csv_path}"
        )
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setValue(0)
        self.time_label.setText("Time: 0.000 / 0.000 s")
        self._update_playback_availability()
        if self.process is not None:
            self.send_command("clear")

    def _update_playback_availability(self):
        available = bool(
            self.trajectory_times and self.trajectory_csv_path.exists()
        )
        self.play_button.setEnabled(available)
        self.pause_button.setEnabled(available)
        self.timeline_slider.setEnabled(available)
        self.speed_box.setEnabled(available)

    def _ensure_trajectory_file(self):
        if self.trajectory_csv_path.exists():
            if self.trajectory_times:
                return True
            self.load_timeline_metadata(self.trajectory_csv_path)
            if self.trajectory_times:
                return True

        if self._trajectory_regenerator is not None:
            try:
                regenerated_path = self._trajectory_regenerator()
            except Exception as exc:
                if not self._missing_trajectory_notice_shown:
                    self.log_box.append(
                        f"Playback cache could not be restored: {exc}"
                    )
                regenerated_path = None
            if regenerated_path:
                candidate = Path(regenerated_path).expanduser().resolve()
                if candidate.exists():
                    self.load_timeline_metadata(candidate)
                    if self.process is not None:
                        self.send_command(f"load {candidate}")
                    self.log_box.append(
                        f"Restored playback cache: {candidate}"
                    )
                    return True

        if not self._missing_trajectory_notice_shown:
            self.log_box.append(
                "No playback trajectory is available. Generate or import "
                "a trajectory first."
            )
            self._missing_trajectory_notice_shown = True
        self._update_playback_availability()
        return False

    def update_time_label(self, index):
        if not self.trajectory_times:
            self.time_label.setText("Time: 0.000 / 0.000 s")
            return

        index = max(0, min(len(self.trajectory_times) - 1, int(index)))
        current_time = self.trajectory_times[index]
        duration = self.trajectory_times[-1]
        self.time_label.setText(
            f"Time: {current_time:.3f} / {duration:.3f} s"
        )

    def open_viewer(self):
        if not self.model_path.exists():
            self.log_box.append(f"Model file not found: {self.model_path}")
            return

        if not self.viewer_script.exists():
            self.log_box.append(f"Viewer script not found: {self.viewer_script}")
            return

        if self.process is not None:
            self.log_box.append("MuJoCo viewer is already running.")
            return

        trajectory_available = self._ensure_trajectory_file()
        self.process = QProcess(self)

        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.on_process_finished)

        self.log_box.append("Launching MuJoCo viewer...")

        arguments = [str(self.viewer_script), "--model", str(self.model_path)]
        if trajectory_available:
            arguments.extend(["--csv", str(self.trajectory_csv_path)])

        executable = self._viewer_python_executable()
        if executable is None:
            return

        self.process.start(executable, arguments)

    def _viewer_python_executable(self):
        if sys.platform != "darwin":
            return sys.executable

        executable_dir = Path(sys.executable).resolve().parent
        candidates = [
            executable_dir / "mjpython",
            shutil.which("mjpython"),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if path.exists():
                return str(path)

        self.log_box.append(
            "mjpython was not found. On macOS, MuJoCo launch_passive() must be "
            "started with mjpython. Re-run bash scripts/install_macos.sh from a "
            "native Python environment."
        )
        self.process = None
        return None

    def close_viewer(self):
        if self.process is None:
            self.log_box.append("No MuJoCo viewer process is running.")
            return

        self.log_box.append("Closing MuJoCo viewer...")
        self.process.terminate()

    def shutdown(self, timeout_ms=1000):
        """Terminate the child viewer and escalate to kill after a timeout."""
        process = self.process
        if process is None:
            return True
        process.terminate()
        if not process.waitForFinished(max(0, int(timeout_ms))):
            process.kill()
            process.waitForFinished(max(0, int(timeout_ms)))
        stopped = process.state() == QProcess.ProcessState.NotRunning
        if stopped:
            self.process = None
            self.stdout_buffer = ""
        return stopped

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def send_command(self, command):
        if self.process is None:
            self.log_box.append("Open the MuJoCo viewer before playback.")
            return

        self.process.write((command + "\n").encode("utf-8"))

    def play(self):
        if not self._ensure_trajectory_file():
            return
        self.send_command("play")

    def pause(self):
        self.send_command("pause")

    def refresh_viewer(self):
        index = self.timeline_slider.value()
        trajectory_available = self._ensure_trajectory_file()
        self.load_timeline_metadata(self.trajectory_csv_path)
        index = min(index, self.timeline_slider.maximum())
        self.timeline_slider.setValue(index)

        if trajectory_available:
            self.send_command(f"load {self.trajectory_csv_path.resolve()}")

        self.send_command(f"seek {index}")
        self.send_command("refresh")

    def on_speed_changed(self, speed):
        self.send_command(f"speed {speed:.3f}")

    def on_timeline_scrubbed(self, index):
        self.update_time_label(index)

    def on_timeline_released(self):
        index = self.timeline_slider.value()
        self.send_command(f"seek {index}")

    def read_stdout(self):
        if self.process is None:
            return

        output = bytes(self.process.readAllStandardOutput()).decode("utf-8")
        self.stdout_buffer += output
        lines = self.stdout_buffer.splitlines(keepends=True)
        self.stdout_buffer = ""

        if lines and not lines[-1].endswith(("\n", "\r")):
            self.stdout_buffer = lines.pop()

        for line in lines:
            self.handle_process_line(line.strip())

    def read_stderr(self):
        if self.process is None:
            return

        output = bytes(self.process.readAllStandardError()).decode("utf-8")
        self.log_box.append(output.strip())

    def on_process_finished(self):
        self.log_box.append("MuJoCo viewer closed.")
        self.process = None
        self.stdout_buffer = ""

    def handle_process_line(self, line):
        if not line:
            return

        if line.startswith("STATE "):
            self.apply_state_line(line)
            return

        self.log_box.append(line)

    def apply_state_line(self, line):
        fields = {}

        for part in line.split()[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value

        try:
            index = int(fields.get("index", "0"))
            count = int(fields.get("count", "0"))
            current_time = float(fields.get("time", "0.0"))
            duration = float(fields.get("duration", "0.0"))
        except ValueError:
            return

        if count > 0 and count != len(self.trajectory_times):
            self.timeline_slider.setMaximum(count - 1)

        if not self.timeline_slider.isSliderDown():
            self._syncing_timeline = True
            self.timeline_slider.setValue(max(0, index))
            self._syncing_timeline = False

        if self.trajectory_times:
            self.update_time_label(index)
        else:
            self.time_label.setText(
                f"Time: {current_time:.3f} / {duration:.3f} s"
            )
