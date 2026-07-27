"""Project, cache, and CSV path helpers shared by application layers."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile

from core.resources import (
    SOURCE_ROOT,
    bundled_resource_root,
    is_source_checkout,
)


# Compatibility constant for scripts and tests that need the Python source
# root. Read-only application assets use BUNDLED_DATA_ROOT instead.
PROJECT_ROOT = SOURCE_ROOT
BUNDLED_DATA_ROOT = bundled_resource_root()


def ghostgui_user_data_dir():
    override = os.environ.get("GHOSTGUI_USER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GhostGUI"
    if sys.platform.startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "GhostGUI"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "ghostgui"
    return Path.home() / ".local" / "share" / "ghostgui"


def writable_data_root():
    if os.environ.get("GHOSTGUI_USER_DATA_DIR"):
        return ghostgui_user_data_dir()
    # Preserve checkout-local examples and projects for source development.
    return PROJECT_ROOT if is_source_checkout(PROJECT_ROOT) else ghostgui_user_data_dir()


CSV_DIR = writable_data_root() / "csv"
QPOS_CSV_DIR = CSV_DIR / "qpos"
TRAJECTORY_CSV_DIR = CSV_DIR / "trajectory"


def ghostgui_cache_dir():
    override = os.environ.get("GHOSTGUI_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "GhostGUI"
    if sys.platform.startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "GhostGUI" / "Cache"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "ghostgui"
    return Path.home() / ".cache" / "ghostgui"


def mujoco_playback_cache_path():
    return ghostgui_cache_dir() / "playback" / "mujoco_playback.csv"


def csv_file_path(filename):
    return CSV_DIR / filename


def prepare_csv_save_path(csv_path):
    path = Path(csv_path).expanduser()
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    if not path.is_absolute():
        path = CSV_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


@contextmanager
def atomic_text_writer(path, *, newline=None):
    """Write a text file beside its destination and atomically replace it."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline=newline,
        ) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
