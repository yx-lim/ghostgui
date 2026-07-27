"""Locate read-only resources in a checkout or an installed wheel."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR_ENV = "GHOSTGUI_RESOURCE_DIR"
INSTALLED_RESOURCE_PARTS = ("share", "ghostgui")


def installed_resource_root() -> Path:
    return Path(sysconfig.get_path("data")).joinpath(*INSTALLED_RESOURCE_PARTS)


def is_source_checkout(root: Path = SOURCE_ROOT) -> bool:
    root = Path(root)
    return (
        (root / "pyproject.toml").is_file()
        and (root / "models").is_dir()
        and (root / "gui" / "assets").is_dir()
    )


def bundled_resource_root() -> Path:
    override = os.environ.get(RESOURCE_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if is_source_checkout():
        return SOURCE_ROOT
    return installed_resource_root()


def resource_path(relative_path, *, required: bool = False) -> Path:
    relative_path = Path(relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"resource path must be relative: {relative_path}")
    root = bundled_resource_root().resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"resource path escapes its root: {relative_path}")
    if required and not path.exists():
        raise FileNotFoundError(
            f"Bundled GhostGUI resource not found: {relative_path} "
            f"(searched {root})"
        )
    return path
