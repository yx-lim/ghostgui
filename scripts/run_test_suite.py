#!/usr/bin/env python3
"""Run tests with isolated writable application and XDG directories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def isolated_environment(runtime_root: Path, *, visual: bool = False):
    environment = os.environ.copy()
    environment.update(
        {
            "GHOSTGUI_CONFIG_DIR": str(runtime_root / "ghostgui-config"),
            "GHOSTGUI_CACHE_DIR": str(runtime_root / "ghostgui-cache"),
            "GHOSTGUI_PROJECTS_DIR": str(runtime_root / "ghostgui-projects"),
            "GHOSTGUI_USER_DATA_DIR": str(runtime_root / "ghostgui-data"),
            "XDG_CONFIG_HOME": str(runtime_root / "xdg-config"),
            "XDG_DATA_HOME": str(runtime_root / "xdg-data"),
            "XDG_CACHE_HOME": str(runtime_root / "xdg-cache"),
        }
    )
    if visual:
        environment["GHOSTGUI_VISUAL_TESTS"] = "1"
        environment.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    else:
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    return environment


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="test*.py",
        help="unittest discovery pattern",
    )
    parser.add_argument(
        "--start-directory",
        default="tests",
        help="unittest discovery start directory",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=(0, 1, 2),
        default=2,
    )
    parser.add_argument("--failfast", action="store_true")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="enable visual tests (normally run under xvfb-run)",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        args.start_directory,
        "-p",
        args.pattern,
    ]
    if args.verbosity == 2:
        command.append("-v")
    elif args.verbosity == 0:
        command.append("-q")
    if args.failfast:
        command.append("-f")
    with tempfile.TemporaryDirectory(prefix="ghostgui-tests-") as directory:
        runtime_root = Path(directory)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=isolated_environment(runtime_root, visual=args.visual),
            check=False,
        )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
