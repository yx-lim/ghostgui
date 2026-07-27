"""Standalone native file picker used by GhostGUI's async launcher stage."""

from __future__ import annotations

import argparse
import json
import sys

from PySide6.QtWidgets import QApplication, QFileDialog


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("open", "save", "directory"),
        required=True,
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--name-filter", required=True)
    parser.add_argument("--filename")
    return parser.parse_args()


def _select_file(args):
    app = QApplication([sys.argv[0]])
    app.setApplicationName("GhostGUI File Selector")

    dialog = QFileDialog()
    # Keep the platform-native picker (including the desktop portal on
    # Wayland). It may block this helper, but never GhostGUI's GUI thread.
    dialog.setWindowTitle(args.title)
    dialog.setDirectory(args.directory)
    dialog.setNameFilter(args.name_filter)

    if args.mode == "save":
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setDefaultSuffix("csv")
        if args.filename:
            dialog.selectFile(args.filename)
    elif args.mode == "directory":
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    else:
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return ""
    selected_files = dialog.selectedFiles()
    return selected_files[0] if selected_files else ""


def main():
    try:
        selected = _select_file(_arguments())
        payload = {"selected": selected}
        exit_code = 0
    except Exception as exc:
        payload = {"error": str(exc)}
        exit_code = 1
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
