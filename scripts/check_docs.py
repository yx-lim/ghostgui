#!/usr/bin/env python3
"""Validate GhostGUI's public Markdown and user-facing terminology."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "AGENTS.md",
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PUBLIC_LEGACY_PATTERN = re.compile(
    r"\b(?:slice|slices|time[- ]slice|time[- ]slices|"
    r"plan preview|accept preview|delete slice|slice step)\b",
    re.IGNORECASE,
)
UI_LEGACY_PATTERN = re.compile(
    r"\b(?:Slice|Time Slice|Plan Preview|Accept Preview|"
    r"Delete Slice|Slice step|saved slices)\b"
)
UI_COPY_FILES = (
    "gui/controls.py",
    "gui/help/help_content.py",
    "gui/main_window.py",
    "gui/robot_viewer_3d.py",
    "gui/tutorial/tutorial_steps.py",
)


def public_markdown_files(root: Path) -> list[Path]:
    files = [root / name for name in PUBLIC_ROOT_FILES]
    files.extend(sorted((root / "docs").glob("*.md")))
    return [path for path in files if path.is_file()]


def github_anchor(value: str) -> str:
    """Approximate GitHub's stable heading slug for local link validation."""
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[`*_~]", "", value)
    value = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", value)
    value = value.strip().lower()
    value = re.sub(r"[^\w\- ]", "", value)
    return re.sub(r"\s+", "-", value)


def heading_anchors(path: Path) -> set[str]:
    anchors = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        base = github_anchor(match.group(2))
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def split_link_target(raw_target: str) -> tuple[str, str]:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    # Markdown titles are not used in this repository. Splitting here keeps a
    # future quoted title from becoming part of the path.
    target = re.split(r'\s+["\']', target, maxsplit=1)[0]
    path_part, separator, anchor = target.partition("#")
    return unquote(path_part), unquote(anchor) if separator else ""


def validate_markdown(path: Path, root: Path) -> list[str]:
    issues = []
    text = path.read_text(encoding="utf-8")
    display = path.relative_to(root)
    lines = text.splitlines()

    if text and not text.endswith("\n"):
        issues.append(f"{display}: missing final newline")
    if any(line.rstrip() != line for line in lines):
        issues.append(f"{display}: trailing whitespace")

    headings = [
        (index, match)
        for index, line in enumerate(lines, start=1)
        if (match := HEADING_PATTERN.match(line))
    ]
    h1_count = sum(len(match.group(1)) == 1 for _, match in headings)
    if h1_count != 1:
        issues.append(f"{display}: expected one level-one heading, found {h1_count}")

    previous_level = 0
    for line_number, match in headings:
        level = len(match.group(1))
        if previous_level and level > previous_level + 1:
            issues.append(
                f"{display}:{line_number}: heading level jumps "
                f"from {previous_level} to {level}"
            )
        previous_level = level

    if text.count("```") % 2:
        issues.append(f"{display}: unbalanced fenced code block")

    legacy = PUBLIC_LEGACY_PATTERN.search(text)
    if legacy:
        line_number = text[:legacy.start()].count("\n") + 1
        issues.append(
            f"{display}:{line_number}: legacy public term {legacy.group(0)!r}"
        )

    for match in LINK_PATTERN.finditer(text):
        raw_target = match.group(1)
        if raw_target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, anchor = split_link_target(raw_target)
        target_path = (
            path.resolve() if not path_part else (path.parent / path_part).resolve()
        )
        try:
            target_path.relative_to(root.resolve())
        except ValueError:
            issues.append(f"{display}: link escapes repository: {raw_target}")
            continue
        if not target_path.exists():
            issues.append(f"{display}: broken link target: {raw_target}")
            continue
        if anchor and target_path.is_file() and target_path.suffix.lower() == ".md":
            if anchor not in heading_anchors(target_path):
                issues.append(f"{display}: broken link anchor: {raw_target}")

    return issues


def validate_docs_index(root: Path) -> list[str]:
    index_path = root / "docs" / "README.md"
    text = index_path.read_text(encoding="utf-8")
    linked = set()
    for match in LINK_PATTERN.finditer(text):
        path_part, _ = split_link_target(match.group(1))
        if not path_part or "://" in path_part:
            continue
        target = (index_path.parent / path_part).resolve()
        if target.suffix.lower() == ".md":
            linked.add(target)

    expected = {
        path.resolve()
        for path in (root / "docs").glob("*.md")
        if path.name != "README.md"
    }
    missing = sorted(expected - linked)
    return [
        f"docs/README.md: public page is not indexed: "
        f"{path.relative_to(root)}"
        for path in missing
    ]


def validate_ui_copy(root: Path) -> list[str]:
    issues = []
    for relative in UI_COPY_FILES:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        for match in UI_LEGACY_PATTERN.finditer(text):
            line_number = text[:match.start()].count("\n") + 1
            issues.append(
                f"{relative}:{line_number}: legacy UI term {match.group(0)!r}"
            )
    return issues


def validate_repository(root: Path = PROJECT_ROOT) -> list[str]:
    root = Path(root).resolve()
    issues = []
    for path in public_markdown_files(root):
        issues.extend(validate_markdown(path, root))
    issues.extend(validate_docs_index(root))
    issues.extend(validate_ui_copy(root))

    readme_lines = len((root / "README.md").read_text(encoding="utf-8").splitlines())
    if not 80 <= readme_lines <= 140:
        issues.append(
            f"README.md: expected 80-140 lines for the landing page, "
            f"found {readme_lines}"
        )
    return issues


def main() -> int:
    issues = validate_repository()
    if issues:
        print("Documentation validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Documentation validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
