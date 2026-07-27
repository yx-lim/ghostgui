#!/usr/bin/env python3
"""Enforce GhostGUI's dependency direction without third-party tooling."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LAYERS = ("core", "application", "gui")
FORBIDDEN_LAYER_IMPORTS = {
    "core": frozenset({"application", "gui"}),
    "application": frozenset({"gui"}),
    "gui": frozenset(),
}
FORBIDDEN_CORE_IMPORTS = frozenset(
    {"OpenGL", "PyQt5", "PyQt6", "PySide2", "PySide6"}
)
COMPOSITION_ROOTS = frozenset({Path("application/launcher.py")})


@dataclass(frozen=True, order=True)
class ArchitectureViolation:
    path: Path
    line: int
    message: str

    def render(self, root: Path = PROJECT_ROOT) -> str:
        try:
            display_path = self.path.relative_to(root)
        except ValueError:
            display_path = self.path
        return f"{display_path}:{self.line}: {self.message}"


def _import_roots(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name.split(".", 1)[0] for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return (node.module.split(".", 1)[0],)
    return ()


def check_file(path: Path, layer: str) -> list[ArchitectureViolation]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        line = int(getattr(exc, "lineno", 1) or 1)
        return [ArchitectureViolation(path, line, f"cannot parse module: {exc}")]

    violations = []
    forbidden_layers = FORBIDDEN_LAYER_IMPORTS[layer]
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = None
    if relative_path in COMPOSITION_ROOTS:
        forbidden_layers = frozenset()
    for node in ast.walk(tree):
        for imported_root in _import_roots(node):
            if imported_root in forbidden_layers:
                violations.append(
                    ArchitectureViolation(
                        path,
                        int(getattr(node, "lineno", 1)),
                        f"{layer} must not import {imported_root}",
                    )
                )
            if layer == "core" and imported_root in FORBIDDEN_CORE_IMPORTS:
                violations.append(
                    ArchitectureViolation(
                        path,
                        int(getattr(node, "lineno", 1)),
                        f"core must remain UI/renderer independent ({imported_root})",
                    )
                )
    return violations


def validate_repository(root: Path = PROJECT_ROOT) -> list[ArchitectureViolation]:
    root = Path(root)
    violations = []
    for layer in SOURCE_LAYERS:
        layer_root = root / layer
        if not layer_root.is_dir():
            violations.append(
                ArchitectureViolation(
                    layer_root,
                    1,
                    f"required source layer is missing: {layer}",
                )
            )
            continue
        for path in sorted(layer_root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                violations.extend(check_file(path, layer))
    return sorted(violations)


def main() -> int:
    violations = validate_repository()
    if violations:
        print("Architecture validation failed:")
        for violation in violations:
            print(f"- {violation.render()}")
        return 1
    print("Architecture validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
