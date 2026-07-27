# GhostGUI Documentation

This directory contains the current user and developer documentation for
GhostGUI.

## Get Started

- [Installation](install.md) covers platform setup and launch commands.
- [User Guide](user_guide.md) walks through the interface and first motion.
- [Troubleshooting](troubleshooting.md) covers common setup and runtime errors.

## Understand The Workflow

- [Preview And Keyframe Concepts](concepts.md) explains temporary, committed,
  and generated states.
- [Data Formats](data_formats.md) defines qpos and trajectory CSV files.
- [Models](models.md) describes bundled models, assets, and runtime caching.

## Configure And Extend

- [Adding Models](adding_models.md) explains URDF and MuJoCo XML imports.
- [Advanced IK](advanced_ik.md) documents solver tasks, weights, and collision
  behavior.

## Develop GhostGUI

- [Architecture](architecture.md) maps the source tree and data flow.
- [Migration Guide](migration.md) covers project and integration upgrades.
- [Operations Guide](operations.md) covers runtime paths, recovery, and health
  checks.
- [Reliability And Quality Gates](reliability.md) records the baseline and
  automated release contracts.
- [Testing](testing.md) lists automated and manual validation.
- [Contributing](../CONTRIBUTING.md) defines the development and review
  workflow.
- [Changelog](../CHANGELOG.md) records user-visible changes.

## Documentation Conventions

The public workflow uses these canonical terms:

| Term | Meaning |
| --- | --- |
| Orange preview | Temporary unsaved robot pose |
| Preview Path | Validate the transition without saving |
| Keyframe | Committed robot and target state at a time |
| Commit Keyframe | Store the current pose as a keyframe |
| Keyframe interval | Amount of time advanced after a commit |

Keep user-facing documentation task-oriented. Put implementation contracts in
the focused technical pages and historical investigations in `archive/`.
