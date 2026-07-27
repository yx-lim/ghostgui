# Changelog

All notable user-visible changes to GhostGUI are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project currently reports version `0.1.0`; no release tag has been created
in this repository.

## Unreleased

### Added

- Focused guides for concepts, data formats, models, model import, advanced IK,
  architecture, migration, operations, reliability, testing, and
  troubleshooting.
- A documentation index, contribution guide, citation metadata, and repository
  documentation checks.
- An MIT license for the project.
- Document, session, controller, command, and typed-event application
  contracts.
- An RViz-inspired visualization context with lifecycle-managed displays,
  tools, panels, and logical frame-pose services.
- Transaction recovery, cancellation, shutdown, architecture, packaging, and
  visual regression tests.

### Changed

- Reworked the root README as a concise product landing page.
- Standardized the public workflow around **Preview Path**,
  **Commit Keyframe**, and **Keyframe interval**.
- Removed stale implementation investigations from the publishable repository.
- Made project saves and autosaves transactional with schema migration and
  project-contained path validation.
- Consolidated quaternion, coordinate, qpos, trajectory, collision, and IK
  contracts and made backend approximation explicit.
- Split history, playback, status, camera, loading, render progress, and IK
  panel ownership out of the large GUI facades.
- Added cancellable background work, deterministic session/process teardown,
  render coalescing, and context-bound OpenGL cleanup.
- Packaged model, mesh, theme, and help resources for installed wheels.

### Fixed

- Clarified the difference between raw qpos trajectory CSV and named backend
  CSV output.
- Clarified where imported model sources and generated runtime caches are
  stored.
- Corrected playback timing, logical-target edit restoration, and Go2 support
  contact acceptance defects from the documented baseline.
