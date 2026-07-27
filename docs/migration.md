# Migration Guide

This guide covers project-file upgrades and the compatibility surfaces available
while integrations move to the current GhostGUI architecture.

## Project Schema Version 1 To 2

Opening a schema version 1 project performs an in-memory migration. The next
successful manual save writes schema version 2 and changes:

- `application.project_format` to `ghostgui.project.v2`;
- an `autosave_files` map for the autosave manifest and document files;
- a `persistence.transaction_format` value of
  `ghostgui.transaction.v1`.

Target-frame JSON, qpos NPZ, workspace JSON, and metadata are then published as
one transaction. The previous coherent files are restored if replacement
fails. Opening a project recovers an interrupted transaction before reading its
metadata.

Before upgrading an important project, close GhostGUI and copy the entire
`.ghostgui` folder. Do not copy only individual data files while the application
is saving. Projects with a schema newer than the installed application supports
are rejected without being rewritten.

## State And Command Ownership

New integrations should use these owners:

| Compatibility surface | Current owner |
| --- | --- |
| `window.trajectory` | `ProjectDocument.trajectory` |
| `window.active_index` | `ProjectDocument.active_index` |
| Direct keyframe-list mutation | Typed commands through `EditorController` |
| `RobotModelSession` | `EditorSession` |
| Ad hoc change callbacks | Typed events through `EditorEventBus` |
| Window-owned undo/redo lists | `HistoryStack` |

The window properties and `RobotModelSession` remain facades, so an integration
can migrate one workflow at a time. New write paths should create a command and
execute it through the controller; readers can use a document snapshot when
they need an independent copy.

## Visualization Extensions

New visualization behavior should implement one of the lifecycle contracts in
`application.visualization`:

- a `Display` consumes immutable scene updates;
- a `Tool` owns one interaction mode;
- a `Panel` presents status or controls;
- a `FramePoseProvider` resolves logical-frame world poses.

Register components with `VisualizationManager` and obtain shared services from
`VisualizationContext`. Avoid retaining the main window inside reusable
application components. Existing `RobotViewer3D.update_scene` and gizmo methods
remain available through GUI adapters during migration.

Components must tolerate repeated shutdown calls. Release timers, processes,
subscriptions, and graphics resources in their shutdown hook rather than
depending on Python object destruction.

## Robotics And Backend Contracts

The current common contracts are:

- distances in meters and angles in radians;
- quaternions in normalized `w, x, y, z` order;
- qpos arrays in the active MuJoCo model's compiled order and exact width;
- nonnegative finite trajectory time;
- shared pose-target solving through `core.ik.solve_pose_targets`.

Backend selection exposes exact and approximate capabilities. Use
`FallbackPolicy.REQUIRE_EXACT` when a workflow cannot accept approximation.
`FallbackPolicy.ALLOW_APPROXIMATE` retains compatibility behavior but reports a
degraded backend and warning instead of silently changing implementations.

## Resources And Writable Data

Do not derive models, theme files, or help paths from the process working
directory. Use `core.resources.resource_path` for bundled read-only files.

An installed wheel stores read-only resources below `share/ghostgui` and writes
user data to platform data directories. A source checkout keeps its development
defaults, while `GHOSTGUI_USER_DATA_DIR` explicitly overrides writable data in
both modes. Imported models use the checkout model library during development
and the user-data model library when installed.

## Background Work

Existing zero-argument work remains compatible with
`SerializedBackgroundJobs.submit`. Use `submit_handle` when existing
zero-argument work needs a cancellation handle. New long-running work should
use `submit_cancellable`, accept a `CancellationToken`, and check it between
bounded operations. Keep result callbacks on the GUI thread and make them safe
when the owning session has already closed.

## Migration Checklist

- Route mutations through `ProjectDocument` commands.
- Subscribe to typed events and retain the returned subscription handle.
- Use logical frame names and the common coordinate contracts.
- Declare whether backend approximation is acceptable.
- Resolve packaged resources through the resource service.
- Use lifecycle-managed visualization components.
- Add cooperative cancellation to bounded background stages.
- Test schema migration, interrupted-save recovery, installed-wheel resources,
  and shutdown behavior for the changed integration.
