# Architecture

GhostGUI is a Python/PySide6 desktop application with MuJoCo-backed model,
kinematics, collision, trajectory, and playback services.

## Data Flow

```text
PySide6 controls and 3D gizmo
              ↓
       separate preview state
              ↓
   IK, limits, and collision checks
              ↓
        committed qpos state
              ↓
 logical target and qpos keyframes
              ↓
      trajectory generation
              ↓
 live playback / MuJoCo simulation / CSV
```

## Source Responsibilities

| Path | Responsibility |
| --- | --- |
| `application/` | Use cases, background jobs, projects, import/export, generation |
| `core/models/` | Registry, model preparation, MuJoCo metadata, mutable state |
| `core/ik/` | IK tasks and collision-aware drag solving |
| `core/trajectory/` | Keyframes, sampling, interpolation, smoothing |
| `gui/` | Main window, sidebars, widgets, help, tutorial, 3D interaction |
| `backend/` | Optional native robot simulation implementation |
| `models/` | Bundled and imported model sources and assets |
| `scripts/` | Installation, launch, and standalone viewer entry points |
| `tests/` | Unit, contract, GUI, model, and trajectory tests |

The enforced dependency direction is:

```text
core <- application <- gui
```

Core cannot import Qt, OpenGL, application, or GUI code. Application services
can use core contracts but cannot import GUI modules. GUI code assembles and
adapts both layers. `application.launcher` is the explicit composition-root
exception that imports the main window after parsing command-line arguments.

## Application Assembly

`application.launcher:main` parses the model key, creates the Qt application,
and opens `RobotGuiMainWindow`.

The main window maintains one session per loaded model. A session groups the
MuJoCo adapter, backend, logical-frame reference provider, 3D viewer, editable
trajectory, and active selection. Model loading runs outside the GUI thread.

## Model Layer

`MuJoCoRobotAdapter` is the model-agnostic facade. It exposes compiled MuJoCo
joints, limits, free roots, logical frames, home qpos, kinematic relationships,
and state factories.

URDF preparation is content-addressed and produces a cached runtime XML.
MuJoCo XML sources load directly. See [Models](models.md) for the asset and cache
contract.

## Editor State

`ProjectDocument` is the authoritative target-keyframe document for one model
session. `EditorController` applies typed commands and publishes typed events;
the Qt window reacts to those events without becoming the mutation gateway.
Each `EditorSession` groups a document with its adapter, backend, reference
provider, viewer, and qpos timeline.

`RobotViewer3D` currently owns distinct committed, preview, and playback
MuJoCo states:

- committed state is authoritative at the active time;
- preview state isolates temporary edits;
- playback state samples motion without changing the edit time.

`RobotStateTimeline` stores qpos keyframes. The application trajectory stores
logical target-frame keyframes used by the backend. A commit updates both
representations for the same time.

## Trajectory Generation

The trajectory layer interpolates target positions and orientations. Quaternion
orientation interpolation uses SLERP. The generation service samples tracks at
a uniform interval and sends them to the active backend.

The Python MuJoCo backend solves target frames with Jacobian IK and returns named
robot configurations. The application can also play raw qpos trajectories
directly.

## Files And Projects

Project workspaces live below `projects/` and persist application state in
`ghostgui_project.json`. CSV import/export is isolated in
`application/csv_io.py`; project and cache paths are centralized in
`application/paths.py`.

Project schema migrations and safe relative-path validation run before project
files are resolved. Manual saves and autosaves use a staged journal with
rollback and startup recovery. Source-checkout paths, installed read-only
resources under `share/ghostgui`, and writable user-data paths are separate
contracts.

The UI should not write files directly when an application service already owns
the relevant workflow.

## Visualization Runtime

The visualization layer follows RViz's separation between a stable runtime
context and lifecycle-managed extensions:

```text
VisualizationManager
    ├── VisualizationContext
    │   ├── active ProjectDocument provider
    │   ├── logical FramePoseProvider
    │   ├── render request callback
    │   └── status and named services
    ├── Displays: robot scene and timeline markers
    ├── Tools: move and rotate
    └── Panels: editor/backend status
```

The contracts in `application/visualization/` are Qt-free. Components move
through new, initialized, enabled/disabled, failed, and shutdown states. The
manager isolates update failures so one display cannot stop unrelated panels.
The adapters in `gui/visualization/` preserve `RobotViewer3D` as a compatibility
facade while new rendering and interaction features target the smaller
display/tool/panel interfaces.

Frame poses use logical model names, world coordinates in meters, and normalized
`wxyz` quaternions. `RobotFramePoseProvider` resolves those names against the
active session's adapter and committed or preview state; unknown and unavailable
frames raise `FramePoseError` instead of returning fabricated transforms.

## UI Ownership During Migration

`RobotGuiMainWindow` remains the compatibility facade and Qt composition root,
but focused components now own the mechanics it previously embedded:

| Component | Ownership |
| --- | --- |
| `application/history.py` | Bounded undo/redo transitions |
| `application/playback.py` | Monotonic elapsed-time and wrapping calculations |
| `gui/history.py` | GUI snapshot schema used by history |
| `gui/model_loading.py` | Off-GUI-thread adapter construction |
| `gui/render_progress.py` | Render-progress overlay behavior |
| `gui/panels/status_panel.py` | Status summary and diagnostic-detail widgets |
| `gui/visualization/` | Main-window display/tool/panel adapters |

`RobotViewer3D` similarly retains its public API while delegating the advanced
IK inspector builders to `gui/viewers/ik_panels.py` and playback math to the
application clock. `RobotCanvas3D` delegates orbit, pan, zoom, and camera basis
math to `gui/viewers/camera.py`. Reusable trajectory input widgets live under
`gui/widgets/trajectory_controls.py`; `gui.controls` re-exports their existing
names so third-party callers can migrate incrementally.

The intended direction for new work is: extend a focused component first, then
keep only thin signal wiring or a compatibility method on the large facade.

## Runtime And Resource Lifecycles

Serialized background work supports cooperative `CancellationToken` checks.
Queued and active work is cancelled during shutdown, late results are dropped
after the callback registry closes, and callback failures cannot corrupt the
busy count. Legacy zero-argument work continues to use the boolean-returning
`submit`; callers that need a handle use `submit_handle`, and
cancellation-aware work uses `submit_cancellable`.

The main window teardown order is autosave and file selectors, visualization
components, model loaders, background jobs, cached editor sessions, the
external MuJoCo process, then event subscriptions. An `EditorSession` closes
its viewer once and detaches its qpos timeline.

`RobotCanvas3D` connects cleanup to the owning OpenGL context's
`aboutToBeDestroyed` signal. Display-list identifiers are deduplicated before
deletion and the GLU quadric is released once. A render-request coalescer
collapses repeated main-scene invalidations into one event-loop repaint; it
drops pending work after shutdown.

## Decision Summary

| Decision | Consequence |
| --- | --- |
| One `ProjectDocument` per editor session | Target-keyframe ownership is independent of widgets |
| Commands plus typed events | Mutations are testable and subscribers cannot roll them back |
| Compatibility facades during migration | Existing callers remain runnable while ownership moves |
| Transactional multi-file persistence | A project opens as one coherent saved generation |
| Strict path and schema validation | Unsafe, ambiguous, and newer data fails before resolution |
| Shared robotics and IK contracts | Units, qpos width, quaternion order, and solver behavior agree |
| Explicit backend fallback policy | Approximation is observable and exact workflows can reject it |
| RViz-inspired component lifecycles | Displays, tools, and panels initialize and shut down predictably |
| Context-bound graphics cleanup | OpenGL identifiers are released while their context still exists |
| Isolated and installed-artifact CI | Local state and source paths cannot hide release failures |

These decisions favor a runnable migration over a single large rewrite. A new
component should have one state owner, explicit inputs and outputs, focused
tests, and an idempotent teardown path.
