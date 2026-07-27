# GhostGUI User Guide

GhostGUI creates robot motion by editing target frames or joint angles,
previewing the result on a live MuJoCo model, committing keyframes, and
generating a trajectory for playback or export.

## Launch

From an installed checkout:

```bash
ghostgui --model g1
```

The bundled model keys are `g1`, `go2`, `h2`, and `z1`. See
[Installation](install.md) for complete setup and launch instructions.

## Core Concepts

**Robot model** determines the available bodies, sites, joints, logical target
frames, home pose, and 3D geometry.

**Target robot frame** is the body, site, or logical frame being edited, such as
`left_hand`, `right_foot`, `base`, or `tool`.

**Orange preview** is the temporary robot pose produced by the transform gizmo,
pose controls, or joint controls. It is not saved automatically.

**Committed state** is the accepted robot pose at the active time. Keyframes,
trajectory generation, and exports use committed states.

**Preview Path** validates the transition from the committed state to the orange
preview. It displays a ghost path but does not save anything.

**Commit Keyframe** records the current pose at the active time and advances by
the configured keyframe interval.

For the complete state model, see [Preview And Keyframe Concepts](concepts.md).

## First Motion Workflow

1. Choose a robot from the **Robot** menu.
2. Select a **Target robot frame**, such as `left_hand`.
3. Move the target with the 3D transform gizmo or the pose controls.
4. Inspect the orange preview and the **Status** panel.
5. Optionally select **Preview Path** to validate the transition.
6. Select **Commit Keyframe** to save the pose at the active time.
7. Move to another time and commit another keyframe.
8. Select **Generate**.
9. Use playback to inspect the motion.
10. Use **File → Export** to save a pose or trajectory.

The main workflow is:

```text
select target → edit → orange preview → Commit Keyframe → Generate → Export
```

## Interface

### Menu Bar

Use the menu bar to create, open, and save projects; select the active robot;
import or export data; switch views; and open help.

### Target

Choose a registered logical frame or use **Advanced target** to select another
body or site exposed by the active MuJoCo model.

### Editing Mode

Use **End Effector** to edit the target frame with X, Y, Z, Roll, Pitch, and Yaw
controls or the 3D transform gizmo.

Use **Joint Angles** to edit joints directly. The controls and 3D view remain
synchronized when switching modes, and both modes update the orange preview.

### Planning

The active time determines where the next keyframe is stored. The time slider
supports live scrubbing and playback. Releasing the slider selects an editable
time.

Planning controls include the keyframe interval, timeline duration, playback
speed, smoothing, collision substeps, and preview/playback opacity.

### Workflow Toolbar

**Preview Path** validates the transition to the orange preview without saving
it.

**Commit Keyframe** records the current pose at the active time.

**Generate** samples the saved keyframes into a robot trajectory.

**Play/Pause** controls the active generated or editable timeline.

**Reset** restores the model home pose at the active time.

**Clear** clears the editable trajectory.

**Move/Rotate** select the transform-gizmo mode.

**Gizmo** shows or hides the transform gizmo.

**Undo/Redo** navigate recorded editing history.

### Sidebars

Drag the dividers to resize the sidebars. Use the divider arrows or
**View → Left Sidebar** and **View → Right Sidebar** to collapse or restore them.
GhostGUI remembers the expanded widths and collapsed states.

The right sidebar contains a compact **Status** summary and the
**IK / Constraints** controls. Expand **Details** to inspect solver and
operation diagnostics.

## Keyboard And Mouse

### Transform Editing

- **T** switches the transform gizmo to translation.
- **R** switches the transform gizmo to rotation.
- **E** or **Esc** cancels the current transform drag.
- **Shift + drag** enables finer movement.
- **Ctrl + drag** snaps movement.

### Inline Values

- Drag a filled value control for continuous adjustment.
- Click either half to move by one logical step.
- Use arrow keys for one step and Page Up/Down for ten steps.
- Use Home/End for the minimum or maximum.
- Press Enter or F2, or double-click the value, to type directly.
- Press Enter to commit typed input or Esc to cancel it.

### History

- **Ctrl+Z** undoes the last recorded action.
- **Ctrl+Shift+Z** redoes the last undone action.

### Camera

- **Left drag** orbits unless a gizmo handle is active.
- **Right drag** pans.
- **Middle drag** or the mouse wheel zooms.

## Import And Export

Use **File → Import** for MuJoCo XML/URDF models, qpos poses, or trajectories.

Use **File → Export** for:

- **Qpos**: the committed pose as one headerless qpos row.
- **Trajectory**: time plus qpos values for each trajectory row.

An uncommitted orange preview is not exported. Select **Commit Keyframe** first
when the pose should become part of the saved motion.

See [Data Formats](data_formats.md) and [Adding Models](adding_models.md) for the
file contracts and import requirements.

## Common Problems

If the orange robot changed but an export did not, the pose is still only a
preview. Commit a keyframe and export again.

If an edit stops or fails, check **Status** and **IK / Constraints** for joint
limits, singularities, or collisions.

If generation does not match the intended motion, verify the keyframe times and
active target frame, then inspect playback before exporting.

See [Troubleshooting](troubleshooting.md) for installation, rendering, model,
and workflow diagnostics.

## Known Limitations

- Linux/Ubuntu is the primary tested platform.
- The transform gizmo is world-aligned; local-frame controls are not available.
- Imported models depend on resolvable mesh files and may require mesh-folder
  selection.
- IK priority numbers are descriptive metadata; the current solver uses one
  weighted task stack rather than strict null-space priority projection.
