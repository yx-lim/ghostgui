# Preview And Keyframe Concepts

GhostGUI deliberately separates temporary edits from saved trajectory state.
Understanding that separation prevents accidental exports and makes timeline
behavior predictable.

## State Flow

```text
model home pose
      ↓
committed state at the active time
      ↓ edit target or joint angles
orange preview
      ├─ Preview Path → temporary validation ghosts
      ├─ cancel/time change → discard
      └─ Commit Keyframe → committed keyframe
                                ↓
                         trajectory generation
                                ↓
                         playback and export
```

## Committed State

The model-colored robot represents the pose accepted at the active timeline
time. It is the reference state for edits and the source used by keyframes,
generation, and pose export.

Moving to a time without an existing keyframe creates an editable state derived
from the timeline. Existing keyframes at other times remain independent.

## Orange Preview

Dragging the transform gizmo, changing a pose value, or changing a joint value
starts an orange preview. The preview is solved in a separate MuJoCo state, so
the committed robot remains fixed while the proposed result changes.

Collision contacts appear as warnings on the preview. A preview containing a
collision cannot be committed. Changing the selected time discards an
uncommitted preview.

Preview opacity affects only the orange robot. It does not change the Qt window
or scene background opacity.

## Preview Path

**Preview Path** samples the transition between the committed pose and the
orange preview. It checks finite qpos values, joint limits, and collisions at
intermediate samples.

When validation succeeds, GhostGUI displays temporary path ghosts. The command
does not change a keyframe, committed state, or export.

## Commit Keyframe

**Commit Keyframe** performs the save operation for the active time:

1. It rejects a non-finite or colliding orange preview.
2. It copies the preview into the committed MuJoCo state.
3. It records the committed qpos state at the active time.
4. It captures the registered logical target frames for trajectory generation.
5. It advances by the configured keyframe interval when possible.

If there is no active preview, the current committed pose is recorded directly.
Committing at an existing time updates that keyframe rather than adding a second
state at the same time.

## Reset And Delete

**Reset** is a one-time action at the active time. It pauses playback, cancels
an active drag, and restores the selected model's home qpos.

**Delete Keyframe** removes both the logical target data and qpos state stored at
the active time when present. It does not delete neighboring keyframes.

## Generation And Playback

Generation samples the logical target tracks created by the keyframes, applies
the configured smoothing, and asks the active backend to solve robot states.
The generated states can then be played in the live view or the MuJoCo
simulation view.

Playback speed changes wall-clock viewing speed, not the trajectory timestamps.
Playback ghosts and Preview Path ghosts are temporary visualizations and are
not exported as additional states.

## Export Rule

Exports use committed or generated states. An orange preview is intentionally
excluded. If a proposed pose should appear in an export, commit its keyframe
first.
