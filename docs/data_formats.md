# Data Formats

GhostGUI uses model-dependent MuJoCo `qpos` arrays for pose and trajectory
interchange. The active model determines the required array width and ordering.

## Qpos Pose CSV

A qpos pose file contains one headerless numeric row:

```text
qpos_0,qpos_1,...,qpos_n
```

The row must:

- contain exactly `model.nq` values for the active model;
- contain only finite numbers;
- use the MuJoCo qpos ordering compiled for that model.

For a floating free joint, MuJoCo stores three translation values followed by a
quaternion in `w, x, y, z` order. Remaining entries follow the compiled model's
joint qpos addresses. Do not assume that two different robots share a qpos
width or column order.

**File → Export → Qpos** writes the current committed pose. An orange preview is
not included.

## Headerless Trajectory CSV

The application trajectory format contains one row per sample:

```text
time,qpos_0,qpos_1,...,qpos_n
```

Requirements:

- time is measured in seconds;
- time cannot be negative;
- every row contains `1 + model.nq` finite values;
- times are nondecreasing;
- every qpos uses the active model's compiled ordering.

**File → Export → Trajectory** writes this format from the generated trajectory,
or from the editable qpos timeline when no generated trajectory is active.

## Named Backend CSV

Trajectory generation also produces a named solver CSV for the MuJoCo
simulation view. Its header begins with:

```text
time,base_x,base_y,base_z,base_qw,base_qx,base_qy,base_qz,...
```

The remaining columns are the active backend's joint names. This format is
useful when consumers need semantic columns rather than raw qpos addresses.
Floating-base orientation uses `w, x, y, z` quaternion order.

## Import Behavior

Qpos import reads the first non-empty row and rejects a header or incorrect
width.

Trajectory import accepts headerless time-plus-qpos rows and requires
nondecreasing times. The standalone MuJoCo viewer also accepts named backend
CSV output.

## Project Folder

A GhostGUI project is a folder ending in `.ghostgui`. Its
`ghostgui_project.json` metadata currently uses schema version 2 and points to
the target trajectory, qpos timeline, workspace, snapshot, and session log by
relative path.

All metadata paths must stay inside the project folder. Absolute paths,
parent-directory traversal, duplicate destinations, and symlink escapes are
rejected when the project opens.

Schema version 1 projects are migrated in memory when opened and written as
version 2 on the next save. Projects from a newer unsupported schema are
rejected without modification.

A manual save stages the target-frame JSON, qpos NPZ, workspace JSON, and
project metadata as one recoverable transaction. If replacement fails, the
previous coherent set is restored. If the process stops during replacement,
the transaction journal is recovered before project metadata is read.
Autosaves use the same transaction mechanism and publish their manifest last.

## Examples

Repository examples are grouped by purpose:

```text
csv/qpos/        single-pose qpos files
csv/trajectory/  trajectory files
```

Treat each example as model-specific. Confirm its intended robot before loading
it.

## Precision

Qpos exports use scientific notation with 18 digits after the decimal point.
Trajectory time values are written with six digits after the decimal point.
Consumers should parse numeric values rather than relying on their textual
formatting.
