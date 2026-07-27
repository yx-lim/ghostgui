# GhostGUI

![GhostGUI editing a robot trajectory](ghostgui.png)

GhostGUI is a desktop application for authoring robot poses and trajectories
against MuJoCo models.

## Overview

GhostGUI lets robotics developers and researchers edit end-effector targets or
joint angles, inspect the resulting pose on a live 3D model, and build a motion
from time-based keyframes.

The application includes Unitree G1, Go2, H2, and Z1 models. It can generate,
play, and export robot trajectories for validation or downstream use.

## Highlights

- Edit end effectors with a 3D transform gizmo and whole-body IK.
- Edit model joint angles directly.
- Review temporary changes on a semi-transparent orange preview.
- Validate motion and collision constraints before committing a keyframe.
- Generate and play trajectories from time-based keyframes.
- Import MuJoCo XML or URDF models and export qpos CSV data.

## Quick Start

Linux/Ubuntu is the primary tested platform.

```bash
git clone https://github.com/yx-lim/ghostgui.git
cd ghostgui
bash scripts/install_linux.sh
bash scripts/run_linux.sh
```

After activating the virtual environment, you can also launch the application
directly:

```bash
ghostgui
ghostgui --model go2
```

macOS and Windows scripts are available but remain experimental. See the
[installation guide](docs/install.md) for platform-specific instructions and
troubleshooting.

## Basic Workflow

1. Choose a robot and target frame.
2. Drag the target or edit its joint angles.
3. Review the orange preview.
4. Optionally use **Preview Path** to validate the motion.
5. Select **Commit Keyframe**.
6. Repeat at another time, then generate, play, or export the trajectory.

## Supported Models

| Model | Type | Source | Bundled visuals |
| --- | --- | --- | --- |
| Unitree G1 | Humanoid | MuJoCo XML | STL |
| Unitree Go2 | Quadruped | URDF | COLLADA |
| Unitree H2 | Humanoid | URDF | STL |
| Unitree Z1 | Manipulator | URDF | COLLADA |

See [Models](docs/models.md) for model loading, asset, cache, and import details.

## Documentation

- [Documentation index](docs/README.md)
- [Installation](docs/install.md)
- [User guide](docs/user_guide.md)
- [Preview and keyframe concepts](docs/concepts.md)
- [Data formats](docs/data_formats.md)
- [Models](docs/models.md)
- [Adding models](docs/adding_models.md)
- [Advanced IK](docs/advanced_ik.md)
- [Architecture](docs/architecture.md)
- [Migration guide](docs/migration.md)
- [Operations guide](docs/operations.md)
- [Testing](docs/testing.md)
- [Troubleshooting](docs/troubleshooting.md)

## Project Status

GhostGUI is at version `0.1.0`. Linux/Ubuntu is the primary tested platform;
macOS and Windows support is experimental. The transform gizmo is currently
world-aligned, and imported models may require manual mesh-path selection.

For current constraints and workarounds, see
[Troubleshooting](docs/troubleshooting.md).

## Citation

Research users can cite the project using [CITATION.cff](CITATION.cff).
