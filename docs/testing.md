# Testing

Run automated tests before submitting changes and perform focused GUI checks
for behavior that depends on rendering or interaction.

## Automated Suite

From a checkout, use the isolated runner so local application settings,
projects, caches, and XDG state cannot affect results:

```bash
python3 scripts/run_test_suite.py
```

The suite covers model adapters, IK and collision behavior, timeline semantics,
CSV playback, transaction recovery, cancellation and shutdown, visualization
lifecycles, transform-gizmo math, GUI controls, themes, status handling,
packaging contracts, and trajectory smoothing.

Run documentation validation separately:

```bash
python3 scripts/check_docs.py
python3 scripts/check_architecture.py
```

## Focused Tests

Use a test module while iterating:

```bash
python3 -m unittest tests.test_robot_viewer_timeline -v
python3 -m unittest tests.test_advanced_ik -v
python3 -m unittest tests.test_model_resources -v
```

Choose the test that owns the changed contract rather than relying only on a
manual launch.

## Package Smoke Test

Build and inspect a wheel, then install it into a clean environment. Run the
smoke script from outside the checkout so source files cannot mask missing
package data:

```bash
python3 -m pip wheel . --no-deps --no-build-isolation --wheel-dir wheelhouse
python3 scripts/check_wheel.py --require-resources wheelhouse/*.whl
python3 -m venv --system-site-packages /tmp/ghostgui-wheel-smoke
/tmp/ghostgui-wheel-smoke/bin/python -m pip install --no-deps wheelhouse/*.whl
cd /tmp
/tmp/ghostgui-wheel-smoke/bin/python \
  /path/to/ghostgui/scripts/smoke_installed_package.py --load-model --gui
```

The smoke gate verifies the console entry point, all registered model sources,
theme and documentation resources, G1 compilation, visualization startup, and
clean GUI shutdown.

## Headless And Visual Gates

The regular suite uses Qt's offscreen platform. CI additionally starts an Xvfb
display with Mesa software OpenGL and runs:

```bash
GHOSTGUI_VISUAL_TESTS=1 QT_QPA_PLATFORM=xcb \
  python3 -m unittest tests.test_visual_smoke -v
```

That test requires a valid OpenGL context, captures the composed window, checks
that it is nonblank and opaque, and compares stable toolbar, tab, and sidebar
structure. It skips outside the explicitly configured visual environment.

## Manual GUI Checks

### Shared Workflow

- Start GhostGUI and confirm the selected model renders.
- Edit an end effector and confirm only the orange preview moves.
- Use **Preview Path** and confirm no keyframe is created.
- Use **Commit Keyframe** and confirm the timeline advances by the configured
  keyframe interval.
- Edit another time and confirm the earlier keyframe remains unchanged.
- Generate, play, pause, scrub, and export the trajectory.
- Confirm preview opacity does not make the application window transparent.
- Confirm reset affects the active time only.

### Model Checks

- **G1:** confirm humanoid hand, foot, torso, and pelvis targets and detailed
  geometry.
- **Go2:** confirm twelve scalar leg joints, four foot targets, the ground, and
  COLLADA-derived visual geometry.
- **H2:** confirm the humanoid targets and bundled STL visuals.
- **Z1:** confirm base, wrist, and tool targets and six arm joints.
- Switch away from a model and back; confirm its editor session is preserved.

### Import And Export

- Load a qpos file with the correct active-model width.
- Reject a qpos or trajectory with the wrong width or non-finite values.
- Export a committed pose and confirm an uncommitted orange preview is absent.
- Import a model with relative meshes.
- Exercise the mesh-folder retry path for an unresolved model asset.

### Path Independence

Launch the compatibility script from a different working directory:

```bash
cd /tmp
python3 /path/to/ghostgui/scripts/run_gui.py --model go2
```

Registry and model paths should remain anchored to the checkout.

## Documentation Checks

Documentation validation should reject:

- broken relative links;
- missing linked files or anchors;
- more than one level-one heading per public page;
- legacy user-facing workflow terminology;
- undocumented public pages;
- trailing whitespace.
