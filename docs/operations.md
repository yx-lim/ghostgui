# Operations Guide

This guide covers runtime paths, health checks, backup and recovery, headless
validation, and failure reporting for GhostGUI installations.

## Runtime Storage

GhostGUI separates read-only installed resources from writable state.

| Setting | Purpose |
| --- | --- |
| `GHOSTGUI_RESOURCE_DIR` | Explicit read-only resource root |
| `GHOSTGUI_USER_DATA_DIR` | CSV and installed user-model data |
| `GHOSTGUI_PROJECTS_DIR` | Default project parent |
| `GHOSTGUI_CONFIG_DIR` | Recent-project and UI configuration |
| `GHOSTGUI_CACHE_DIR` | Prepared models and MuJoCo playback cache |
| `XDG_CONFIG_HOME` | Linux configuration fallback |
| `XDG_DATA_HOME` | Linux writable-data fallback |
| `XDG_CACHE_HOME` | Linux cache fallback |

Installed resources normally resolve to the Python installation's
`share/ghostgui` directory. Source checkouts use their bundled repository
resources. Explicit environment overrides are useful for CI, containers, and
portable deployments.

Project folders and imported model sources are durable user data. Prepared URDF
models and playback CSV files are disposable caches.

## Startup Health Check

From the active environment:

```bash
python3 -m pip check
ghostgui --model g1
```

Confirm that G1 loads, the 3D view renders, the **Status** summary reports the
selected backend, and the process exits without a remaining MuJoCo viewer or
file-selector process.

For a release artifact, run the installed-package smoke procedure in
[Testing](testing.md#package-smoke-test). It verifies the console entry point,
registered models, theme and help resources, G1 compilation, visualization
startup, and shutdown from outside the source checkout.

## Project Backup

Close GhostGUI before taking a filesystem backup, then copy the complete
`.ghostgui` folder. A useful backup includes:

- `ghostgui_project.json`;
- the `data`, `workspace`, `autosave`, `metadata`, and `snapshots` directories;
- any project-specific exports needed by downstream consumers.

The application fsyncs staged files and publishes a journaled group during
manual save and autosave. A normal completed save leaves no
`.ghostgui-transactions` directory.

## Interrupted Save Recovery

Opening a project runs transaction recovery before metadata is parsed:

- a prepared or replacing transaction restores backups and removes files that
  did not exist before the save;
- a committed transaction keeps the new files and removes its transaction
  directory;
- a pre-journal staging directory is safe to discard because no destination
  was replaced;
- an invalid or unsafe journal stops opening and remains on disk for diagnosis.

If an invalid-journal error occurs:

1. close every GhostGUI process using that project;
2. copy the entire project folder to a separate recovery location;
3. preserve `.ghostgui-transactions` with the copy;
4. record the complete error and filesystem/storage event;
5. restore a known-good full-project backup or request a journal review.

Do not delete the transaction directory before preserving the evidence. Avoid
editing individual JSON or NPZ files into a mixture of saved generations.

## Cache Maintenance

The prepared-model cache is content-addressed. If a cached model is suspected,
close the application and remove only its entry below the configured cache
root. GhostGUI rebuilds it on the next load.

The MuJoCo playback CSV is also disposable and is regenerated from active solved
states when possible. Never treat cache contents as the only copy of a project
or imported model.

## Headless Validation

Run the isolated offscreen suite:

```bash
python3 scripts/run_test_suite.py
```

For a Linux visual gate, provide Xvfb and Mesa software OpenGL:

```bash
GHOSTGUI_VISUAL_TESTS=1 QT_QPA_PLATFORM=xcb \
  xvfb-run -a --server-args="-screen 0 1280x1024x24" \
  python3 -m unittest tests.test_visual_smoke -v
```

The offscreen suite does not require a valid `QOpenGLWidget` context. The Xvfb
gate does and also captures the composed application window.

## Shutdown Expectations

Normal shutdown stops autosave, cancels selectors and serialized work, requests
model-loader interruption, closes every cached editor session, releases OpenGL
resources, terminates the external MuJoCo process, and removes event
subscriptions.

A cancellation-aware background task should check its token often enough to
finish within the shutdown timeout. Native model construction cannot always
stop immediately; GhostGUI retains a slow loader object until its native call
returns so its thread is not destroyed while running.

## Incident Information

Include the following when reporting an operational failure:

- operating system, display server, GPU or software-renderer details;
- Python, PySide6, MuJoCo, and GhostGUI versions;
- installed-wheel or source-checkout mode;
- selected model key and source path;
- relevant environment overrides from the runtime-storage table;
- complete terminal and **Status** detail messages;
- project schema version and whether a transaction directory exists;
- the smallest safe project or model input that reproduces the issue.
