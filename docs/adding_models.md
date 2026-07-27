# Adding Models

GhostGUI can import URDF and MuJoCo XML models into the checkout's model library.
For stable, named logical targets, developers can also add an explicit registry
entry.

## Import Through The Application

1. Open **File → Import → Model**.
2. Select a `.urdf` or `.xml` source.
3. If mesh resolution fails, select the external mesh folder when prompted.
4. Wait for GhostGUI to copy, prepare, and validate the model.
5. Select the newly registered model.

The import is staged in a temporary directory and validated with MuJoCo before
the source and assets are moved into `models/`.

## Import Output

For a source named `example.urdf`, the default output resembles:

```text
models/
├── example.urdf
└── assets-example/
    └── copied-mesh-files
```

If that name already exists, GhostGUI appends a numeric suffix. Imported models
are discovered from the `models/` directory on later launches.

## Mesh Resolution

GhostGUI tries:

- absolute and source-relative paths;
- ROS `package://` paths inferred from nearby directories;
- common `assets`, `dae`, and `meshes` directories;
- the optional mesh folder selected in the application;
- matching filenames below the selected mesh folder.

Supported mesh extensions are `.stl`, `.obj`, `.msh`, and convertible `.dae`.
COLLADA inputs must contain triangulated geometry and use `Z_UP`.

## Generic Discovery

An imported model without a registry entry uses generic hints:

- root candidates such as `base`, `base_link`, `trunk`, `pelvis`, `link00`,
  `root`, or `world`;
- root-joint candidates such as `floating_base`, `root`, or `freejoint`;
- bodies and sites exposed by the compiled MuJoCo model.

This is enough for direct joint editing and advanced target selection, but it
may not produce the preferred names or target set.

## Add A Registry Entry

For a maintained model, add a `RobotModelInfo` entry to
`core/models/registry.py` containing:

- a stable key and display name;
- model type and source path;
- root body and root joint candidates;
- logical frame aliases;
- any ROS package-to-directory mappings;
- optional home-joint values.

Add tests for model loading, logical-frame binding, qpos dimensions, and
representative visual assets. Update [Models](models.md) only after those tests
pass.

## Import Limitations

- Import writes into the repository checkout, so that directory must be
  writable.
- Texture images are not copied as a general material pipeline; prefer mesh
  geometry with direct material colors.
- Complex COLLADA features beyond triangulated geometry may require conversion
  to OBJ or STL before import.
- Logical frame semantics cannot always be inferred from arbitrary names.
