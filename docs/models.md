# Models

GhostGUI registers robot source files, logical target frames, home-pose hints,
and mesh-package mappings in `core/models/registry.py`.

## Bundled Models

| Key | Display name | Type | Source file | Logical targets |
| --- | --- | --- | --- | --- |
| `g1` | Unitree G1 | Humanoid | `models/g1_29dof.xml` | Pelvis, torso, hands, feet |
| `go2` | Unitree Go2 | Quadruped | `models/go2_description.urdf` | Base, trunk, four feet |
| `h2` | Unitree H2 | Humanoid | `models/h2.urdf` | Pelvis, torso, hands, feet |
| `z1` | Unitree Z1 | Manipulator | `models/z1.urdf` | Base, wrist, tool |

The exact body, site, joint, and qpos names come from the compiled MuJoCo model.
Logical targets are model-specific aliases over those objects.

## Source And Runtime Models

MuJoCo XML sources load directly.

URDF sources are prepared before use:

1. Referenced mesh paths are resolved.
2. COLLADA visuals are converted into cached OBJ material parts when needed.
3. Visual geometry is retained alongside collision geometry.
4. A floating root, editor sites, lighting, ground, and home keyframe are added
   to the generated MuJoCo XML.
5. The generated model is compiled once to surface errors early.

The public model path remains the selected source file even when GhostGUI uses a
generated runtime XML internally.

## Runtime Cache

Prepared URDF models are stored below:

```text
~/.cache/ghostgui/models/
```

Set `GHOSTGUI_CACHE_DIR` to override the GhostGUI cache root.

The cache key includes:

- GhostGUI's model-cache format version;
- the MuJoCo version;
- the source URDF contents;
- referenced mesh names and contents, or their resolution errors.

Changing one of those inputs creates a different cache entry. Deleting the cache
is safe; GhostGUI rebuilds it when the model is selected again.

## Mesh Formats

MuJoCo-compatible direct mesh formats are:

- STL
- OBJ
- MSH

GhostGUI can convert triangulated COLLADA (`.dae`) visuals using a `Z_UP`
coordinate system. Missing, malformed, or unsupported assets produce an
explicit load error rather than silently substituting unrelated geometry.

## Imported Models

The application copies imported `.urdf` or `.xml` sources into the repository's
`models/` library and copies their meshes into a neighboring
`models/assets-<name>/` directory. Imported models use generic root and target
discovery unless they are registered explicitly in code.

See [Adding Models](adding_models.md) for the import workflow and
[Troubleshooting](troubleshooting.md#model-loading) for common failures.
