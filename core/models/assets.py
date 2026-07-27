"""Robot mesh validation, package URI resolution, and COLLADA conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


DIRECT_MUJOCO_MESH_FORMATS = {".stl", ".obj", ".msh"}
CONVERTIBLE_MESH_FORMATS = {".dae"}
COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


@dataclass(frozen=True)
class ResolvedMeshAsset:
    reference: str
    path: Path | None
    error: str | None = None


@dataclass(frozen=True)
class ConvertedMeshPart:
    path: Path
    rgba: tuple[float, float, float, float]
    material_name: str


def resolve_mesh_path(mesh_filename, model_dir, package_map=None):
    """Resolve relative and ROS ``package://`` mesh references without cwd use."""
    reference = str(mesh_filename or "").strip()
    model_dir = Path(model_dir).resolve()
    package_map = {
        name: Path(root).resolve() for name, root in (package_map or {}).items()
    }
    if not reference:
        return ResolvedMeshAsset(reference, None, "empty mesh filename")

    candidates = []
    if reference.startswith("package://"):
        payload = reference[len("package://"):]
        package, separator, relative = payload.partition("/")
        if not separator:
            return ResolvedMeshAsset(
                reference, None, f"invalid package URI: {reference}"
            )
        package_root = package_map.get(package)
        if package_root is not None:
            relative_path = Path(relative)
            candidates.append(package_root / relative_path)
            # Some vendored descriptions flatten their dae/meshes directory.
            if relative_path.parts and relative_path.parts[0].lower() in {
                "dae", "meshes", "assets"
            }:
                candidates.append(package_root.joinpath(*relative_path.parts[1:]))
        relative_path = Path(relative)
        candidates.extend((
            model_dir / relative_path,
            model_dir / package / relative_path,
        ))
    else:
        path = Path(reference)
        candidates.append(path if path.is_absolute() else model_dir / path)

    # Common vendored layouts: models/assets-go2/base.dae or assets/base.stl.
    basename = Path(reference).name
    candidates.extend((
        model_dir / basename,
        model_dir / "assets" / basename,
        model_dir / "dae" / basename,
        model_dir / "meshes" / basename,
    ))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix not in DIRECT_MUJOCO_MESH_FORMATS | CONVERTIBLE_MESH_FORMATS:
                return ResolvedMeshAsset(
                    reference,
                    candidate,
                    f"unsupported mesh format {suffix or '<none>'}: {candidate}",
                )
            return ResolvedMeshAsset(reference, candidate)
    roots = ", ".join(str(path) for path in candidates[:4])
    return ResolvedMeshAsset(
        reference,
        None,
        f"unresolved mesh {reference!r}; checked {roots}",
    )


def validate_model_assets(model_path, package_map=None):
    """Return one explicit resolution result for each URDF/MJCF mesh reference."""
    model_path = Path(model_path).resolve()
    root = ET.parse(model_path).getroot()
    attribute = "filename" if root.tag == "robot" else "file"
    model_dir = model_path.parent
    if root.tag != "robot":
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir") if compiler is not None else None
        if meshdir:
            model_dir = (model_dir / meshdir).resolve()
    return [
        resolve_mesh_path(mesh.get(attribute), model_dir, package_map)
        for mesh in root.findall(".//mesh")
        if mesh.get(attribute)
    ]


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "material"


def _source_vectors(mesh, source_id):
    source = mesh.find(f"c:source[@id='{source_id}']", COLLADA_NS)
    if source is None:
        raise RuntimeError(f"COLLADA source {source_id!r} is missing")
    array = source.find("c:float_array", COLLADA_NS)
    if array is None or not array.text:
        raise RuntimeError(f"COLLADA source {source_id!r} has no float array")
    accessor = source.find("c:technique_common/c:accessor", COLLADA_NS)
    stride = int(accessor.get("stride", "3")) if accessor is not None else 3
    values = np.fromstring(array.text, sep=" ", dtype=float)
    if stride < 3 or values.size % stride:
        raise RuntimeError(f"Invalid COLLADA source stride for {source_id!r}")
    return values.reshape(-1, stride)[:, :3]


def _collada_material_colors(root):
    effects = {}
    for effect in root.findall(".//c:library_effects/c:effect", COLLADA_NS):
        color = effect.find(
            ".//c:profile_COMMON/c:technique/*/c:diffuse/c:color", COLLADA_NS
        )
        if color is None:
            color = effect.find(
                ".//c:profile_COMMON/c:technique/*/c:emission/c:color",
                COLLADA_NS,
            )
        values = np.fromstring(color.text or "", sep=" ") if color is not None else []
        if len(values) >= 3:
            rgba = tuple(float(value) for value in values[:4])
            effects[effect.get("id")] = rgba if len(rgba) == 4 else (*rgba, 1.0)
    materials = {}
    for material in root.findall(
        ".//c:library_materials/c:material", COLLADA_NS
    ):
        instance = material.find("c:instance_effect", COLLADA_NS)
        effect_id = (instance.get("url", "").lstrip("#") if instance is not None else "")
        materials[material.get("id")] = effects.get(effect_id, (0.65, 0.68, 0.75, 1.0))
    return materials


def convert_collada_to_obj_parts(source_path, output_dir):
    """Convert the simple triangulated COLLADA used by Go2 into MuJoCo OBJ parts."""
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source_path).getroot()
    unit = root.find("c:asset/c:unit", COLLADA_NS)
    unit_scale = float(unit.get("meter", "1")) if unit is not None else 1.0
    up_axis = root.findtext("c:asset/c:up_axis", default="Z_UP", namespaces=COLLADA_NS)
    if up_axis != "Z_UP":
        raise RuntimeError(
            f"Unsupported COLLADA up axis {up_axis!r} in {source_path}; expected Z_UP"
        )

    materials = _collada_material_colors(root)
    symbol_targets = {}
    for binding in root.findall(".//c:instance_material", COLLADA_NS):
        symbol_targets[binding.get("symbol")] = binding.get("target", "").lstrip("#")

    geometry_transforms = {}
    for node in root.findall(
        ".//c:library_visual_scenes/c:visual_scene//c:node", COLLADA_NS
    ):
        matrix_node = node.find("c:matrix", COLLADA_NS)
        matrix = np.eye(4)
        if matrix_node is not None and matrix_node.text:
            values = np.fromstring(matrix_node.text, sep=" ")
            if values.size != 16:
                raise RuntimeError(f"Invalid COLLADA node matrix in {source_path}")
            matrix = values.reshape(4, 4)
        for instance in node.findall("c:instance_geometry", COLLADA_NS):
            geometry_transforms[instance.get("url", "").lstrip("#")] = matrix

    parts = []
    for geometry in root.findall(
        ".//c:library_geometries/c:geometry", COLLADA_NS
    ):
        mesh = geometry.find("c:mesh", COLLADA_NS)
        if mesh is None:
            continue
        vertices_sources = {}
        for vertices in mesh.findall("c:vertices", COLLADA_NS):
            position = vertices.find("c:input[@semantic='POSITION']", COLLADA_NS)
            if position is not None:
                vertices_sources[vertices.get("id")] = position.get("source", "").lstrip("#")
        matrix = geometry_transforms.get(geometry.get("id"), np.eye(4))
        for part_index, triangles in enumerate(mesh.findall("c:triangles", COLLADA_NS)):
            inputs = triangles.findall("c:input", COLLADA_NS)
            vertex_input = next(
                (item for item in inputs if item.get("semantic") == "VERTEX"), None
            )
            if vertex_input is None:
                raise RuntimeError(f"COLLADA triangles have no VERTEX input: {source_path}")
            source_id = vertex_input.get("source", "").lstrip("#")
            position_source = vertices_sources.get(source_id)
            if not position_source:
                raise RuntimeError(f"COLLADA vertices source {source_id!r} is invalid")
            vertices = _source_vectors(mesh, position_source) * unit_scale
            homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
            vertices = (matrix @ homogeneous.T).T[:, :3]
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            vertex_offset = int(vertex_input.get("offset", "0"))
            indices_node = triangles.find("c:p", COLLADA_NS)
            raw = np.fromstring(indices_node.text or "", sep=" ", dtype=np.int64)
            if raw.size % (stride * 3):
                raise RuntimeError(f"Malformed COLLADA triangle indices: {source_path}")
            faces = raw.reshape(-1, 3, stride)[:, :, vertex_offset]
            used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
            local_vertices = vertices[used]
            local_faces = inverse.reshape(-1, 3) + 1
            symbol = triangles.get("material") or f"part_{part_index}"
            material_id = symbol_targets.get(symbol, symbol)
            rgba = materials.get(material_id, (0.65, 0.68, 0.75, 1.0))
            output_path = output_dir / (
                f"{source_path.stem}_{part_index}_{_safe_name(symbol)}.obj"
            )
            lines = [f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in local_vertices]
            lines.extend(f"f {a} {b} {c}\n" for a, b, c in local_faces)
            output_path.write_text("".join(lines), encoding="ascii")
            parts.append(ConvertedMeshPart(output_path, rgba, symbol))
    if not parts:
        raise RuntimeError(f"No triangulated geometry found in COLLADA file {source_path}")
    return parts


def prepare_urdf_visual_meshes(root, model_path, cache_dir, package_map=None):
    """Resolve URDF visuals and replace DAE meshes with cached OBJ submeshes."""
    model_path = Path(model_path).resolve()
    converted_files = {}
    visual_count = 0
    dae_count = 0
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                materials = visual.findall("material")
                for extra in materials[1:]:
                    visual.remove(extra)
                continue
            resolved = resolve_mesh_path(
                mesh.get("filename"), model_path.parent, package_map
            )
            if resolved.error:
                raise RuntimeError(resolved.error)
            visual_count += 1
            if resolved.path.suffix.lower() in DIRECT_MUJOCO_MESH_FORMATS:
                mesh.set("filename", str(resolved.path))
                materials = visual.findall("material")
                for extra in materials[1:]:
                    visual.remove(extra)
                continue

            dae_count += 1
            parts = converted_files.get(resolved.path)
            if parts is None:
                parts = convert_collada_to_obj_parts(
                    resolved.path, Path(cache_dir) / "meshes" / resolved.path.stem
                )
                converted_files[resolved.path] = parts
            origin = visual.find("origin")
            insert_at = list(link).index(visual)
            link.remove(visual)
            for offset, part in enumerate(parts):
                replacement = ET.Element("visual")
                if origin is not None:
                    replacement.append(ET.fromstring(ET.tostring(origin)))
                geometry = ET.SubElement(replacement, "geometry")
                ET.SubElement(geometry, "mesh", {"filename": str(part.path)})
                material = ET.SubElement(
                    replacement, "material", {"name": _safe_name(part.material_name)}
                )
                ET.SubElement(material, "color", {
                    "rgba": " ".join(f"{value:.6g}" for value in part.rgba)
                })
                link.insert(insert_at + offset, replacement)
    return visual_count, dae_count, sum(len(parts) for parts in converted_files.values())
