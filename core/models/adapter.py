"""Shared MuJoCo-backed robot metadata and state adapter."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .model import RobotModel3D
from .registry import RobotModelInfo, get_model_info
from .assets import prepare_urdf_visual_meshes, resolve_mesh_path


MODEL_CACHE_VERSION = 3
HOME_GROUND_CLEARANCE = 0.002


class MuJoCoRobotAdapter(RobotModel3D):
    """Model-agnostic facade used by GhostGUI's controls and viewers."""

    def __init__(self, model: str | RobotModelInfo = "g1", model_path=None):
        if model is None and model_path is not None:
            path = Path(model_path).resolve()
            model = RobotModelInfo(
                key=path.stem,
                display_name=path.stem,
                model_type="generic",
                model_path=path,
                root_body_candidates=(
                    "base", "base_link", "trunk", "pelvis", "link00",
                    "root", "world",
                ),
            )
        self.info = get_model_info(model)
        self.model_name = self.info.display_name
        self.model_type = self.info.model_type
        self.model_path = Path(model_path or self.info.model_path).resolve()
        self.asset_root = self.model_path.parent
        self.package_map = dict(self.info.package_map)
        self.load_warning = None
        self.runtime_model_path = self._prepare_model_path(self.model_path)
        super().__init__(self.runtime_model_path)
        # Public paths describe the selected source, not an implementation cache.
        self.model_path = Path(model_path or self.info.model_path).resolve()
        self._apply_registered_home()
        self._ground_home_qpos()
        self.default_qpos = self.home_qpos.copy()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_data.qpos[:] = self.home_qpos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.joint_names = self.get_joint_names()
        self.actuated_joints = list(self.joint_names)
        self.joint_limits = {
            name: self.get_joint_limits(name) for name in self.actuated_joints
        }
        self.root_body = self._first_body(self.info.root_body_candidates)
        self.root_joint = self._first_joint(self.info.root_joint_candidates)
        self.logical_frame_bindings = self._build_logical_frames()
        self.end_effectors = {
            key: value for key, value in self.logical_frame_bindings.items()
            if key.lower().endswith(("foot", "hand"))
        }
        self.trajectory_frames = list(self.logical_frame_bindings)
        self.kinematic_tree = self._build_kinematic_tree()

    @classmethod
    def load_model(cls, model_path, model=None):
        return cls(model=model, model_path=model_path)

    def _prepare_model_path(self, path: Path) -> Path:
        if path.suffix.lower() != ".urdf":
            return path
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise RuntimeError(f"Invalid URDF {path}: {exc}") from exc
        cache_key = self._urdf_cache_key(path, root)
        cache_root = Path(os.environ.get(
            "GHOSTGUI_CACHE_DIR",
            Path.home() / ".cache" / "ghostgui",
        ))
        cache_dir = cache_root / "models" / cache_key
        runtime_path = cache_dir / "model.xml"
        metadata_path = cache_dir / "metadata.json"
        if runtime_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("cache_version") == MODEL_CACHE_VERSION:
                    self.load_warning = metadata.get("warning")
                    return runtime_path
            except (OSError, ValueError):
                pass

        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            visual_count, dae_count, mesh_part_count = prepare_urdf_visual_meshes(
                root, path, cache_dir, self.package_map
            )
        except (OSError, RuntimeError, ET.ParseError) as exc:
            raise RuntimeError(f"Failed to prepare visual meshes for {path}: {exc}") from exc
        # MuJoCo's URDF importer discards visual geoms by default. Keep the
        # resolved/converted meshes alongside collision geoms in the runtime
        # MJCF; the renderer selects the resulting visual group.
        mujoco_extension = root.find("mujoco")
        if mujoco_extension is None:
            mujoco_extension = ET.SubElement(root, "mujoco")
        compiler_extension = mujoco_extension.find("compiler")
        if compiler_extension is None:
            compiler_extension = ET.SubElement(mujoco_extension, "compiler")
        compiler_extension.set("discardvisual", "false")
        # MuJoCo fixes a plain URDF root to the world. Add an explicit floating
        # parent so root editing has the same well-defined semantics as MJCF.
        root_link = self._urdf_root_link(root)
        root_link = self._collapse_virtual_world_root(root, root_link)
        if root_link:
            world_link = ET.Element("link", {"name": "ghostgui_world"})
            floating = ET.Element("joint", {
                "name": "floating_base", "type": "floating"
            })
            ET.SubElement(floating, "parent", {"link": "ghostgui_world"})
            ET.SubElement(floating, "child", {"link": root_link})
            root.insert(0, floating)
            root.insert(0, world_link)
        sanitized_path = cache_dir / "source_collision.urdf"
        ET.ElementTree(root).write(sanitized_path, encoding="unicode")
        runtime_path = self._build_urdf_runtime_mjcf(
            sanitized_path, cache_dir, runtime_path
        )
        if dae_count:
            self.load_warning = (
                f"{path.name}: converted {dae_count} COLLADA visual references "
                f"into {mesh_part_count} unique cached OBJ material parts."
            )
        elif visual_count:
            self.load_warning = f"{path.name}: loaded {visual_count} visual meshes."
        else:
            self.load_warning = (
                f"{path.name}: no visual meshes found; using collision geometry."
            )
        metadata_path.write_text(json.dumps({
            "cache_version": MODEL_CACHE_VERSION,
            "source": str(path),
            "mujoco_version": getattr(mujoco, "__version__", "unknown"),
            "warning": self.load_warning,
        }, indent=2), encoding="utf-8")
        return runtime_path

    def _urdf_root_link(self, root):
        child_links = {
            child.attrib.get("link")
            for joint in root.findall("joint")
            for child in joint.findall("child")
        }
        return next(
            (link.attrib.get("name") for link in root.findall("link")
             if link.attrib.get("name") not in child_links),
            None,
        )

    def _collapse_virtual_world_root(self, root, root_link):
        if root_link not in {"world", "map", "odom"}:
            return root_link
        root_link_element = root.find(f"./link[@name='{root_link}']")
        if root_link_element is None:
            return root_link
        physical_tags = {"inertial", "visual", "collision"}
        if any(child.tag in physical_tags for child in root_link_element):
            return root_link

        fixed_children = []
        for joint in root.findall("joint"):
            if joint.attrib.get("type") != "fixed":
                continue
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            if parent.attrib.get("link") == root_link and child.attrib.get("link"):
                fixed_children.append((joint, child.attrib["link"]))
        if len(fixed_children) != 1:
            return root_link

        fixed_joint, child_link = fixed_children[0]
        root.remove(root_link_element)
        root.remove(fixed_joint)
        return child_link

    def _urdf_cache_key(self, path, root):
        digest = hashlib.sha256()
        digest.update(f"ghostgui-model-cache-v{MODEL_CACHE_VERSION}".encode())
        digest.update(getattr(mujoco, "__version__", "unknown").encode())
        digest.update(path.read_bytes())
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename", "")
            digest.update(filename.encode())
            resolved = resolve_mesh_path(filename, path.parent, self.package_map)
            if resolved.path is not None:
                digest.update(resolved.path.read_bytes())
            elif resolved.error:
                digest.update(resolved.error.encode())
        return digest.hexdigest()[:24]

    def _build_urdf_runtime_mjcf(self, urdf_path, cache_dir, mjcf_path=None):
        """Compile sanitized URDF and add editor sites plus a lit scene."""
        compiled = mujoco.MjModel.from_xml_path(str(urdf_path))
        mjcf_path = mjcf_path or cache_dir / f"{urdf_path.stem}_runtime.xml"
        mujoco.mj_saveLastXML(str(mjcf_path), compiled)
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")

        asset = ET.Element("asset")
        ET.SubElement(asset, "texture", {
            "name": "ghostgui_ground", "type": "2d", "builtin": "checker",
            "rgb1": "0.18 0.20 0.24", "rgb2": "0.32 0.35 0.40",
            "width": "512", "height": "512",
        })
        ET.SubElement(asset, "material", {
            "name": "ghostgui_ground", "texture": "ghostgui_ground",
            "texrepeat": "8 8", "reflectance": "0.15",
        })
        root.insert(1, asset)

        visual = ET.Element("visual")
        ET.SubElement(visual, "headlight", {
            "ambient": "0.45 0.45 0.45", "diffuse": "0.8 0.8 0.8",
            "specular": "0.25 0.25 0.25",
        })
        ET.SubElement(visual, "rgba", {
            "haze": "0.12 0.15 0.20 1"
        })
        ET.SubElement(visual, "global", {
            "azimuth": "135", "elevation": "-20"
        })
        root.insert(1, visual)

        ET.SubElement(worldbody, "light", {
            "name": "key_light", "directional": "true",
            "pos": "0 -2 3", "dir": "0 0.5 -1", "diffuse": "0.8 0.8 0.8",
        })
        ET.SubElement(worldbody, "geom", {
            "name": "ground", "type": "plane", "size": "0 0 0.05",
            "material": "ghostgui_ground", "rgba": "0.8 0.8 0.8 1",
        })

        body_by_name = {
            body.attrib.get("name"): body for body in worldbody.iter("body")
        }
        for leg in ("FL", "FR", "RL", "RR"):
            calf = body_by_name.get(f"{leg}_calf")
            if calf is not None:
                ET.SubElement(calf, "site", {
                    "name": f"{leg}_foot", "pos": "-0.002 0 -0.213",
                    "size": "0.012", "rgba": "0.95 0.35 0.08 1",
                })

        for body_name, body in body_by_name.items():
            if body_name == "base":
                color = "0.72 0.76 0.84 1"
            elif body_name and "thigh" in body_name:
                color = "0.58 0.63 0.72 1"
            elif body_name and "calf" in body_name:
                color = "0.16 0.19 0.24 1"
            else:
                color = "0.28 0.32 0.38 1"
            for geom in body.findall("geom"):
                # URDF visual meshes are group 1 and already carry the DAE
                # material color produced by prepare_urdf_visual_meshes.
                # Only recolor collision/fallback primitives.
                if geom.get("group") != "1":
                    geom.set("rgba", color)
            if body_name and "calf" in body_name:
                geoms = [
                    geom for geom in body.findall("geom")
                    if geom.get("group") != "1"
                ]
                if geoms:
                    geoms[-1].set("rgba", "0.05 0.06 0.08 1")

        data = mujoco.MjData(compiled)
        for joint_id in range(compiled.njnt):
            joint_type = int(compiled.jnt_type[joint_id])
            address = int(compiled.jnt_qposadr[joint_id])
            name = mujoco.mj_id2name(
                compiled, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                data.qpos[address + 2] = 0.45
            elif name in self.info.home_joints:
                data.qpos[address] = self.info.home_joints[name]
        keyframe = ET.SubElement(root, "keyframe")
        ET.SubElement(keyframe, "key", {
            "name": "home",
            "qpos": " ".join(f"{float(value):.9g}" for value in data.qpos),
        })
        tree.write(mjcf_path, encoding="unicode")
        # Compile once here so conversion errors identify the generated model.
        mujoco.MjModel.from_xml_path(str(mjcf_path))
        return mjcf_path

    def _apply_registered_home(self):
        if self.model_type == "quadruped":
            for free_joint in self.free_joints_by_body.values():
                self.home_qpos[free_joint.qpos_address + 2] = 0.45
        for name, value in self.info.home_joints.items():
            joint = self.joints.get(name)
            if joint is not None:
                self.home_qpos[joint.qpos_address] = value

    def _ground_home_qpos(self, clearance=HOME_GROUND_CLEARANCE):
        if not self.free_joints_by_body:
            return
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = self.home_qpos
        mujoco.mj_forward(self.mj_model, data)
        lowest_z = self._lowest_robot_geom_z(data)
        if lowest_z is None:
            return
        delta = float(clearance) - lowest_z
        if abs(delta) < 1e-9:
            return
        for free_joint in self.free_joints_by_body.values():
            self.home_qpos[free_joint.qpos_address + 2] += delta

    def _lowest_robot_geom_z(self, data):
        lowest = None
        for geom_id in range(self.mj_model.ngeom):
            if self._is_non_robot_ground_geom(geom_id):
                continue
            geom_z = self._geom_lowest_z(data, geom_id)
            if geom_z is None:
                continue
            lowest = geom_z if lowest is None else min(lowest, geom_z)
        return lowest

    def _is_non_robot_ground_geom(self, geom_id):
        geom_type = int(self.mj_model.geom_type[geom_id])
        if geom_type in {
            int(mujoco.mjtGeom.mjGEOM_PLANE),
            int(mujoco.mjtGeom.mjGEOM_HFIELD),
        }:
            return True
        if int(self.mj_model.geom_bodyid[geom_id]) == 0:
            return True
        name = mujoco.mj_id2name(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name == "ground"

    def _geom_lowest_z(self, data, geom_id):
        geom_type = int(self.mj_model.geom_type[geom_id])
        center_z = float(data.geom_xpos[geom_id][2])
        z_axis = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)[2]
        size = np.asarray(self.mj_model.geom_size[geom_id], dtype=float)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return center_z - float(size[0])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            extent = float(size[0]) + abs(float(z_axis[2])) * float(size[1])
            return center_z - extent
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radial = float(size[0]) * float(np.linalg.norm(z_axis[:2]))
            axial = float(size[1]) * abs(float(z_axis[2]))
            return center_z - radial - axial
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return center_z - float(np.dot(np.abs(z_axis), size[:3]))
        if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return center_z - float(np.linalg.norm(z_axis * size[:3]))
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(self.mj_model.geom_dataid[geom_id])
            if mesh_id < 0:
                return center_z - float(self.mj_model.geom_rbound[geom_id])
            start = int(self.mj_model.mesh_vertadr[mesh_id])
            count = int(self.mj_model.mesh_vertnum[mesh_id])
            vertices = self.mj_model.mesh_vert[start:start + count]
            if len(vertices) == 0:
                return None
            return center_z + float(np.min(vertices @ z_axis))
        return center_z - float(self.mj_model.geom_rbound[geom_id])

    def _name_variants(self, name):
        plain = self.plain_name(name)
        return {name, plain, name.lower(), plain.lower()}

    def _find_name(self, candidates, names):
        names = list(names)
        for candidate in candidates:
            for name in names:
                if self._name_variants(candidate) & self._name_variants(name):
                    return name
        return None

    def _first_body(self, candidates):
        return self._find_name(candidates, self.body_names)

    def _first_joint(self, candidates):
        return self._find_name(candidates, self.joints)

    def _build_logical_frames(self):
        bindings = {}
        for logical_name, candidates in self.info.logical_frames.items():
            site = self._find_name(candidates, self.site_names)
            body = self._find_name(candidates, self.body_names)
            if site is not None:
                bindings[logical_name] = ("site", site)
            elif body is not None:
                bindings[logical_name] = ("body", body)
        return bindings or self._infer_logical_frames()

    def _unique_logical_name(self, name, existing):
        base = self.plain_name(name).replace(" ", "_") or "frame"
        candidate = base
        index = 2
        while candidate in existing:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def _infer_logical_frames(self):
        bindings = {}
        ignored = tuple(token.lower() for token in self.info.ignored_body_tokens)

        def include(name):
            return name and name != "world" and not any(
                token in name.lower() for token in ignored
            )

        if include(self.root_body):
            logical = self._unique_logical_name(self.root_body, bindings)
            bindings[logical] = ("body", self.root_body)

        for site_name in self.site_names:
            if include(site_name):
                logical = self._unique_logical_name(site_name, bindings)
                bindings[logical] = ("site", site_name)

        if len(bindings) > (1 if self.root_body else 0):
            return bindings

        parent_ids = set(int(parent) for parent in self.mj_model.body_parentid)
        for body_id in range(1, self.mj_model.nbody):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if body_id not in parent_ids and include(name):
                logical = self._unique_logical_name(name, bindings)
                bindings[logical] = ("body", name)
        return bindings

    def resolve_logical_frame(self, name):
        binding = self.logical_frame_bindings.get(name)
        if binding is not None:
            return binding
        site = self._find_name((name,), self.site_names)
        if site is not None:
            return "site", site
        body = self._find_name((name,), self.body_names)
        if body is not None:
            return "body", body
        return None

    def logical_frame_for_body(self, body_name):
        """Map a picked MuJoCo body to the nearest editable logical frame."""
        body_id = self._body_id(body_name)
        if body_id is None:
            return None
        owner_by_logical = {}
        for logical, (kind, object_name) in self.logical_frame_bindings.items():
            if kind == "body":
                owner = object_name
            else:
                site_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SITE, object_name
                )
                if site_id < 0:
                    continue
                owner_id = int(self.mj_model.site_bodyid[site_id])
                owner = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, owner_id
                )
            owner_by_logical[logical] = owner
            if owner == body_name:
                return logical

        plain = self.plain_name(body_name).lower()
        semantic = None
        if any(token in plain for token in ("hand", "palm", "wrist", "elbow", "shoulder", "arm")):
            semantic = "hand"
        elif any(token in plain for token in ("foot", "ankle", "calf", "knee", "thigh", "hip", "leg")):
            semantic = "foot"
        elif any(token in plain for token in ("pelvis", "base", "root")):
            semantic = "base"
        elif any(token in plain for token in ("torso", "trunk", "waist")):
            semantic = "torso"

        side = None
        for token in ("left", "right", "fl", "fr", "rl", "rr"):
            if plain.startswith(token + "_") or f"/{token}_" in body_name.lower():
                side = token
                break

        def ancestors(name):
            result = {}
            distance = 0
            while name is not None and name not in result:
                result[name] = distance
                name = self.get_parent_body(name)
                distance += 1
            return result

        picked_ancestors = ancestors(body_name)
        scored = []
        for logical, owner in owner_by_logical.items():
            owner_ancestors = ancestors(owner)
            common = set(picked_ancestors) & set(owner_ancestors)
            if not common:
                continue
            distance = min(
                picked_ancestors[name] + owner_ancestors[name] for name in common
            )
            lower = logical.lower()
            penalty = 0
            if semantic == "hand" and "hand" not in lower:
                penalty += 100
            elif semantic == "foot" and "foot" not in lower:
                penalty += 100
            elif semantic == "base" and not any(x in lower for x in ("pelvis", "base")):
                penalty += 100
            elif semantic == "torso" and not any(x in lower for x in ("torso", "trunk")):
                penalty += 100
            if side and not lower.startswith(side):
                aliases = {"left": "l", "right": "r"}
                if not lower.startswith(aliases.get(side, side)):
                    penalty += 50
            scored.append((penalty + distance, logical))
        return min(scored)[1] if scored else None

    def get_body_names(self):
        return list(self.body_names)

    def get_site_names(self):
        return list(self.site_names)

    def get_body_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"Unknown MuJoCo body: {body_name}")
        return self.mj_data.xpos[body_id].copy(), self.mj_data.xquat[body_id].copy()

    def get_site_pose(self, site_name):
        site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise KeyError(f"Unknown MuJoCo site: {site_name}")
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self.mj_data.site_xmat[site_id])
        return self.mj_data.site_xpos[site_id].copy(), quaternion

    def get_qpos(self):
        return self.mj_data.qpos.copy()

    def set_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.mj_model.nq,):
            raise ValueError(f"Expected qpos shape ({self.mj_model.nq},), got {qpos.shape}")
        self.mj_data.qpos[:] = qpos
        self.forward_kinematics()

    def forward_kinematics(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _body_id(self, body_name):
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return body_id if body_id >= 0 else None

    def get_parent_body(self, body_name):
        body_id = self._body_id(body_name)
        if body_id in (None, 0):
            return None
        parent_id = int(self.mj_model.body_parentid[body_id])
        return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

    def get_child_bodies(self, body_name):
        body_id = self._body_id(body_name)
        if body_id is None:
            return []
        children = []
        for child_id in range(1, self.mj_model.nbody):
            if int(self.mj_model.body_parentid[child_id]) == body_id:
                name = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, child_id
                )
                if name:
                    children.append(name)
        return children

    def get_kinematic_edges(self, important_only=False):
        edges = []
        ignored = tuple(token.lower() for token in self.info.ignored_body_tokens)
        for body_id in range(1, self.mj_model.nbody):
            child = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            parent_id = int(self.mj_model.body_parentid[body_id])
            parent = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
            if not child or not parent:
                continue
            if important_only and any(token in child.lower() for token in ignored):
                continue
            edges.append((parent, child))
        return edges

    def _build_kinematic_tree(self):
        tree = {name: [] for name in self.body_names}
        for parent, child in self.get_kinematic_edges():
            tree.setdefault(parent, []).append(child)
        return tree

    def skeleton_positions(self):
        positions = {}
        for body_id in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name:
                positions[name] = self.mj_data.xpos[body_id].copy()
        return positions
