"""
Convert stecker captures (capture 10 etc.) + yellow-connector mesh to MegaPose example format.

Data layout expected:
  --captures-root/
    {capture_id}/
      rgb/{idx:05d}.png
      depth/{idx:05d}.png
      camera.json            # {"camera_intrinsics": [[fx,0,cx],...], "depth_scale": <m/unit>}
      sam3_annotations/
        masks/
          obj_{n}__class_{c}__{class_name}/
            frame_{idx:06d}.png   # binary uint8 mask (0 / 255)

  --mesh-dir/
    mesh_textured.obj   (+ .mtl + textures)  -- assumed in METERS

Output (LOCAL_DATA_DIR/examples/{label}/):
  meshes/{label}/textured_mesh.obj  (scaled to mm, shared)
  {idx:06d}/
    image_rgb.png
    image_depth.png       (uint16, mm)
    camera_data.json
    inputs/object_data.json
    meshes/{label} -> ../../meshes/{label}  (symlink)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import trimesh

from megapose.config import LOCAL_DATA_DIR
from megapose.scripts.convert_bundlesdf_to_examples import (
    copy_mtl_and_textures,
    extract_bbox_modal_from_mask,
    parse_mtl_references,
    sanitize_label,
    scale_obj_vertices,
)
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

CAPTURES_ROOT_DEFAULT = Path("/vol/coro/dtrofimov/data/projects/captures")
MESH_DIR_DEFAULT = Path("/vol/coro/dtrofimov/data/projects/yellow_connector_manual/plug")
MIN_MESH_VERTICES_DEFAULT = 2000
MATERIAL_COLOR_TEXTURE_PREFIX = "material_color"


def clamp_color_channel(value: float) -> int:
    return int(round(max(0.0, min(1.0, value)) * 255.0))


def parse_mtl_texture_references(mtl_path: Path) -> Set[Path]:
    refs: Set[Path] = set()
    for raw_line in mtl_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        is_map_key = key.startswith("map_") or key in {"bump", "disp", "decal", "refl"}
        if is_map_key:
            refs.add(Path(parts[-1]))
    return refs


def parse_mtl_diffuse_colors(mtl_path: Path) -> Dict[str, Tuple[int, int, int]]:
    colors: Dict[str, Tuple[int, int, int]] = {}
    current_material: Optional[str] = None

    for raw_line in mtl_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0].lower() == "newmtl" and len(parts) >= 2:
            current_material = " ".join(parts[1:])
        elif parts[0].lower() == "kd" and current_material is not None and len(parts) >= 4:
            kd = tuple(clamp_color_channel(float(x)) for x in parts[1:4])
            colors[current_material] = kd

    return colors


def collect_obj_mtllibs(mesh_obj: Path) -> List[Path]:
    mtllibs: List[Path] = []
    for raw_line in mesh_obj.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("mtllib "):
            mtllibs.extend(Path(part) for part in line.split()[1:])
    return mtllibs


def collect_obj_materials(mesh_obj: Path) -> List[str]:
    materials: List[str] = []
    seen: Set[str] = set()
    for raw_line in mesh_obj.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("usemtl "):
            material = line.split(maxsplit=1)[1]
            if material not in seen:
                materials.append(material)
                seen.add(material)
    return materials


def find_mask_dir(capture_dir: Path, class_id: int) -> Path:
    """Find the per-object mask subdirectory for a given class id."""
    masks_root = capture_dir / "sam3_annotations" / "masks"
    if not masks_root.exists():
        raise FileNotFoundError(f"No sam3_annotations/masks in {capture_dir}")
    for d in sorted(masks_root.iterdir()):
        if d.is_dir() and f"__class_{class_id}__" in d.name:
            return d
    raise FileNotFoundError(
        f"No mask dir with class_id={class_id} found in {masks_root}"
    )


def load_camera(camera_json_path: Path):
    """Return (K_3x3_list, depth_scale_m_per_unit) from the capture camera.json."""
    data = json.loads(camera_json_path.read_text())
    K = data["camera_intrinsics"]  # [[fx,0,cx],[0,fy,cy],[0,0,1]]
    depth_scale = float(data.get("depth_scale", 1e-4))
    return K, depth_scale


def convert_depth_to_mm_uint16(depth_raw: np.ndarray, depth_scale_m_per_unit: float) -> np.ndarray:
    scale_to_mm = depth_scale_m_per_unit * 1000.0
    depth_mm = np.round(depth_raw.astype(np.float64) * scale_to_mm)
    return np.clip(depth_mm, 0, 65535).astype(np.uint16)


def count_obj_vertices(mesh_obj: Path) -> int:
    return sum(1 for line in mesh_obj.read_text().splitlines() if line.startswith("v "))


def loaded_mesh_vertex_count(mesh_obj: Path) -> int:
    scene_or_mesh = trimesh.load(
        mesh_obj,
        group_material=False,
        process=False,
        skip_materials=True,
        maintain_order=True,
    )
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise RuntimeError(f"Loaded empty mesh scene from {mesh_obj}")
        mesh = trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()
            )
        )
    else:
        mesh = scene_or_mesh
    return int(len(mesh.vertices))


def parse_obj_vertex_index(token: str, n_vertices: int) -> int:
    raw = token.split("/")[0]
    idx = int(raw)
    if idx < 0:
        return n_vertices + idx
    return idx - 1


def triangulate_face(face: List[int]) -> List[Tuple[int, int, int]]:
    if len(face) < 3:
        return []
    return [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]


def subdivide_obj_once(mesh_obj: Path) -> int:
    """Midpoint-subdivide OBJ faces while preserving mtllib/usemtl records."""
    vertices: List[np.ndarray] = []
    header_lines: List[str] = []
    records = []
    midpoint_cache = {}

    for raw_line in mesh_obj.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            vertices.append(np.array([float(x) for x in line.split()[1:4]], dtype=np.float64))
        elif line.startswith("f "):
            face = [parse_obj_vertex_index(tok, len(vertices)) for tok in line.split()[1:]]
            records.append(("face", face))
        elif line.startswith(("vn ", "vt ")):
            continue
        elif line.startswith(("usemtl ", "g ", "o ", "s ")):
            records.append(("line", raw_line))
        else:
            header_lines.append(raw_line)

    def midpoint(i: int, j: int) -> int:
        key = tuple(sorted((i, j)))
        if key not in midpoint_cache:
            vertices.append((vertices[i] + vertices[j]) * 0.5)
            midpoint_cache[key] = len(vertices) - 1
        return midpoint_cache[key]

    out_records = []
    for kind, value in records:
        if kind == "line":
            out_records.append(("line", value))
            continue
        for a, b, c in triangulate_face(value):
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            out_records.extend(
                [
                    ("face", [a, ab, ca]),
                    ("face", [ab, b, bc]),
                    ("face", [ca, bc, c]),
                    ("face", [ab, bc, ca]),
                ]
            )

    out_lines = []
    out_lines.extend(header_lines)
    out_lines.extend(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}" for v in vertices)
    for kind, value in out_records:
        if kind == "line":
            out_lines.append(value)
        else:
            out_lines.append("f " + " ".join(str(i + 1) for i in value))
    mesh_obj.write_text("\n".join(out_lines) + "\n")
    return len(vertices)


def ensure_min_mesh_vertices(mesh_obj: Path, min_vertices: int) -> int:
    if min_vertices <= 0:
        loaded = loaded_mesh_vertex_count(mesh_obj)
        logger.info("Mesh %s loads with %d vertices.", mesh_obj, loaded)
        return loaded

    loaded = loaded_mesh_vertex_count(mesh_obj)
    logger.info("Mesh %s loads with %d vertices.", mesh_obj, loaded)
    while loaded < min_vertices:
        logger.info(
            "Mesh has %d vertices, below requested minimum %d; subdividing.",
            loaded,
            min_vertices,
        )
        subdivide_obj_once(mesh_obj)
        loaded = loaded_mesh_vertex_count(mesh_obj)
        logger.info("Subdivided mesh now loads with %d vertices.", loaded)
    return loaded


def format_obj_face_token(token: str, vt_index: int) -> str:
    parts = token.split("/")
    vertex_index = parts[0]
    normal_index = parts[2] if len(parts) >= 3 and parts[2] else None
    if normal_index is not None:
        return f"{vertex_index}/{vt_index}/{normal_index}"
    return f"{vertex_index}/{vt_index}"


def write_material_color_textures(
    mesh_dir: Path,
    material_names: List[str],
    material_colors: Dict[str, Tuple[int, int, int]],
    texture_size: int = 64,
) -> Tuple[Dict[str, List[int]], List[str], Dict[str, str]]:
    material_to_vts: Dict[str, List[int]] = {}
    material_to_texture: Dict[str, str] = {}
    vt_lines: List[str] = []

    for i, material in enumerate(material_names):
        color = material_colors.get(material, (255, 255, 255))
        texture_name = f"{MATERIAL_COLOR_TEXTURE_PREFIX}_{i:02d}.png"
        texture = np.full((texture_size, texture_size, 3), color, dtype=np.uint8)
        Image.fromarray(texture).save(mesh_dir / texture_name)
        material_to_texture[material] = texture_name

        # Each material uses its own solid-color texture, so all UVs can be central.
        u0 = 0.25
        u1 = 0.75
        v0 = 0.25
        v1 = 0.75
        vt_start = len(vt_lines) + 1
        vt_lines.extend(
            [
                f"vt {u0:.9f} {v0:.9f}",
                f"vt {u1:.9f} {v0:.9f}",
                f"vt {u0:.9f} {v1:.9f}",
            ]
        )
        material_to_vts[material] = [vt_start, vt_start + 1, vt_start + 2]

    return material_to_vts, vt_lines, material_to_texture


def add_texture_reference_to_mtl(mtl_path: Path, material_to_texture: Dict[str, str]) -> None:
    out_lines: List[str] = []
    current_material: Optional[str] = None
    material_has_map = False

    def maybe_add_map() -> None:
        if (
            current_material is not None
            and not material_has_map
            and current_material in material_to_texture
        ):
            out_lines.append(f"map_Kd {material_to_texture[current_material]}")

    for raw_line in mtl_path.read_text().splitlines():
        line = raw_line.strip()
        parts = line.split()
        if parts and parts[0].lower() == "newmtl":
            maybe_add_map()
            current_material = " ".join(parts[1:])
            material_has_map = False
        elif parts:
            key = parts[0].lower()
            if key.startswith("map_") or key in {"bump", "disp", "decal", "refl"}:
                material_has_map = True
        out_lines.append(raw_line)

    maybe_add_map()
    mtl_path.write_text("\n".join(out_lines) + "\n")


def assign_material_color_texture(mesh_obj: Path) -> bool:
    mtllibs = collect_obj_mtllibs(mesh_obj)
    if not mtllibs:
        logger.info("No mtllib references found in %s; skipping material color texture.", mesh_obj)
        return False

    mtl_paths = [mesh_obj.parent / mtl for mtl in mtllibs]
    if any(parse_mtl_texture_references(mtl_path) for mtl_path in mtl_paths if mtl_path.exists()):
        logger.info("Mesh %s already has texture references; leaving materials unchanged.", mesh_obj)
        return False

    material_colors: Dict[str, Tuple[int, int, int]] = {}
    for mtl_path in mtl_paths:
        if not mtl_path.exists():
            raise FileNotFoundError(f"OBJ references missing MTL: {mtl_path}")
        material_colors.update(parse_mtl_diffuse_colors(mtl_path))

    material_names = [
        material for material in collect_obj_materials(mesh_obj) if material in material_colors
    ]
    if not material_names:
        logger.info("No diffuse material colors found for %s; skipping texture atlas.", mesh_obj)
        return False

    material_to_vts, vt_lines, material_to_texture = write_material_color_textures(
        mesh_obj.parent,
        material_names,
        material_colors,
    )

    lines = mesh_obj.read_text().splitlines()
    lines_without_vt = [line for line in lines if not line.strip().startswith("vt ")]
    insert_after = 0
    for n, raw_line in enumerate(lines_without_vt):
        stripped = raw_line.strip()
        if stripped.startswith(("v ", "vn ")):
            insert_after = n + 1

    out_lines: List[str] = []
    current_material: Optional[str] = None
    for raw_line in lines_without_vt:
        line = raw_line.strip()
        if line.startswith("usemtl "):
            current_material = line.split(maxsplit=1)[1]
            out_lines.append(raw_line)
        elif line.startswith("f ") and current_material in material_to_vts:
            tokens = line.split()[1:]
            vt_indices = material_to_vts[current_material]
            face_tokens = [
                format_obj_face_token(token, vt_indices[i % len(vt_indices)])
                for i, token in enumerate(tokens)
            ]
            out_lines.append("f " + " ".join(face_tokens))
        else:
            out_lines.append(raw_line)

    out_lines[insert_after:insert_after] = vt_lines
    mesh_obj.write_text("\n".join(out_lines) + "\n")

    for mtl_path in mtl_paths:
        add_texture_reference_to_mtl(mtl_path, material_to_texture)

    logger.info(
        "Created material color textures in %s for %d material(s).",
        mesh_obj.parent,
        len(material_names),
    )
    return True


def build_shared_mesh(
    mesh_dir: Path,
    shared_mesh_dir: Path,
    mesh_scale: float,
    min_mesh_vertices: int,
    material_color_textures: bool,
    overwrite: bool,
) -> None:
    if shared_mesh_dir.exists() and not overwrite:
        logger.info("Reusing existing shared mesh dir: %s", shared_mesh_dir)
        return
    if shared_mesh_dir.exists():
        shutil.rmtree(shared_mesh_dir)

    mesh_obj_src = mesh_dir / "mesh_textured.obj"
    if not mesh_obj_src.exists():
        candidates = list(mesh_dir.glob("*.obj"))
        if not candidates:
            raise FileNotFoundError(f"No .obj mesh found in {mesh_dir}")
        mesh_obj_src = candidates[0]

    mesh_obj_dst = shared_mesh_dir / mesh_obj_src.name
    mtllibs = scale_obj_vertices(mesh_obj_src, mesh_obj_dst, mesh_scale=mesh_scale)
    copy_mtl_and_textures(mesh_obj_src.parent, shared_mesh_dir, mtllibs)
    vertex_count = ensure_min_mesh_vertices(mesh_obj_dst, min_mesh_vertices)
    if material_color_textures:
        assign_material_color_texture(mesh_obj_dst)
    else:
        logger.info("Material color texture generation disabled for %s.", mesh_obj_dst)
    logger.info(
        "Shared mesh written to %s (scale=%.1f, vertices=%d)",
        shared_mesh_dir,
        mesh_scale,
        vertex_count,
    )


def symlink_mesh(shared_mesh_dir: Path, frame_dir: Path, label: str, overwrite: bool) -> None:
    link = frame_dir / "meshes" / label
    if link.exists() or link.is_symlink():
        if overwrite:
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        else:
            return
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(shared_mesh_dir, start=link.parent)
    link.symlink_to(rel, target_is_directory=True)


def convert_frames(
    capture_dir: Path,
    mask_dir: Path,
    frame_indices: Optional[List[int]],
    K: list,
    depth_scale: float,
    example_root: Path,
    label: str,
    overwrite: bool,
) -> int:
    # Resolve frame indices from available mask files
    all_mask_stems = sorted(p.stem for p in mask_dir.glob("frame_*.png"))
    # mask stem: frame_000000 → index 0
    all_indices = [int(s.split("_")[1]) for s in all_mask_stems]

    if frame_indices is not None:
        missing = [i for i in frame_indices if i not in all_indices]
        if missing:
            raise RuntimeError(f"Requested frame indices not found in masks: {missing}")
        indices = frame_indices
    else:
        indices = all_indices

    rgb_dir = capture_dir / "rgb"
    depth_dir = capture_dir / "depth"

    converted = 0
    for idx in indices:
        rgb_path = rgb_dir / f"{idx:05d}.png"
        depth_path = depth_dir / f"{idx:05d}.png"
        mask_path = mask_dir / f"frame_{idx:06d}.png"

        for p in (rgb_path, depth_path, mask_path):
            if not p.exists():
                logger.warning("Skipping frame %d: %s not found", idx, p)
                break
        else:
            pass

        if not rgb_path.exists() or not depth_path.exists() or not mask_path.exists():
            continue

        frame_dir = example_root / f"{idx:06d}"
        if frame_dir.exists() and not overwrite:
            logger.debug("Skipping existing frame dir: %s", frame_dir)
            converted += 1
            continue
        if frame_dir.exists() and overwrite:
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)

        # RGB
        shutil.copy2(rgb_path, frame_dir / "image_rgb.png")

        # Depth → uint16 mm
        depth_raw = np.array(Image.open(depth_path))
        H, W = depth_raw.shape[:2]
        depth_mm = convert_depth_to_mm_uint16(depth_raw, depth_scale)
        Image.fromarray(depth_mm).save(frame_dir / "image_depth.png")

        # Camera data
        cam_data = {"K": K, "resolution": [H, W]}
        (frame_dir / "camera_data.json").write_text(json.dumps(cam_data))

        # Bbox from mask
        mask = np.array(Image.open(mask_path))
        bbox = extract_bbox_modal_from_mask(mask)
        (frame_dir / "inputs").mkdir(exist_ok=True)
        (frame_dir / "inputs" / "object_data.json").write_text(
            json.dumps([{"label": label, "bbox_modal": bbox}])
        )

        # Mesh symlink
        symlink_mesh(example_root / "meshes" / label, frame_dir, label, overwrite)

        converted += 1
        logger.debug("Frame %06d → %s  bbox=%s", idx, frame_dir, bbox)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert stecker captures to MegaPose example format."
    )
    parser.add_argument("--capture-id", type=str, default="10")
    parser.add_argument(
        "--captures-root", type=Path, default=CAPTURES_ROOT_DEFAULT
    )
    parser.add_argument("--mesh-dir", type=Path, default=MESH_DIR_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_DATA_DIR / "examples" / "yellow-connector",
    )
    parser.add_argument("--label", type=str, default="yellow-connector")
    parser.add_argument(
        "--class-id",
        type=int,
        default=1,
        help="SAM3 class id for the target object (1 = plug_yellow_body)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Frame indices to convert. Default: all frames that have a mask.",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1000.0,
        help="Scale applied to mesh vertices (default 1000 converts meters to mm).",
    )
    parser.add_argument(
        "--min-mesh-vertices",
        type=int,
        default=MIN_MESH_VERTICES_DEFAULT,
        help=(
            "Minimum loaded mesh vertex count required by MegaPose. "
            "If lower, the converted OBJ is midpoint-subdivided until it reaches this count."
        ),
    )
    parser.add_argument(
        "--no-material-color-textures",
        action="store_true",
        help=(
            "Do not synthesize solid texture PNGs from MTL diffuse colors. "
            "Use this for textureless comparison exports."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_logging_level("debug" if args.debug else "info")

    capture_dir = args.captures_root / args.capture_id
    if not capture_dir.exists():
        raise FileNotFoundError(f"Capture dir not found: {capture_dir}")

    label = sanitize_label(args.label)
    example_root = args.output_dir.resolve()
    example_root.mkdir(parents=True, exist_ok=True)

    K, depth_scale = load_camera(capture_dir / "camera.json")
    logger.info("Camera K loaded. depth_scale=%.2e m/unit", depth_scale)

    mask_dir = find_mask_dir(capture_dir, args.class_id)
    logger.info("Mask dir: %s", mask_dir)

    shared_mesh_dir = example_root / "meshes" / label
    build_shared_mesh(
        args.mesh_dir,
        shared_mesh_dir,
        args.mesh_scale,
        args.min_mesh_vertices,
        not args.no_material_color_textures,
        args.overwrite,
    )

    n = convert_frames(
        capture_dir=capture_dir,
        mask_dir=mask_dir,
        frame_indices=args.frames,
        K=K,
        depth_scale=depth_scale,
        example_root=example_root,
        label=label,
        overwrite=args.overwrite,
    )

    logger.info("Done. Converted %d frame(s) to %s", n, example_root)

    examples_root = (LOCAL_DATA_DIR / "examples").resolve()
    if example_root.is_relative_to(examples_root):
        first_name = str((example_root / "000000").relative_to(examples_root))
        logger.info(
            "Run inference: python -m megapose.scripts.run_inference_on_example %s --run-inference",
            first_name,
        )


if __name__ == "__main__":
    main()
