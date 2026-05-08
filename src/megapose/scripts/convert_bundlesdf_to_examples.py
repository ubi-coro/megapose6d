"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Third Party
import numpy as np
from PIL import Image

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def parse_cam_K_txt(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != 3:
            raise ValueError(f"Expected 3 values per row in {path}, got {vals}")
        rows.append(vals)
    K = np.array(rows, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K in {path}, got shape={K.shape}")
    return K


def sanitize_label(label: str) -> str:
    label = label.strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in label:
        label = label.replace("--", "-")
    return label


def list_frame_stems(rgb_dir: Path, depth_dir: Path, masks_dir: Path) -> List[str]:
    rgb_stems = {p.stem for p in rgb_dir.glob("*.png")}
    depth_stems = {p.stem for p in depth_dir.glob("*.png")}
    mask_stems = {p.stem for p in masks_dir.glob("*.png")}
    common = sorted(rgb_stems & depth_stems & mask_stems)
    if not common:
        raise RuntimeError("No common frame ids across rgb/depth/masks")

    missing_rgb = sorted((depth_stems & mask_stems) - rgb_stems)
    missing_depth = sorted((rgb_stems & mask_stems) - depth_stems)
    missing_masks = sorted((rgb_stems & depth_stems) - mask_stems)
    if missing_rgb or missing_depth or missing_masks:
        raise RuntimeError(
            "Mismatched frame sets across rgb/depth/masks. "
            f"missing_rgb={missing_rgb[:5]} missing_depth={missing_depth[:5]} "
            f"missing_masks={missing_masks[:5]}"
        )
    return common


def extract_bbox_modal_from_mask(mask: np.ndarray) -> List[int]:
    fg = mask > 0
    ys, xs = np.where(fg)
    if len(xs) == 0:
        raise RuntimeError("Mask is empty (no foreground pixels).")
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1
    return [xmin, ymin, xmax, ymax]


def convert_depth_to_mm(
    depth_raw: np.ndarray,
    depth_mode: str,
    depth_scale_m_per_unit: Optional[float],
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str]:
    if depth_mode == "raw-mm":
        scale_to_mm = 1.0
    elif depth_mode == "metadata":
        if depth_scale_m_per_unit is None:
            raise RuntimeError("depth_scale_m_per_unit is not available in metadata.")
        scale_to_mm = depth_scale_m_per_unit * 1000.0
    elif depth_mode == "auto":
        if depth_scale_m_per_unit is None:
            scale_to_mm = 1.0
            depth_mode = "raw-mm"
        else:
            if mask is not None:
                nonzero = depth_raw[(depth_raw > 0) & (mask > 0)]
            else:
                nonzero = depth_raw[depth_raw > 0]
            if nonzero.size == 0:
                scale_to_mm = depth_scale_m_per_unit * 1000.0
                depth_mode = "metadata"
            else:
                med_raw = float(np.median(nonzero))
                med_meta = med_raw * depth_scale_m_per_unit * 1000.0
                # Heuristic tailored to close-range captures:
                # if metadata makes depths implausibly tiny (<5cm) while raw values are plausible mm.
                if med_meta < 50.0 <= med_raw:
                    scale_to_mm = 1.0
                    depth_mode = "raw-mm"
                else:
                    scale_to_mm = depth_scale_m_per_unit * 1000.0
                    depth_mode = "metadata"
    else:
        raise ValueError(f"Unknown depth_mode={depth_mode}")

    depth_mm = np.round(depth_raw.astype(np.float64) * scale_to_mm)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_mm, depth_mode


def parse_mtl_references(obj_lines: Iterable[str]) -> Set[Path]:
    mtllibs: Set[Path] = set()
    for line in obj_lines:
        line = line.strip()
        if line.startswith("mtllib "):
            for part in line.split()[1:]:
                mtllibs.add(Path(part))
    return mtllibs


def parse_texture_refs_from_mtl(mtl_lines: Iterable[str]) -> Set[Path]:
    refs: Set[Path] = set()
    for raw_line in mtl_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].lower()
        is_map_key = key.startswith("map_") or key in {"bump", "disp", "decal", "refl"}
        if not is_map_key:
            continue
        refs.add(Path(parts[-1]))
    return refs


def copy_mtl_and_textures(
    src_mesh_dir: Path,
    dst_mesh_dir: Path,
    mtllib_paths: Set[Path],
) -> None:
    for mtl_rel in mtllib_paths:
        src_mtl = src_mesh_dir / mtl_rel
        dst_mtl = dst_mesh_dir / mtl_rel
        if not src_mtl.exists():
            raise FileNotFoundError(f"OBJ references missing MTL: {src_mtl}")
        dst_mtl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_mtl, dst_mtl)

        texture_refs = parse_texture_refs_from_mtl(src_mtl.read_text().splitlines())
        for tex_rel in texture_refs:
            src_tex = src_mtl.parent / tex_rel
            if not src_tex.exists():
                logger.warning(
                    "Texture reference not found from %s: %s",
                    src_mtl,
                    tex_rel,
                )
                continue
            dst_tex = dst_mtl.parent / tex_rel
            dst_tex.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_tex, dst_tex)


def scale_obj_vertices(src_obj_path: Path, dst_obj_path: Path, mesh_scale: float) -> Set[Path]:
    obj_lines = src_obj_path.read_text().splitlines()
    mtllibs = parse_mtl_references(obj_lines)
    out_lines: List[str] = []
    for line in obj_lines:
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                out_lines.append(
                    f"v {x * mesh_scale:.9f} {y * mesh_scale:.9f} {z * mesh_scale:.9f}"
                )
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
    dst_obj_path.parent.mkdir(parents=True, exist_ok=True)
    dst_obj_path.write_text("\n".join(out_lines) + "\n")
    return mtllibs


def maybe_symlink_or_copy_mesh(src_mesh_dir: Path, dst_link_dir: Path, overwrite: bool) -> None:
    dst_link_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_link_dir.exists() or dst_link_dir.is_symlink():
        if overwrite:
            if dst_link_dir.is_symlink() or dst_link_dir.is_file():
                dst_link_dir.unlink()
            else:
                shutil.rmtree(dst_link_dir)
        else:
            return

    try:
        rel = os.path.relpath(src_mesh_dir, start=dst_link_dir.parent)
        dst_link_dir.symlink_to(rel, target_is_directory=True)
    except OSError:
        shutil.copytree(src_mesh_dir, dst_link_dir, dirs_exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Convert BundleSDF-style captures into MegaPose example folders."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=LOCAL_DATA_DIR / "power_connector",
        help="Directory containing captures/, mesh/, and optional object_spec.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_DATA_DIR / "examples" / "power-connector",
        help="Output root. Frame folders will be created inside this directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="power-connector",
        help="Object label used in object_data.json and meshes/<label>/",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        nargs="+",
        default=None,
        help="Optional frame indices (e.g. 12 15). By default converts all frames.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("auto", "raw-mm", "metadata"),
        default="raw-mm",
        help=(
            "How to convert input depth to millimeters expected by MegaPose PNG loading. "
            "'auto' uses a sanity check on metadata scale."
        ),
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1000.0,
        help=(
            "Scale applied to mesh vertices when writing converted OBJ. "
            "Default 1000 converts mesh coordinates from meters to millimeters."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already existing frame folders and converted mesh directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_logging_level("debug" if args.debug else "info")

    source_dir: Path = args.source_dir.resolve()
    captures_dir = source_dir / "captures"
    mesh_dir = source_dir / "mesh"
    metadata_path = captures_dir / "metadata.json"
    cam_k_path = captures_dir / "cam_K.txt"
    object_spec_path = source_dir / "object_spec.json"

    rgb_dir = captures_dir / "rgb"
    depth_dir = captures_dir / "depth"
    masks_dir = captures_dir / "masks"

    for path in (rgb_dir, depth_dir, masks_dir, mesh_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    object_spec = read_json(object_spec_path) if object_spec_path.exists() else {}

    mesh_rel = object_spec.get("mesh_path", "./mesh/textured_mesh.obj")
    mesh_rel_path = Path(mesh_rel)
    mesh_obj = mesh_rel_path if mesh_rel_path.is_absolute() else (source_dir / mesh_rel_path)
    if not mesh_obj.exists():
        mesh_obj = mesh_dir / "textured_mesh.obj"
    if not mesh_obj.exists():
        raise FileNotFoundError("Could not find source mesh OBJ.")

    if "aligned_color_cam_K" in metadata:
        K = np.array(metadata["aligned_color_cam_K"], dtype=float)
    elif cam_k_path.exists():
        K = parse_cam_K_txt(cam_k_path)
    else:
        raise FileNotFoundError(
            "No camera intrinsics found. Expected metadata.aligned_color_cam_K or captures/cam_K.txt"
        )
    if K.shape != (3, 3):
        raise RuntimeError(f"Expected K shape (3, 3), got {K.shape}")

    stems_all = list_frame_stems(rgb_dir, depth_dir, masks_dir)
    if args.frame_index is None:
        stems = stems_all
    else:
        stems = [f"{idx:06d}" for idx in args.frame_index]
        missing = [s for s in stems if s not in stems_all]
        if missing:
            raise RuntimeError(f"Requested frame(s) not found: {missing}")

    example_root: Path = args.output_dir.resolve()
    example_root.mkdir(parents=True, exist_ok=True)
    label = sanitize_label(args.label)

    # Build one converted mesh folder shared by all frame examples.
    shared_mesh_dir = example_root / "meshes" / label
    if shared_mesh_dir.exists():
        if args.overwrite:
            shutil.rmtree(shared_mesh_dir)
        else:
            logger.info("Reusing existing converted mesh directory: %s", shared_mesh_dir)

    if not shared_mesh_dir.exists():
        converted_obj = shared_mesh_dir / mesh_obj.name
        mtllibs = scale_obj_vertices(mesh_obj, converted_obj, mesh_scale=args.mesh_scale)
        copy_mtl_and_textures(mesh_obj.parent, shared_mesh_dir, mtllibs)
        logger.info(
            "Converted mesh written to %s (mesh_scale=%s).",
            converted_obj,
            args.mesh_scale,
        )

    depth_scale_m_per_unit = metadata.get("depth_scale_m_per_unit", None)
    if depth_scale_m_per_unit is not None:
        depth_scale_m_per_unit = float(depth_scale_m_per_unit)

    logger.info(
        "Converting %d frame(s) from %s to %s",
        len(stems),
        source_dir,
        example_root,
    )

    converted = 0
    used_depth_mode: Optional[str] = None
    first_frame_dir: Optional[Path] = None
    for stem in stems:
        rgb_path = rgb_dir / f"{stem}.png"
        depth_path = depth_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"

        rgb = Image.open(rgb_path)
        rgb_arr = np.array(rgb)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise RuntimeError(f"Expected RGB image shape [H,W,3], got {rgb_arr.shape} for {rgb_path}")
        H, W = rgb_arr.shape[:2]

        depth_raw = np.array(Image.open(depth_path))
        if depth_raw.shape != (H, W):
            raise RuntimeError(f"Depth shape {depth_raw.shape} does not match RGB {(H, W)} for {stem}")

        mask = np.array(Image.open(mask_path))
        if mask.shape != (H, W):
            raise RuntimeError(f"Mask shape {mask.shape} does not match RGB {(H, W)} for {stem}")

        depth_mm, depth_mode_used = convert_depth_to_mm(
            depth_raw=depth_raw,
            depth_mode=args.depth_mode,
            depth_scale_m_per_unit=depth_scale_m_per_unit,
            mask=mask,
        )
        if used_depth_mode is None:
            used_depth_mode = depth_mode_used
        elif used_depth_mode != depth_mode_used:
            logger.warning(
                "Auto depth mode changed across frames (%s -> %s).",
                used_depth_mode,
                depth_mode_used,
            )
            used_depth_mode = depth_mode_used

        bbox_modal = extract_bbox_modal_from_mask(mask)

        frame_dir = example_root / stem
        if frame_dir.exists() and args.overwrite:
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

        # images
        shutil.copy2(rgb_path, frame_dir / "image_rgb.png")
        Image.fromarray(depth_mm).save(frame_dir / "image_depth.png")

        # camera file
        cam_data = {"K": K.tolist(), "resolution": [int(H), int(W)]}
        (frame_dir / "camera_data.json").write_text(json.dumps(cam_data))

        # detection file
        in_dir = frame_dir / "inputs"
        in_dir.mkdir(exist_ok=True)
        obj_data = [{"label": label, "bbox_modal": bbox_modal}]
        (in_dir / "object_data.json").write_text(json.dumps(obj_data))

        # mesh link/copy
        frame_mesh_object_dir = frame_dir / "meshes" / label
        maybe_symlink_or_copy_mesh(shared_mesh_dir, frame_mesh_object_dir, overwrite=args.overwrite)

        converted += 1
        if first_frame_dir is None:
            first_frame_dir = frame_dir

    logger.info("Done. Converted %d frame(s).", converted)
    logger.info("Depth mode used: %s", used_depth_mode)
    if first_frame_dir is not None:
        examples_root = (LOCAL_DATA_DIR / "examples").resolve()
        if first_frame_dir.is_relative_to(examples_root):
            example_name = str(first_frame_dir.relative_to(examples_root))
            logger.info(
                "Run example (detections): python -m megapose.scripts.run_inference_on_example %s --vis-detections",
                example_name,
            )
            logger.info(
                "Run example (inference): python -m megapose.scripts.run_inference_on_example %s --run-inference",
                example_name,
            )
        else:
            logger.info(
                "First frame written to %s (outside LOCAL_DATA_DIR/examples).",
                first_frame_dir,
            )


if __name__ == "__main__":
    main()
