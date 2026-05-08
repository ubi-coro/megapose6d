# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import cv2
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.rigid_mesh_database import MeshDataBase
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)


def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
) -> None:
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    detections = load_detections(example_dir)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / "detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:

    model_info = NAMED_MODELS[model_name]

    observation = load_observation_tensor(
        example_dir, load_depth=model_info["requires_depth"]
    ).cuda()
    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )

    save_predictions(example_dir, output)
    return


def project_points_to_image(
    points_3d: np.ndarray,
    TCO: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate((points_3d, np.ones((len(points_3d), 1))), axis=1)
    points_cam = (TCO @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((len(points_3d), 2), np.nan, dtype=np.float64)
    if valid.any():
        projected = (K @ points_cam[valid].T).T
        uv[valid] = projected[:, :2] / projected[:, 2:3]
    return uv, valid


def draw_projected_line(
    image: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    i: int,
    j: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if not (valid[i] and valid[j]):
        return
    p1 = tuple(np.round(uv[i]).astype(np.int64))
    p2 = tuple(np.round(uv[j]).astype(np.int64))
    cv2.line(image, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def make_pose_bbox_and_axes_overlay(
    rgb: np.ndarray,
    camera_data: CameraData,
    object_datas: List[ObjectData],
    object_dataset: RigidObjectDataset,
) -> np.ndarray:
    vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mesh_db = MeshDataBase.from_object_ds(object_dataset)

    bbox_by_label = {}
    for label, mesh in mesh_db.meshes.items():
        obj = mesh_db.obj_dict[label]
        points_m = np.array(mesh.vertices, dtype=np.float64) * obj.scale
        bbox_by_label[label] = (points_m.min(axis=0), points_m.max(axis=0))

    K = np.asarray(camera_data.K, dtype=np.float64)
    bbox_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for object_data in object_datas:
        if object_data.TWO is None:
            continue
        if object_data.label not in bbox_by_label:
            logger.warning(f"Skipping unknown object label in visualization: {object_data.label}")
            continue

        bbox_min, bbox_max = bbox_by_label[object_data.label]
        x_min, y_min, z_min = bbox_min
        x_max, y_max, z_max = bbox_max
        bbox_corners = np.array(
            [
                [x_min, y_max, z_max],
                [x_max, y_max, z_max],
                [x_max, y_min, z_max],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_max, y_max, z_min],
                [x_max, y_min, z_min],
                [x_min, y_min, z_min],
            ],
            dtype=np.float64,
        )

        TCO = np.asarray(object_data.TWO.matrix, dtype=np.float64)
        uv_bbox, valid_bbox = project_points_to_image(bbox_corners, TCO, K)
        for i, j in bbox_edges:
            draw_projected_line(
                vis_bgr,
                uv_bbox,
                valid_bbox,
                i,
                j,
                color=(0, 255, 0),
                thickness=2,
            )

        axis_scale = float(np.max(bbox_max - bbox_min))
        if axis_scale <= 0:
            axis_scale = 0.05
        axis_scale *= 0.6
        axis_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_scale, 0.0, 0.0],
                [0.0, axis_scale, 0.0],
                [0.0, 0.0, axis_scale],
            ],
            dtype=np.float64,
        )
        uv_axes, valid_axes = project_points_to_image(axis_points, TCO, K)
        if valid_axes[0]:
            origin = tuple(np.round(uv_axes[0]).astype(np.int64))
            axis_colors = [
                (0, 0, 255),  # X (red)
                (0, 255, 0),  # Y (green)
                (255, 0, 0),  # Z (blue)
            ]
            for axis_idx, color in zip([1, 2, 3], axis_colors):
                if not valid_axes[axis_idx]:
                    continue
                endpoint = tuple(np.round(uv_axes[axis_idx]).astype(np.int64))
                cv2.arrowedLine(
                    vis_bgr,
                    origin,
                    endpoint,
                    color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.15,
                )

    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def make_output_visualization(
    example_dir: Path,
) -> None:

    rgb, _, camera_data = load_observation(example_dir, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)
    pose_overlay = make_pose_bbox_and_axes_overlay(
        rgb=rgb,
        camera_data=camera_data,
        object_datas=object_datas,
        object_dataset=object_dataset,
    )

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_pose_overlay = plotter.plot_image(pose_overlay)
    fig_all = gridplot(
        [[fig_rgb, fig_contour_overlay, fig_mesh_overlay, fig_pose_overlay]],
        toolbar_location=None,
    )
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_pose_overlay, filename=vis_dir / "pose_bbox_axes_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")
    logger.info(f"Wrote visualizations to {vis_dir}.")
    return


# def make_mesh_visualization(RigidObject) -> List[Image]:
#     return


# def make_scene_visualization(CameraData, List[ObjectData]) -> List[Image]:
#     return


# def run_inference(example_dir, use_depth: bool = False):
#     return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        make_detections_visualization(example_dir)

    if args.run_inference:
        run_inference(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)
