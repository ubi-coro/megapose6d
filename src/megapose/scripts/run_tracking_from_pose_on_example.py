from __future__ import annotations

# Standard Library
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Third Party
import torch

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.scene_dataset import ObjectData
from megapose.inference.types import PoseEstimatesType
from megapose.lib3d.transform import Transform
from megapose.scripts.run_inference_on_example import (
    load_detections,
    load_observation_tensor,
    make_object_dataset,
)
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level


logger = get_logger(__name__)


def resolve_example_dir(example_name_or_path: str) -> Path:
    path = Path(example_name_or_path).expanduser()
    if path.is_absolute():
        return path
    return LOCAL_DATA_DIR / "examples" / path


def save_predictions(path: Path, pose_estimates: PoseEstimatesType) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data: List[ObjectData] = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps([x.to_json() for x in object_data]))
    logger.info(f"Wrote predictions: {path}")


def run_tracking_from_pose(
    example_dir: Path,
    model_name: str,
    run_scoring: bool,
    n_refiner_iterations: int | None,
) -> None:
    model_info = NAMED_MODELS[model_name]
    inference_parameters = dict(model_info["inference_parameters"])
    if n_refiner_iterations is not None:
        inference_parameters["n_refiner_iterations"] = n_refiner_iterations

    observation = load_observation_tensor(
        example_dir, load_depth=model_info["requires_depth"]
    ).cuda()
    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    logger.info("Running full coarse + refiner inference.")
    full_output, full_extra = pose_estimator.run_inference_pipeline(
        observation,
        detections=detections,
        **inference_parameters,
    )
    save_predictions(example_dir / "outputs" / "object_data_full.json", full_output)

    logger.info("Running tracking from the full-output pose.")
    with torch.no_grad():
        tracking_output, tracking_extra = pose_estimator.run_tracking_pipeline(
            observation,
            initial_estimates=full_output,
            run_scoring=run_scoring,
            n_refiner_iterations=inference_parameters["n_refiner_iterations"],
        )
    save_predictions(example_dir / "outputs" / "object_data_tracking.json", tracking_output)

    print("Full inference timing:")
    print(f"  {full_extra.get('timing_str')}")
    print("Tracking-from-pose timing:")
    print(f"  {tracking_extra.get('timing_str')}")
    print(f"  scoring={'enabled' if run_scoring else 'disabled'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MegaPose tracking from an initial pose.")
    parser.add_argument("example", help="Example name under local_data/examples or an absolute path")
    parser.add_argument("--model", default="megapose-1.0-RGBD")
    parser.add_argument("--no-scoring", action="store_true")
    parser.add_argument("--n-refiner-iterations", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    set_logging_level("info")
    args = parse_args()
    run_tracking_from_pose(
        resolve_example_dir(args.example),
        model_name=args.model,
        run_scoring=not args.no_scoring,
        n_refiner_iterations=args.n_refiner_iterations,
    )
    # Panda3D renderer workers can hang during interpreter teardown after all
    # requested files and metrics are written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
