from pathlib import Path

import torch.cuda
from ultralytics import YOLO
import shutil
import pandas as pd
from sahi.predict import predict, get_sliced_prediction

from pcb.utils import timer, get_logger
from pcb.model.utils import (read_yolo_labels_from_file, yolo_to_original_annot)
from pcb.visualizations.annotations import visualize_annotations

logger = get_logger(__name__)

@timer
def inference(dest_results_dir: Path, output_dir: Path, run_tiled_inference: bool = False):
    """
    Run inference on the test dataset using the best model - stores predicted labels in run/detect/predict/labels
    :param dest_results_dir:
    :param output_dir:
    :return:
    """
    # Load the best model
    best_model_path = dest_results_dir / 'weights/best.pt'
    model = YOLO(best_model_path)

    logger.info("Run inference on the test process")
    test_data_dir = output_dir / 'images' / 'val'

    model_device = 'cuda'  if torch.cuda.is_available() else 'cpu'
    if run_tiled_inference:
        logger.info("Running tiled inference on the test dataset:")
        predict(
            model_type="yolov8",
            model_path="best_model_path",
            model_device=model_device, # or 'cuda:0'
            model_confidence_threshold=0.4,
            source=test_data_dir,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
    else:
        logger.info("Running standard inference on the test dataset:")
        metrics = model(source=test_data_dir, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)
    logger.info("Inference completed.")

    return metrics


def main():
    # GLOBAL SETTINGS
    classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    root_dir = Path.cwd().parent.resolve()
    dest_results_dir = root_dir / 'results'
    dataset_dir = root_dir / 'PCB_DATASET'
    images_dir = dataset_dir / 'images'
    output_dir = dataset_dir / 'output'

    # LOAD ANNOTATIONS
    annot_df = pd.read_parquet(dataset_dir / 'annotation.parquet')

    metric = inference(dest_results_dir=dest_results_dir, output_dir=output_dir)
    logger.debug(f"Metrics: {metric}")

    # Copy the results directory to the root directory
    predict_dir = 'runs/detect/predict'
    dest_predict_dir = root_dir / 'results' / 'predict'

    shutil.copytree(predict_dir, dest_predict_dir, dirs_exist_ok=True)


    # SET FILE FOR ANALYSIS:
    # TODO: this is hard coded - choose randomly from test set
    test_name = '12_spurious_copper_05'

    # Read the YOLO labels from the test dataset
    file_path = dest_predict_dir / 'labels' / f"{test_name}.txt"
    yolo_labels = read_yolo_labels_from_file(file_path)

    logger.info(f"{images_dir}")
    pred_annot_df = yolo_to_original_annot(image_name=f"{test_name}.jpg", yolo_labels=yolo_labels,
                                           annot_df=annot_df, classes=classes)
    logger.info(pred_annot_df)
    visualize_annotations(image_name=f'{test_name}.jpg', images_dir=str(images_dir), annot_df=pred_annot_df,
                          is_subfolder=True)

    visualize_annotations(image_name=f'{test_name}.jpg', images_dir=str(images_dir), annot_df=annot_df,
                          is_subfolder=True)
