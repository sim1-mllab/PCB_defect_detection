from pathlib import Path
from ultralytics import YOLO
import shutil
import pandas as pd

from pcb.model.utils import (read_yolo_labels_from_file, yolo_to_original_annot)
from pcb.visualizations.annotations import visualize_annotations


def inference(dest_results_dir: Path, output_dir: Path):
    """
    Run inference on the test dataset using the best model
    :param dest_results_dir:
    :param output_dir:
    :return:
    """
    # Load the best model
    best_model_path = dest_results_dir / 'weights/best.pt'
    model = YOLO(best_model_path)

    # RUN INFERENCE on the test process
    test_data_dir = output_dir / 'images' / 'val'
    metrics = model(source=test_data_dir, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)

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


    pred_annot_df = yolo_to_original_annot(image_name=f"{test_name}.jpg", yolo_labels=yolo_labels,
                                           annot_df=annot_df, classes=classes)

    visualize_annotations(image_name=f'{test_name}.jpg', images_dir=images_dir, annot_df=pred_annot_df,
                          is_subfolder=True)

    visualize_annotations(image_name=f'{test_name}.jpg', images_dir=images_dir, annot_df=annot_df, is_subfolder=True)
