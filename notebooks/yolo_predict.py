from pathlib import Path
from ultralytics import YOLO
import shutil
import pandas as pd

from pcb.model.utils import (read_yolo_labels_from_file, yolo_to_original_annot)
from pcb.visualizations.annotations import visualize_annotations

# GLOBAL SETTINGS
classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
root_dir = Path.cwd().parent.resolve()
dest_results_dir = root_dir / 'results'
dataset_dir = root_dir / 'PCB_DATASET'
output_dir = dataset_dir / 'output'

# Load the best model
best_model_path = dest_results_dir / 'weights/best.pt'
model = YOLO(best_model_path)

# RUN INFERENCE on the test process
test_data_dir = output_dir / 'images/val'
metrics = model(source=test_data_dir, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)




# Copy the results directory to the root directory
predict_dir = '/content/runs/detect/predict'
dest_predict_dir = root_dir / 'results/predict'

shutil.copytree(predict_dir, dest_predict_dir)

# Read the YOLO labels from the test dataset
file_path = dest_predict_dir / 'labels/12_spurious_copper_10.txt'
yolo_labels = read_yolo_labels_from_file(file_path)

# %%
yolo_labels

# LOAD ANNOTATIONS
# %%
annot_df = pd.from_parquet(dataset_dir / 'annot_df.parquet')

# %%
annot_df.head()

# %%
pred_annot_df = yolo_to_original_annot(image_name='12_spurious_copper_10.jpg', yolo_labels=yolo_labels,
                                       annot_df=annot_df, classes=classes)
pred_annot_df.head()

#%%
visualize_annotations('12_spurious_copper_10.jpg', images_dir, pred_annot_df, is_subfolder=True)n
visualize_annotations('12_spurious_copper_10.jpg', images_dir, annot_df, is_subfolder=True)

