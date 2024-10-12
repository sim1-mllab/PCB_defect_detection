from ultralytics import YOLO
from pathlib import Path

# GLOBAL SETTINGS
root_dir = Path.cwd().parent.resolve()
output_dir = root_dir / 'PCB_DATASET' / 'output'

model_name = 'yolov8n'
# Additional arguments here
epochs = 10  # Number of epochs to train the model
batch = 16  # batch size, default 16. If batch=-1 it uses 60% of GPU memory, of set to float <1: uses % of GPU memory, if int, uses batch size - ONLY WHEN RUN ON GPU
imgsz = 640  # Image size
save_period = 1  # Save model every `save_period` epochs
verbose = True  # Print output metrics for further analysis
mixup = 0.3  # Blends two images and their labels, creating a composite image
# box=6.0 # Weight of the box loss component in the loss function
# cls = 2.0 # Weight of the classification loss in the total loss function

all_data_yaml = f"""
path: {output_dir}
train: images/train
val: images/val

names:
    0: missing_hole
    1: mouse_bite
    2: open_circuit
    3: short
    4: spur
    5: spurious_copper
"""

data_path = root_dir / 'process.yaml'

with open(data_path, 'w') as f:
    f.write(all_data_yaml)

# %%
project = f'pcb_{model_name}_all_epochs_{epochs}_batch_{batch}'  # save results to project/name
model_yolo = YOLO(f'{model_name}.pt')
result = model_yolo.train(data=str(data_path),
                          epochs=epochs,
                          batch=batch,
                          lr0=0.001,
                          lrf=0.0001,
                          imgsz=imgsz,
                          save_period=save_period,
                          verbose=verbose,
                          project=project,
                          mixup=mixup)
