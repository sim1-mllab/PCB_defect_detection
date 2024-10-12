from ultralytics import YOLO
from pathlib import Path
from pcb.utils import get_logger

logger = get_logger(__name__)


def train(data_path: Path = Path.cwd() / 'process.yaml',
          model_name: str = 'yolov8n', epochs: int = 10, batch: int = 16, imgsz: int = 640, save_period: int = 1,
          verbose: bool = True, mixup: float = 0.3):
    """
    Train a YOLO model
    :param model_name:
    :param epochs:
    :param batch:
    :param imgsz:
    :param save_period:
    :param verbose:
    :param mixup:
    :return:
    """
    # ToDo: project name pattern to be consistent with the rest of the project - pydantic
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

    return result


def main():
    # GLOBAL SETTINGS
    root_dir = Path.cwd().parent.resolve()
    output_dir = root_dir / 'PCB_DATASET' / 'output'

    # TODO: put this into a config file and let the method handle the config file
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

    train(data_path=data_path, model_name='yolov8n', epochs=1, batch=16, imgsz=640, save_period=1, verbose=True,
          mixup=0.3)
