import timeit
from typing import Literal
from ultralytics import YOLO
from pathlib import Path
from pcb.utils import get_logger
import torch
logger = get_logger(__name__)

def _is_mps_available() -> bool:
    """
    Check if MPS is available
    :return: boolean whether MPS is available
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.warning("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            logger.warning("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

        return False

    return True


def train(data_path: Path = Path.cwd() / 'process.yaml',
          model_name: str = 'yolov8n', epochs: int = 10, batch: int = 16, imgsz: int = 640, save_period: int = 1,
          verbose: bool = True, mixup: float = 0.3, device: Literal['cpu', 'mpg', 'cpu'] | None = None,
          mosaic: float=1.0):
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
    logger.info("Start training.")
    # ToDo: project name pattern to be consistent with the rest of the project - pydantic
    project = f'pcb_{model_name}_all_epochs_{epochs}_batch_{batch}'  # save results to project/name

    if device == 'mps':
        # ToDo: current implementation only considers MPS for MacOS or CPUs and not for GPUs
        if not _is_mps_available():
            logger.warning("MPS is not available. Switching to CPU.")
            logger.info("If GPUs are available, please specify the device as 'cuda'.")
            device = 'cpu'

    logger.info(f"Storing data in project: {project}")
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
                              mixup=mixup,
                              device=device,
                              mosaic=mosaic
                              )

    logger.info("Training completed.")
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

    start_time = timeit.timeit()
    train(data_path=data_path, model_name='yolov8n', epochs=10, batch=16, imgsz=640, save_period=5, verbose=True,
          mixup=0.3, device='cpu', mosaic=1.0)
    end_time = timeit.timeit()
    total_time = end_time - start_time
    logger.info(f"Training took {total_time} seconds, which is equivalent to {total_time / 60} minutes")


if __name__ == '__main__':
    main()