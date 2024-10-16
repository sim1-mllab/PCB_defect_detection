import logging
from typing import Literal
import functools
import time

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s() --  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the configuration defined in this file.

    :param name: Name of the logger (usually __name__ / for developing __file__).
    :return:     Logger with the specified name and configuration.
    """

    if "src/" in name:
        name = name.split("src/")[-1]

    logger = logging.getLogger(name)

    return logger


def timer(func: callable) -> callable:
    """
    Decorator to measure the time of a function
    :param func: any exectuable function
    :return:
    """
    logger = get_logger(__name__)

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        logger.info(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


def get_subfolder(
    image_name: str,
) -> Literal[
    "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"
]:
    """
    Get the subfolder name from the image name
    :param image_name: file name of the image
    :return: class name
    """
    if "missing" in image_name.split("_"):
        return "Missing_hole"
    if "mouse" in image_name.split("_"):
        return "Mouse_bite"
    if "open" in image_name.split("_"):
        return "Open_circuit"
    if "short" in image_name.split("_"):
        return "Short"
    if "spur" in image_name.split("_"):
        return "Spur"
    if "spurious" in image_name.split("_"):
        return "Spurious_copper"
