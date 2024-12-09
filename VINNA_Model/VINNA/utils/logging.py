import sys
import logging
from os import makedirs
from os.path import dirname
from sys import stdout as _stdout


def setup_logging(log_file_path: str):
    """
    Sets up the logging
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    handlers = [logging.StreamHandler(_stdout)]

    if log_file_path:
        log_folder = dirname(log_file_path)
        makedirs(log_folder, exist_ok=True)
        handlers.append(logging.FileHandler(filename=log_file_path, mode='a'))

    logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=handlers)


def getLogger(name):
    """
    Retrieve the logger with the given name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)
