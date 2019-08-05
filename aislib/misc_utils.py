import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Creates a logger with a debug level and a custom format.

    :param name: Name to give logger.
    :return: Logger object.
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        '%H:%M:%S')
    handler.setFormatter(formatter)
    logger_.addHandler(handler)
    return logger_


def ensure_path_exists(path: Path,
                       is_folder: bool = False
                       ) -> Path:
    """
    Small utility function to ensure a path exists before trying to save a file.

    :param path: Path to make directories for.
    :param is_folder: Whether the output is a folder or not, if not we just
    create the parent folder.
    :return: The generated path the function just created.
    """
    if not is_folder:
        path = Path('/'.join(path.parts[:-1]))

    path.mkdir(parents=True, exist_ok=True)

    return path
