import logging
import sys


def setup_logger(name: str = "MLOpsPipeline") -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid duplication if logger is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s")

    # StreamHandler to log to terminal (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
