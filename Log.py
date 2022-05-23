import logging
import sys


class Log:
    def __init__(self) -> None:
        self.logger = self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger("logger")
        h = logging.FileHandler("logs.log")
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        h.setLevel(logging.INFO)
        logger.addHandler(h)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(h)
        
        return logger

    def __call__(self, message) -> None:
        self.logger.warn(message)
