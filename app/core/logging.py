import logging
import sys
from time import time


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = ("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(logging.Formatter(fmt))
    root.setLevel(level)
    root.addHandler(handler)


class Timer:
    def __init__(self):
        self._start = time()

    def elapsed_ms(self) -> float:
        return (time() - self._start) * 1000.0

