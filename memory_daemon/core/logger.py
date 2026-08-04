import time
from cache.config import settings


DEBUG = getattr(settings, "DEBUG", True)


def debug(*args):
    if DEBUG:
        print("[DEBUG]", *args)


def info(*args):
    print("[INFO]", *args)


def warn(*args):
    print("[WARN]", *args)


def error(*args):
    print("[ERROR]", *args)


class Timer:

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        debug(f"{self.label}: {elapsed:.6f}s")
