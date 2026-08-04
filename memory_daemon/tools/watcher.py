from core.logger import debug
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ChangeHandler(FileSystemEventHandler):

    def __init__(self):
        self.last_run = 0

    def on_modified(self, event):

        if event.is_directory:
            return

        # debounce (prevents spam reload)
        if time.time() - self.last_run < 2:
            return

        self.last_run = time.time()

        debug("\n[WATCHER] Change detected → running regression...\n")

        subprocess.run([
            "python",
            "tests/regression.py"
        ])


if __name__ == "__main__":

    handler = ChangeHandler()
    observer = Observer()

    observer.schedule(handler, path="memory/", recursive=True)
    observer.schedule(handler, path="api.py", recursive=False)
    observer.schedule(handler, path="memory/", recursive=True)

    observer.start()

    debug("[WATCHER] Running...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
