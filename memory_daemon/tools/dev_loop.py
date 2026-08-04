from core.logger import debug
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Runner(FileSystemEventHandler):

    def __init__(self):
        self.last = 0

    def run_tests(self):

        debug("\n[DEV LOOP] Running regression...\n")

        subprocess.run(["python", "tests/regression.py"])

    def on_modified(self, event):

        if event.is_directory:
            return

        if time.time() - self.last < 2:
            return

        self.last = time.time()

        self.run_tests()


if __name__ == "__main__":

    observer = Observer()
    handler = Runner()

    observer.schedule(handler, path="memory/", recursive=True)
    observer.schedule(handler, path="api.py", recursive=False)

    observer.start()

    debug("[DEV LOOP] Active")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
