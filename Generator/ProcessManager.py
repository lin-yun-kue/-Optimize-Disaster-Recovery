import atexit
from threading import Thread
from subprocess import Popen
import subprocess
import threading
import time
from typing import Literal, Callable, cast

MessageType = Literal["FINISHED_RUN", "CLOSED_CONSTROBE", "GET", "TRACE", "RESULTS", "MESSAGE"]


class ProcessManager:
    def __init__(self, path_to_exe: str = r"C:\Program Files\constrobe\constrobe\constrobe.exe"):
        self.process: Popen[str] = subprocess.Popen(
            [path_to_exe, "--from-python"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

        self.callbacks: dict[MessageType, Callable[[str], str | None]] = {}
        self.keep_reading: bool = True  # Flag to control the reading loop
        self.finishRunFlag: bool = False
        self.gotTraceFlag: bool = False
        self.gotResultsFlag: bool = False
        self.reader_thread: Thread = threading.Thread(target=self.read_messages)
        self.reader_thread.daemon = True  # Allow thread to exit when main program exits
        self.reader_thread.start()

        self._cleaned_up = False
        atexit.register(self.cleanup)

    def load_jstrx(self, path: str):
        self.write_message(f"LOAD {path};")

    def reset_model(self):
        self.write_message("RESETMODEL;")

    def set_animate(self, animate: bool):
        self.write_message(f'SETANIMATE {"true" if animate else "false"};')

    def run_model(self, blocking: bool = False):
        self.finishRunFlag = False
        self.write_message("RUNMODEL;")

        if blocking:
            while self.finishRunFlag == False:
                time.sleep(0.5)
            # fetch results once the run is done
            self.write_message("GETRESULTS;")
            self.write_message("GETTRACE;")

    def fetch_results(self):
        """Explicitly request results and trace after a non-blocking run."""
        self.write_message("GETRESULTS;")
        self.write_message("GETTRACE;")

    def close(self):
        self.write_message("CLOSE;")

    def write_message(self, message: str):
        # print(f"Sending message: {message}")
        assert self.process.stdin is not None

        self.process.stdin.write(message + "\n")
        self.process.stdin.flush()

    def read_messages(self):
        """Read messages from the process's stdout."""
        assert self.process.stdout is not None
        while self.keep_reading:
            response = self.process.stdout.readline().strip()
            # print(f"Received message: {response}")

            if response:
                parts = response.split(" ", 1)
                type: MessageType = cast(MessageType, parts[0])
                message = parts[1] if len(parts) > 1 else ""

                callback = self.callbacks.get(type)
                response_message = ""
                if callback:
                    # print(f"Invoking callback for message type: {type}")
                    response_message = callback(message)

                if type == "FINISHED_RUN":
                    self.finishRunFlag = True
                elif type == "CLOSED_CONSTROBE":
                    self.finishRunFlag = True
                    self.keep_reading = False  # Stop reading messages
                elif type == "GET":
                    self.write_message(f"RESPONSE_TO_GET {response_message}")
                elif type == "TRACE":
                    self.gotTraceFlag = True
                elif type == "RESULTS":
                    self.gotResultsFlag = True

    def register_callback(self, type: MessageType, callback: Callable[[str], str | None]):
        self.callbacks[type] = callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.cleanup()
        return False

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True

        print("Cleaning up resources...")
        self.keep_reading = False

        print("Closing stdin...")
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        except OSError:
            pass

        print("Terminating ConStrobe process...")
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not terminate, force killing it...")
                self.process.kill()
                self.process.wait()

        print("Waiting for reader thread to finish...")
        self.reader_thread.join(timeout=2)

        print("Closing remaining pipes...")
        try:
            if self.process.stdout and not self.process.stdout.closed:
                self.process.stdout.close()
        except OSError:
            pass
        try:
            if self.process.stderr and not self.process.stderr.closed:
                self.process.stderr.close()
        except OSError:
            pass
