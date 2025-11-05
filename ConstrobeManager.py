import os
import json
import time
from pyconstrobe import ProcessManager


class ConStrobeManager:
    def __init__(self, model_path: str, callback=None, verbose: bool = True):
        self.model_path = os.path.abspath(model_path)
        self.manager = ProcessManager(callback or self._default_callback)
        self.verbose = verbose
        self.isGetResult = False
        self.message_queue = []

        self.manager.finishRunFlag = False
        self.manager.gotTraceFlag = False

        self._load_model()

    # ---------------- Utility ----------------
    def _log(self, *args, **kwargs):
        """Handles controlled console logging."""
        if self.verbose:
            print(*args, **kwargs)

    def _write_message(self, cmd):
        """Wrapper for ProcessManager.write_message()"""
        self.manager.write_message(cmd)

    # ---------------- Internal ----------------
    def _default_callback(self, type, message):
        """Handles messages returned by ConStrobe."""
        try:
            if type == "RESULTS":
                parsed = json.loads(message)
                self.message_queue.append(parsed)
                self.isGetResult = True
            elif type == "TRACE":
                self._log(f"[TRACE] {message}")
                self.manager.gotTraceFlag = True
            elif type == "MESSAGE":
                self._log(f"[MSG] {message}")
        except Exception as e:
            self._log(f"Error parsing message: {e}")

    def _load_model(self):
        """Loads the specified model into ConStrobe."""
        cmd = f"LOAD {self.model_path};"
        self.manager.write_message(cmd)
        self._log(f"[INFO] Model loaded: {self.model_path}")

    def _wait_for_flag(self, flag_name: str, interval=0.1, timeout=60):
        """Waits synchronously until a specific flag becomes True."""
        start_time = time.time()
        while not getattr(self.manager, flag_name, False):
            time.sleep(interval)
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for flag: {flag_name}")

    # ---------------- Public API ----------------
    def set_attributes(self, **kwargs):
        """Sets multiple model attributes in a single command batch."""
        if not kwargs:
            return

        cmds = "\n".join(
            [f"SETATTRIBUTE {key} InitialContent {value};" for key, value in kwargs.items()]
        )
        self._write_message(cmds)

        time.sleep(0.1)
        self._log(f"[INFO] Sent {len(kwargs)} SETATTRIBUTE commands in one batch.")

    def run_model(self, animate=False):
        """Runs the model and waits for completion."""
        cmds = (
            "RESETMODEL;\n"
            f"SETANIMATE {str(animate).lower()};\n"
            "RUNMODEL;"
        )
        self._write_message(cmds)

        self.manager.finishRunFlag = False
        self._wait_for_flag("finishRunFlag")
        self._log("[INFO] Simulation completed.")

    def write_get_results(self):
        """Sends command to request simulation results."""
        self._wait_for_flag("finishRunFlag")
        self.manager.write_message("GETRESULTS;")

    def read_result(self):
        """Reads and returns the result once it becomes available."""
        start_time = time.time()
        while not self.isGetResult:
            time.sleep(0.1)
            if time.time() - start_time > 30:
                raise TimeoutError("Timeout waiting for results.")
        self.isGetResult = False
        return self.message_queue.pop(0)

    def get_trace(self, trace_name="trace"):
        """Retrieves trace data synchronously."""
        self.manager.gotTraceFlag = False
        self.manager.write_message(f"GETTRACE {trace_name};")
        self._wait_for_flag("gotTraceFlag")
        self._log("[INFO] Trace retrieved.")

    def close(self):
        """Closes the currently loaded model."""
        self.manager.write_message("CLOSE;")
        time.sleep(0.2)
        self._log("[INFO] Model closed.")

    def cleanup(self):
        """Releases ConStrobe resources and closes the connection."""
        code = self.manager.cleanup()
        self._log("[INFO] Connection closed. Exit code:", code)
        return code
