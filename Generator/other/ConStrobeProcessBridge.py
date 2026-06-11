"""
ConStrobeProcessBridge.py — Thread-safe observation/action synchronisation
for the ConStrobe subprocess.

Extends ProcessManager with two queues:
  _obs_queue   : background reader thread → main thread (gym)
                 Carries raw observation payload dicts assembled from OBS: POSTs.
  _action_queue: main thread (gym) → background reader thread
                 Carries integer actions that answer pending GET requests.

The lifecycle of one decision step:

  Reader thread                         Main thread (gym.step)
  ─────────────────────────────────     ───────────────────────────────
  receive MESSAGE "OBS:…"               …
  → _obs_queue.put(parsed_payload)  ──► wait_for_obs() returns payload
                                        policy runs → action = int
  receive GET <get_id>              ◄── deliver_action(action)
  → _action_queue.get() returns action
  → write "RESPONSE_TO_GET <float>"
  ConStrobe resumes
  …                                 ──► (next wait_for_obs)
"""
from __future__ import annotations

import queue
import threading
from typing import Any

from .ProcessManager import ProcessManager
from .JSTRXGenerator import JSTRXGenerator
from .ResultsParser import ResultsParser, SimulationResults


# ---------------------------------------------------------------------------
# Observation payload type
# ---------------------------------------------------------------------------

class ObsPayload:
    """
    Raw observation payload parsed from a ConStrobe OBS: POST message.

    Attributes
    ----------
    callback_id : str
        Identifies which DecisionNode fired (matches DecisionNode.name).
    values : list[str]
        Ordered list of raw string values from ConStrobe, one per variable
        in the DecisionNode's observation_variable_names list.
    """

    __slots__ = ("callback_id", "values")

    def __init__(self, callback_id: str, values: list[str]) -> None:
        self.callback_id = callback_id
        self.values = values

    def get_float(self, index: int, default: float = 0.0) -> float:
        try:
            return float(self.values[index])
        except (IndexError, ValueError):
            return default

    def __repr__(self) -> str:  # pragma: no cover
        return f"ObsPayload(id={self.callback_id!r}, n={len(self.values)})"


# ---------------------------------------------------------------------------
# ConStrobeProcessBridge
# ---------------------------------------------------------------------------

class ConStrobeProcessBridge(ProcessManager):
    """
    Extends ProcessManager with thread-safe obs/action synchronisation.

    Usage
    -----
    ::
        bridge = ConStrobeProcessBridge(exe_path, generator)
        bridge.load_jstrx(path)
        bridge.reset_model()
        bridge.set_animate(False)
        bridge.start_run()           # non-blocking

        obs = bridge.wait_for_obs()  # blocks until first decision point
        action = policy(obs)
        bridge.deliver_action(action)

        obs = bridge.wait_for_obs()  # blocks until next decision point
        ...

        if bridge.wait_for_done():   # True when FINISHED_RUN received
            results = bridge.last_results
    """

    # Sentinel placed in _obs_queue when the episode ends so that a blocking
    # wait_for_obs() call returns None rather than hanging.
    _EPISODE_DONE_SENTINEL: object = object()

    def __init__(
        self,
        exe_path: str,
        generator: JSTRXGenerator,
        *,
        obs_timeout: float = 60.0,
        action_timeout: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        exe_path : str
            Path to constrobe.exe.
        generator : JSTRXGenerator
            The same generator used to produce the .jstrx file.  Its
            ``_post_callback`` and ``_get_callback`` methods are still called
            for non-OBS messages and GET responses respectively.
        obs_timeout : float
            Seconds to wait for an observation before returning None
            (treated as truncation by the gym).
        action_timeout : float
            Seconds for the reader thread to wait for an action before raising
            (indicates a bug — gym didn't call deliver_action).
        """
        # ProcessManager.__init__ launches the subprocess and reader thread.
        super().__init__(exe_path)

        self._generator = generator
        self._obs_timeout = obs_timeout
        self._action_timeout = action_timeout

        # Queues that bridge the two threads
        self._obs_queue: queue.Queue[ObsPayload | object] = queue.Queue()
        self._action_queue: queue.Queue[int] = queue.Queue(maxsize=1)

        # Episode completion
        self._episode_done = threading.Event()
        self.last_results: SimulationResults | None = None

        # Register our handlers on top of the base class callbacks
        self.register_callback("MESSAGE", self._on_message)
        self.register_callback("GET", self._on_get)
        self.register_callback("FINISHED_RUN", self._on_finished)
        self.register_callback("RESULTS", self._on_results)

    # ------------------------------------------------------------------
    # Internal callbacks (called from the reader thread)
    # ------------------------------------------------------------------

    def _on_message(self, message: str) -> None:
        """Route POST messages: OBS: payloads go to _obs_queue; others to generator."""
        if message.startswith("OBS:"):
            payload = self._parse_obs_message(message)
            self._obs_queue.put(payload)
        else:
            # Regular CALLBACK: messages → generator's existing handler
            self._generator._post_callback(message)

    def _on_get(self, get_id: str) -> str:
        """
        Called from the reader thread when ConStrobe emits a GET request.
        Blocks until the main thread (gym.step) puts an action in _action_queue.
        Returns the float string that ConStrobe expects as RESPONSE_TO_GET.
        """
        try:
            action = self._action_queue.get(timeout=self._action_timeout)
        except queue.Empty:
            raise RuntimeError(
                f"ConStrobeProcessBridge: timed out after {self._action_timeout}s "
                "waiting for an action. Did you call deliver_action()?"
            )
        return str(float(action))

    def _on_finished(self, _message: str) -> None:
        """ConStrobe has finished the run; unblock wait_for_done() and any pending wait_for_obs()."""
        self._episode_done.set()
        # Unblock a gym that is still waiting for an obs that will never come
        self._obs_queue.put(self._EPISODE_DONE_SENTINEL)

    def _on_results(self, message: str) -> None:
        """Parse and store the final RESULTS payload."""
        try:
            self.last_results = ResultsParser.parse(message)
        except Exception:
            self.last_results = None

    # ------------------------------------------------------------------
    # Main-thread API (called by ConStrobeGym)
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """
        Begin a model run (non-blocking).
        Clears episode state from any prior run before issuing RUNMODEL.
        """
        self._episode_done.clear()
        self.last_results = None
        # Drain any stale items left over from a previous run
        while not self._obs_queue.empty():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                break
        while not self._action_queue.empty():
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                break
        self.run_model(blocking=False)

    def wait_for_obs(self) -> ObsPayload | None:
        """
        Block until the next OBS: POST arrives or the episode ends.

        Returns
        -------
        ObsPayload
            The next observation payload from ConStrobe.
        None
            If the episode ended (FINISHED_RUN received) or timed out.
        """
        try:
            item = self._obs_queue.get(timeout=self._obs_timeout)
        except queue.Empty:
            return None

        if item is self._EPISODE_DONE_SENTINEL:
            return None
        return item  # type: ignore[return-value]

    def deliver_action(self, action: int) -> None:
        """
        Deliver the policy's action to the reader thread, which forwards it
        to ConStrobe as the answer to the pending GET.

        Must be called exactly once per observation received from wait_for_obs().
        Raises if called when no GET is pending (action_queue already full).
        """
        try:
            self._action_queue.put_nowait(action)
        except queue.Full:
            raise RuntimeError(
                "ConStrobeProcessBridge.deliver_action called but no GET is pending. "
                "Possible double-deliver or out-of-order call."
            )

    def wait_for_done(self, timeout: float = 0.0) -> bool:
        """
        Check (or wait) for episode completion.

        Parameters
        ----------
        timeout : float
            If > 0, block up to this many seconds.  0 means non-blocking poll.

        Returns
        -------
        bool
            True if the episode is done.
        """
        if timeout > 0.0:
            return self._episode_done.wait(timeout=timeout)
        return self._episode_done.is_set()

    def blocking_get_fn(self) -> float:
        """
        A Callable[[], float] suitable for injection into DecisionNode.set_get_fn().

        This method IS the blocking GET handler from the Python policy side:
        it puts itself at the end of the queue and waits for deliver_action().
        In practice the reader thread calls _on_get (which calls
        _action_queue.get) rather than this method directly — this is provided
        as a convenience for direct use in testDecisions-style scripts.
        """
        action = self._action_queue.get(timeout=self._action_timeout)
        return float(action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_obs_message(message: str) -> ObsPayload:
        """
        Parse  "OBS:<callback_id>:<val0>,<val1>,..."  into an ObsPayload.

        The message arrives without the leading "OBS:" if the ProcessManager
        already stripped the type token — we handle both forms.
        """
        # Strip leading "OBS:" if still present (depends on how ProcessManager
        # delivers it — currently it passes the full body of the MESSAGE line)
        if message.startswith("OBS:"):
            message = message[4:]

        # Now: "<callback_id>:<val0>,<val1>,..."
        colon_pos = message.find(":")
        if colon_pos == -1:
            return ObsPayload(callback_id="unknown", values=[])

        callback_id = message[:colon_pos]
        values_str = message[colon_pos + 1:]
        values = values_str.split(",") if values_str else []
        return ObsPayload(callback_id=callback_id, values=values)

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    def cleanup(self) -> None:  # type: ignore[override]
        """
        Extend ProcessManager.cleanup to unblock any threads waiting on queues
        before joining the reader thread.
        """
        # Unblock any pending wait_for_obs
        self._obs_queue.put(self._EPISODE_DONE_SENTINEL)
        # Unblock any pending _on_get in the reader thread
        try:
            self._action_queue.put_nowait(-1)
        except queue.Full:
            pass
        super().cleanup()
