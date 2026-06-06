from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import heapq
import itertools
import random
import math


# =========================
# Engine (generic DES core)
# =========================

@dataclass(order=True)
class Event:
    # heap ordering: time -> priority -> seq
    time: float
    priority: int
    seq: int = field(compare=True)
    etype: str = field(compare=False, default="")
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)


class Engine:
    """
    Generic discrete-event engine.
    - Maintains a min-heap of events
    - Advances simulation time to next event
    - Dispatches to handlers
    - Tracks time-weighted metrics hook (optional)
    """
    def __init__(self, seed: int = 42):
        self.now: float = 0.0
        self._last: float = 0.0
        self._heap: List[Event] = []
        self._seq = itertools.count()
        self.handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self.rng = random.Random(seed)

        # optional hooks
        self.on_advance_time: Optional[Callable[[float], None]] = None  # dt -> None

    def register(self, etype: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        self.handlers[etype] = handler

    def schedule(self, time: float, etype: str, payload: Dict[str, Any] | None = None, priority: int = 50) -> None:
        if payload is None:
            payload = {}
        ev = Event(time=time, priority=priority, seq=next(self._seq), etype=etype, payload=payload)
        heapq.heappush(self._heap, ev)

    def run(self, until_time: Optional[float] = None, until_events: Optional[int] = None) -> None:
        processed = 0
        while self._heap:
            ev = heapq.heappop(self._heap)
            if until_time is not None and ev.time > until_time:
                # stop before executing events beyond horizon
                break

            # advance time and update time-weighted metrics
            dt = ev.time - self._last
            if dt < -1e-12:
                raise RuntimeError(f"Time went backwards: {ev.time} < {self._last}")
            if dt > 0 and self.on_advance_time is not None:
                self.on_advance_time(dt)

            self.now = ev.time
            self._last = ev.time

            handler = self.handlers.get(ev.etype)
            if handler is None:
                raise KeyError(f"No handler registered for event type: {ev.etype}")

            handler(ev.payload)

            processed += 1
            if until_events is not None and processed >= until_events:
                break
