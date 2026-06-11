"""
decision_node.py — Generator extensions for policy-driven dispatch in ConStrobe.

Adds three things on top of JSTRXGenerator:
  - PostObsAction      : Action that emits a structured OBS: POST payload.
  - DecisionNode       : Wires a zero-duration COMBI gate with a GET round-trip
                         and per-slot semaphore routing.
  - DisasterActivityNode: A COMBI work unit with baked-in disaster metadata so
                          the observation builder can reference it by ID.

These classes are designed to be imported alongside JSTRXGenerator and used
inside a `with JSTRXGenerator() as g:` block, exactly like QueueNode / CombiNode.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from .JSTRXGenerator import (
    CURRENT_GRAPH,
    Action,
    ActionStatement,
    ActivityCallbackData,
    AddToQueueAction,
    AssignAction,
    Callback,
    CombiNode,
    NodeBase,
    QueueNode,
)
from .expressions import Expression, Literal, Var

if TYPE_CHECKING:
    from .JSTRXGenerator import JSTRXGenerator


# ---------------------------------------------------------------------------
# PostObsAction
# ---------------------------------------------------------------------------

@dataclass
class PostObsAction(Action):
    """
    Emits  POST "OBS:<callback_id>:<{Var1}>,<{Var2}>,..."
    so the Python bridge can distinguish observation payloads from normal
    CALLBACK: messages.

    Parameters
    ----------
    callback_id : str
        A stable string the bridge uses to route this payload (typically
        the DecisionNode's name).
    variable_names : list[str]
        ConStrobe variable/savevalue/attribute names whose current values
        should be included in the payload, e.g. ``["SimTime",
        "TruckQueue.CurCount", "_active_disasters"]``.
        These are emitted as ``{VarName}`` tokens inside the POST string so
        ConStrobe substitutes their live values at fire time.
    """

    callback_id: str
    variable_names: list[str]

    def to_code_string(self) -> str:
        # Build the  {VarName},{VarName},...  substitution string
        subs = ",".join(f"{{{v}}}" for v in self.variable_names)
        return f'POST "OBS:{self.callback_id}:{subs}"'


# ---------------------------------------------------------------------------
# DecisionNode
# ---------------------------------------------------------------------------

@dataclass
class DecisionNode:
    """
    Generates the complete dispatch-decision subgraph inside a JSTRX model.

    The subgraph consists of:
      - A zero-duration COMBI node (``self.combi``) that acts as the gate.
      - A ``_dispatch_choice`` savevalue initialised to -1.
      - On ONSTART of the COMBI:
          1. A ``PostObsAction`` that emits the current state as an OBS POST
             (non-blocking, Python receives it and assembles the observation).
          2. An ``AssignAction`` that calls ``Get(get_fn)`` — this **blocks**
             ConStrobe until ``ConStrobeProcessBridge`` answers with
             ``RESPONSE_TO_GET <action_index>``.
      - ``max_candidates`` semaphore ``QueueNode`` objects (``self.semaphores``).
      - ``max_candidates`` global IF guards routing each integer choice to the
        corresponding semaphore queue.

    After constructing a ``DecisionNode``, call
    ``node.combi.linkTo(intermediate_queue)`` to connect the entity flow, then
    ``semaphores[i].linkTo(work_activity_i)`` for each downstream branch.

    Parameters
    ----------
    name : str
        Base name used for the COMBI and all generated semaphore nodes.
    max_candidates : int
        Number of candidate slots (must equal ``max_visible_disasters`` in the
        Gym).  Determines how many semaphore queues are created.
    observation_variable_names : list[str]
        ConStrobe variable/attribute strings included in the OBS POST payload.
        Order must match what ``ConStrobeGym._build_obs()`` expects.
    get_fn : Callable[[], float]
        Python callable that blocks until an action is available and returns it
        as a float.  In practice this is
        ``ConStrobeProcessBridge._blocking_get_fn``; inject it after creating
        the DecisionNode via ``node.set_get_fn(fn)``.
    """

    name: str = "DispatchDecision"
    max_candidates: int = 8
    observation_variable_names: list[str] = field(default_factory=list)
    get_fn: Callable[[], float] | None = None

    # Set by __post_init__
    combi: CombiNode = field(init=False, repr=False)
    semaphores: list[QueueNode] = field(init=False, repr=False)
    _savevalue_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g = CURRENT_GRAPH.get()
        if g is None:
            raise RuntimeError(
                "DecisionNode must be created inside a 'with JSTRXGenerator() as g:' block."
            )

        self._savevalue_name = "_dispatch_choice"

        # --- COMBI decision gate (zero duration, serialised via SEMAPHORE 1) ---
        self.combi = CombiNode(name=self.name, duration=0, semaphore=1)

        # --- Semaphore queues, one per candidate slot ---
        self.semaphores = [
            QueueNode(name=f"{self.name}_Sem{i}", initialContent=0)
            for i in range(self.max_candidates)
        ]

        # --- Savevalue initialised to -1 (sentinel for "no decision yet") ---
        g.add_savevalue(self._savevalue_name, Literal(-1.0))

        # --- ONSTART action 1: emit observation POST (non-blocking) ---
        if self.observation_variable_names:
            self.combi.onStart(
                PostObsAction(
                    callback_id=self.name,
                    variable_names=self.observation_variable_names,
                )
            )

        # --- ONSTART action 2: assign choice = Get(get_fn) (blocking) ---
        # get_fn may be None at model-build time; it is injected before the
        # first run via set_get_fn().  We register a wrapper that delegates to
        # whatever is currently stored in _get_fn_holder so it can be replaced
        # without regenerating the JSTRX file.
        self._get_fn_holder: list[Callable[[], float]] = []

        def _dispatch_get_fn() -> float:
            if not self._get_fn_holder:
                raise RuntimeError(
                    f"DecisionNode '{self.name}': no get_fn registered. "
                    "Call node.set_get_fn(fn) before running the model."
                )
            return self._get_fn_holder[0]()

        _dispatch_get_fn.__name__ = f"get_{self.name}"

        from .JSTRXGenerator import Get  # local import to avoid circular at module level
        self.combi.onStart(AssignAction(self._savevalue_name, Get(_dispatch_get_fn)))

        # If a get_fn was supplied at construction time, register it now.
        if self.get_fn is not None:
            self._get_fn_holder.append(self.get_fn)

        # --- Global IF guards: route choice → semaphore ---
        for i, sem in enumerate(self.semaphores):
            g.onIf(
                Var(self._savevalue_name).eq(float(i)),
                AddToQueueAction(sem),
            )

        # Reset savevalue after routing to avoid stale value on the next cycle
        g.onIf(
            Var(self._savevalue_name) >= 0.0,
            AssignAction(self._savevalue_name, Literal(-1.0)),
        )

    def set_get_fn(self, fn: Callable[[], float]) -> None:
        """
        Inject or replace the callable that blocks until the Python policy
        delivers an action.  Can be called at any time before the next run.
        """
        self._get_fn_holder.clear()
        self._get_fn_holder.append(fn)


# ---------------------------------------------------------------------------
# DisasterActivityNode
# ---------------------------------------------------------------------------

@dataclass
class DisasterActivityNode:
    """
    A named COMBI work-activity node carrying disaster metadata.

    The metadata (id, type, scale, location) is stored as Python attributes
    so ``ConStrobeJSTRXBuilder`` can include them when composing the OBS POST
    variable list for the upstream ``DecisionNode``.

    In the JSTRX model the node is a plain COMBI with a computed duration
    (scale_cubic_yards / work_rate_per_minute) so entities spend the right
    amount of simulation time here.

    Parameters
    ----------
    disaster_id : int
        Monotonic integer assigned by ConStrobeJSTRXBuilder.
    disaster_type : str
        "landslide" | "wildfire_debris"
    scale_cubic_yards : float
        Total work in cubic yards.
    work_rate_per_minute : float
        Combined excavator+truck throughput rate (cubic yards / minute).
    x_norm : float
        Normalised x-coordinate in [0, 1] (UTM metres / CLATSOP_LOCAL_COORD_MAX).
    y_norm : float
        Normalised y-coordinate in [0, 1].
    name : str | None
        If None, auto-generated as "Disaster{disaster_id}".
    """

    disaster_id: int
    disaster_type: str
    scale_cubic_yards: float
    work_rate_per_minute: float
    x_norm: float
    y_norm: float
    name: str | None = None

    # Set by __post_init__
    combi: CombiNode = field(init=False, repr=False)

    def __post_init__(self) -> None:
        node_name = self.name or f"Disaster{self.disaster_id}"

        duration_minutes = (
            self.scale_cubic_yards / max(self.work_rate_per_minute, 0.001)
        )

        self.combi = CombiNode(name=node_name, duration=round(duration_minutes, 2))

    @property
    def one_hot_index(self) -> int:
        """Returns the disaster type one-hot index matching SimPy gym's convention."""
        return {"landslide": 0, "wildfire_debris": 1}.get(self.disaster_type, 0)
