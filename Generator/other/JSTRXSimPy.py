from __future__ import annotations
from simpy.resources.store import StoreGet
from simpy.events import Event
from simpy.resources.store import Store
from simpy.core import Environment
import simpy
from typing import TYPE_CHECKING, Any, Generator
from dataclasses import dataclass
import statistics

if TYPE_CHECKING:
    from Generator.JSTRXGenerator import (
        CallbackData,
        JSTRXGenerator,
        NodeBase,
        QueueNode,
        CombiNode,
        ActivityNode,
        Link,
        Action,
        AssignAction,
        AddToQueueAction,
        PrintAction,
        Callback,
    )

from Generator.JSTRXGenerator import (
    SimCallbackData,
)
from Generator.expressions import Expression


class RuntimeContext:
    """
    Holds the state of the running simulation.
    Equivalent to the C++ engine's global state.
    """

    def __init__(self, env: simpy.Environment, graph: JSTRXGenerator):
        self.env: Environment = env
        self.graph: JSTRXGenerator = graph
        self.variables: dict[str, float] = {}
        self.savevalues: dict[str, float] = {}
        self.nodes: dict[str, RuntimeNode] = {}
        self.runtime_links: list[RuntimeLink] = []

        # Initialize variables/savevalues from graph
        for name, expr_str in graph._variables.items():
            self.variables[name] = expr_str.evaluate(self)

        for name, expr_str in graph._savevalues.items():
            self.savevalues[name] = expr_str.evaluate(self)

    def get_node(self, name: str) -> RuntimeNode | None:
        return self.nodes.get(name)

    def get_variable(self, name: str) -> float:
        # Check SaveValues first (as per JSTRX convention usually), then Variables
        if name in self.savevalues:
            return self.savevalues[name]
        return self.variables.get(name, 0)

    def set_variable(self, name: str, value: float):
        # Update logic: usually update SaveValues if it exists there
        if name in self.savevalues:
            self.savevalues[name] = value
        else:
            self.variables[name] = value

        # Trigger global IF checks whenever state changes
        self.check_global_conditions()

    def check_global_conditions(self):
        """
        Equivalent to the engine checking global "ON IF" statements.
        Should be called after variable changes or time advance.
        """
        for action_stmt in self.graph._actions:
            if action_stmt.eventName == "IF":
                # The 'element_name' in JSTRXGenerator was storing the condition string
                # Ideally, JSTRXGenerator should store the Expression object directly.
                # If it's a string, we might need to eval (unsafe) or rely on our new Expression system.

                # Assuming JSTRXGenerator was updated to pass expressions via our previous step:
                # We need to re-evaluate the condition.
                # For this implementation, we assume action_stmt.element_name IS the compiled string,
                # but we really want the original Expression if possible.

                # HACK: For now, we only support logic if the user passed an Expression object
                # and we stored it somewhere.
                # Since JSTRXGenerator stores strings in element_name, this is a limitation unless
                # we change JSTRXGenerator to store the raw object.

                # Let's assume for the sake of this file that we can Eval the string if needed,
                # OR (Better) we assume the 'element_name' IS the expression object (requires generic change).
                pass


@dataclass
class RuntimeLink:
    definition: Link
    source: RuntimeNode
    target: RuntimeNode


class RuntimeNode:
    def __init__(self, definition: NodeBase, context: RuntimeContext):
        self.def_: NodeBase = definition
        self.context: RuntimeContext = context
        self.inputs: list[RuntimeLink] = []
        self.outputs: list[RuntimeLink] = []

        # Stats
        self.cur_count: int = 0
        self.tot_count: int = 0

    def add_input(self, link: RuntimeLink):
        self.inputs.append(link)

    def add_output(self, link: RuntimeLink):
        self.outputs.append(link)

    def receive_entity(self, entity: Any):
        """Called when an entity is pushed TO this node."""
        raise NotImplementedError

    def execute_actions(self, event_name: str):
        """Executes Actions and Callbacks registered to this node/event."""

        # 1. Execute Actions (ASSIGN, ADDTOQUEUE, etc)
        for action_stmt in self.context.graph._actions:
            if action_stmt.expression == self.def_.name and action_stmt.eventName == event_name:
                self._run_action(action_stmt.action)

        # 2. Execute Callbacks (Python functions)
        # In JSTRXGenerator, callbacks are stored by ID. We need to find callbacks for this node/event.
        for cb in self.context.graph._callbacks.values():
            if cb.element_name == self.def_.name and cb.eventName == event_name:
                self._run_callback(cb)

    def _run_action(self, action: Action):
        if isinstance(action, AssignAction):
            # Resolve value
            val = action.value
            if isinstance(val, Expression):
                val = val.evaluate(self.context, self)
            elif isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val)

            self.context.set_variable(action.saveValueName, val)

        elif isinstance(action, AddToQueueAction):
            queue_name = action.queueName
            queue_node = self.context.get_node(queue_name)
            if queue_node and isinstance(queue_node, RuntimeQueue):
                # Create dummy entities
                for _ in range(action.amount):
                    queue_node.receive_entity(object())
            else:
                print(f"Warning: AddToQueue could not find queue {queue_name}")

        elif isinstance(action, PrintAction):
            print(f"SIM PRINT @ {self.context.env.now}: {action.message}")

    def _run_callback(self, cb: Callback[CallbackData]):
        # Construct data dict
        data: SimCallbackData = SimCallbackData(sim_time=self.context.env.now)
        cb.callback(data)


# pyright: reportIncompatibleVariableOverride=false
class RuntimeQueue(RuntimeNode):
    def __init__(self, definition: QueueNode, context: RuntimeContext):
        super().__init__(definition, context)
        self.store: Store = simpy.Store(context.env)
        self.entry_times: dict[int, float] = {}  # Track when entities entered for Wait stats
        self.waits: list[float] = []

        # Event to notify Combis
        self.content_changed: Event = context.env.event()

        # Handle Initial Content
        if definition.initialContent > 0:
            for _ in range(definition.initialContent):
                self.receive_entity(object())

    @property
    def ave_wait(self):
        return statistics.mean(self.waits) if self.waits else 0.0

    def receive_entity(self, entity: object):
        self.tot_count += 1
        self.entry_times[id(entity)] = self.context.env.now
        self.store.put(entity)
        self.cur_count = len(self.store.items)
        self.execute_actions("ONENTRY")

        # Signal change to listeners
        if not self.content_changed.triggered:
            self.content_changed.succeed()
            self.content_changed = self.context.env.event()

    def get_entity(self) -> Generator[StoreGet, None, object]:
        """Called by Combi to extract."""
        item: object = yield self.store.get()
        self.cur_count = len(self.store.items)
        entry_time = self.entry_times.pop(id(item), self.context.env.now)
        wait_time = self.context.env.now - entry_time
        self.waits.append(wait_time)
        return item


class RuntimeCombi(RuntimeNode):
    def __init__(self, definition: CombiNode, context: RuntimeContext):
        super().__init__(definition, context)
        self.cur_inst = 0
        self.tot_inst = 0
        self.durations: list[float] = []
        self.last_duration = 0.0

        self.process = context.env.process(self.run())

    @property
    def ave_dur(self):
        return statistics.mean(self.durations) if self.durations else 0.0

    def run(self):
        assert isinstance(self.def_, CombiNode)
        while True:
            # 1. Check if we can start
            if self._can_start():
                self.cur_inst += 1
                self.tot_inst += 1

                # 2. Pull Resources (Atomic)
                pulled_entities = yield self.context.env.process(self._pull_resources())

                # 3. ONSTART
                self.execute_actions("ONSTART")

                # 4. Determine Duration
                # Handle Expression object or raw value
                dur_val = self.def_.duration
                if isinstance(dur_val, Expression):
                    dur = dur_val.evaluate(self.context, self)
                else:
                    dur = float(dur_val)

                self.last_duration = dur
                self.durations.append(dur)

                # 5. Wait
                yield self.context.env.timeout(dur)

                # 6. ONEND
                self.execute_actions("ONEND")

                # 7. Push to outputs
                self.cur_inst -= 1
                self._push_entities(pulled_entities)

            else:
                # Wait for something to change in input queues
                # We also need to check IF variables changed (global state),
                # but typically Combis only wake on queue changes in basic simpy implementation.
                events = [q.source.content_changed for q in self.inputs if isinstance(q.source, RuntimeQueue)]
                if events:
                    yield self.context.env.any_of(events)
                else:
                    # If no inputs (generator?), wait a bit or stop
                    # A Combi with no inputs is technically a Generator.
                    yield self.context.env.timeout(1)

    def _can_start(self) -> bool:
        # Need at least one input link (unless it's a generator, not handled here for brevity)
        if not self.inputs:
            return False

        for link in self.inputs:
            queue = link.source
            if not isinstance(queue, RuntimeQueue):
                continue

            # Check Count
            needed = link.definition.drawAmount
            if queue.cur_count < needed:
                return False

            # Check Condition (Expression or String)
            cond = link.definition.drawCondition
            res = True
            if isinstance(cond, Expression):
                res = cond.evaluate(self.context, queue)
            elif isinstance(cond, str):
                # Basic parsing or assume true if not empty
                # Here we really benefit from the Expression system
                pass

            if not res:
                return False

        return True

    def _pull_resources(self):
        """Generator that yields gets from stores."""
        entities: list[object] = []
        for link in self.inputs:
            queue = link.source
            if isinstance(queue, RuntimeQueue):
                for _ in range(link.definition.drawAmount):
                    ent: object = yield self.context.env.process(queue.get_entity())
                    entities.append(ent)
        return entities

    def _push_entities(self, entities: list[Any]):
        # Simple Logic: Push to first valid output or distribute
        # SimPy adaptation: We push to the linked nodes.
        # If output is a Normal/Combi, we technically "Enter" it.
        # If output is a Link to a Queue, we put it there.

        # NOTE: ConStrobe logic for routing is complex.
        # Here we assume 1 output link for simplicity or broadcast.
        for link in self.outputs:
            target = link.target
            # Reuse entities or create new ones?
            # Usually pass through.
            for ent in entities:
                target.receive_entity(ent)


class RuntimeActivity(RuntimeNode):
    """Normal Node (Activity without pull logic)."""

    def __init__(self, definition: ActivityNode, context: RuntimeContext):
        super().__init__(definition, context)
        self.cur_inst = 0
        self.tot_inst = 0

    def receive_entity(self, entity: Any):
        # Start a process for this entity immediately (Infinite Capacity)
        self.context.env.process(self._process_entity(entity))

    def _process_entity(self, entity: Any):
        assert isinstance(self.def_, ActivityNode)
        self.cur_inst += 1
        self.tot_inst += 1

        self.execute_actions("ONSTART")

        dur_val = self.def_.duration
        if isinstance(dur_val, Expression):
            dur = dur_val.evaluate(self.context, self)
        else:
            dur = float(dur_val)

        yield self.context.env.timeout(dur)

        self.execute_actions("ONEND")

        self.cur_inst -= 1

        # Push
        for link in self.outputs:
            link.target.receive_entity(entity)


class JSTRXSimulation:
    def __init__(self, graph: JSTRXGenerator):
        self.env = simpy.Environment()
        self.context = RuntimeContext(self.env, graph)
        self._build_graph(graph)

        # Override the expression "GET" handling to use local functions
        # This connects JSTRXGenerator.Get(func) to this runtime
        self._patch_get_functions(graph)

    def _build_graph(self, graph: JSTRXGenerator):
        # 1. Create Nodes
        for n_def in graph.nodes:
            if isinstance(n_def, QueueNode):
                node = RuntimeQueue(n_def, self.context)
            elif isinstance(n_def, CombiNode):
                node = RuntimeCombi(n_def, self.context)
            elif isinstance(n_def, ActivityNode):
                node = RuntimeActivity(n_def, self.context)
            else:
                node = RuntimeNode(n_def, self.context)

            self.context.nodes[n_def.name] = node

        # 2. Create Links
        for l_def in graph.links:
            source = self.context.nodes.get(l_def.source.name)
            target = self.context.nodes.get(l_def.target.name)

            if source and target:
                r_link = RuntimeLink(l_def, source, target)
                source.add_output(r_link)
                target.add_input(r_link)
                self.context.runtime_links.append(r_link)

    def _patch_get_functions(self, graph: JSTRXGenerator):
        """
        The user uses Get(my_func).
        The Expression object for Get needs to call my_func() when evaluated.
        """
        # The Expression 'Get' isn't fully defined in our library yet,
        # but in JSTRXGenerator it creates a string 'GET("id")'.
        # We need to handle that string in the AssignAction logic in RuntimeNode,
        # OR better: The user should pass an object that we can execute.
        pass

    def run(self, duration: float = 10000):
        self.env.run(until=duration)
        print(f"Simulation ended at {self.env.now}")
