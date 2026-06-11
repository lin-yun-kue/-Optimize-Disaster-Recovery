from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Literal,
    Callable,
    TypedDict,
    TypeVar,
    Generic,
)
from contextvars import ContextVar
import textwrap
from typing_extensions import override
import uuid
import os
import time
import math

from Generator.expressions import GetExpression, Var

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

if TYPE_CHECKING:
    from Generator.expressions import Expression

# Context var which holds the current graph in this execution context
CURRENT_GRAPH: ContextVar[JSTRXGenerator | None] = ContextVar("CURRENT_GRAPH", default=None)


class SimCallbackData(TypedDict):
    """Base callback data available in all callbacks."""

    sim_time: float


# ============================================================================
# MARK: Nodes
# ============================================================================

NodeType = Literal["QUEUE", "COMBI", "LINK", "NORMAL", "CONSOLIDATOR", "DYNAFORK", "FORK"]


@dataclass(kw_only=True)
class NodeBase:
    name: str
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self):
        g = CURRENT_GRAPH.get()

        if g is not None:
            g._register_node(self)

            # Set x and y coordinates
            if self.x == 0 and self.y == 0:
                self.x = (g.positionCount * 80.0) % 1200 + g.startPos[0]
                self.y = int(g.positionCount / (1200 / 80)) * 60.0 + g.startPos[1]
            # self.y = len(g.nodes) * 10.0
            # self.x = len(g.nodes) * 10.0

    @property
    def type(self) -> NodeType:
        raise NotImplementedError

    def to_definitions_string(self) -> str:
        raise NotImplementedError

    def to_initial_content_string(self) -> str:
        return ""

    def to_position_string(self) -> str:
        return f"POSNODE {self.name} {self.x} {self.y};"

    def _get_callback_variables(self) -> str:
        """Return the comma-separated list of variables to include in callback POST."""
        return "SimTime"

    def _register_callback(
        self,
        eventName: str,
        element_name: str,
        callback: Callable[[CBDataT], None],
    ) -> None:
        g = CURRENT_GRAPH.get()
        if g is not None:
            g._register_callback(
                Callback(
                    eventName=eventName,
                    element_name=element_name,
                    callback=callback,
                    callback_variables=self._get_callback_variables(),
                    node_type=self.type,
                )
            )

    def _register_action(
        self,
        eventName: str,
        element_name: str,
        action: Action,
    ) -> None:
        g = CURRENT_GRAPH.get()
        if g is not None:
            g._register_action(
                ActionStatement(
                    action=action,
                    eventName=eventName,
                    expression=element_name,
                    node_type=self.type,
                )
            )


# ============================================================================
# MARK: Queue
# ============================================================================


class QueueCallbackData(SimCallbackData):
    """Callback data for Queue nodes."""

    cur_count: int
    tot_count: int
    ave_wait: float
    ave_cont: float


defaultDrawCondition = Var("CurCount").__gt__(0.0)


@dataclass(kw_only=True)
class QueueNode(NodeBase):
    resource: str = "Default"
    initialContent: int = 0

    @property
    def type(self) -> NodeType:
        return "QUEUE"

    def to_definitions_string(self) -> str:
        return f"{self.type} {self.name} {self.resource};"

    def to_initial_content_string(self) -> str:
        return f"EZINITIALCONTENT {self.name} {self.initialContent};"

    def _get_callback_variables(self) -> str:
        return f"SimTime,{self.name}.CurCount,{self.name}.TotCount,{self.name}.AveWait,{self.name}.AveCont"

    def onEntry(self, callback: Callable[[QueueCallbackData], None]):
        self._register_callback("ONENTRY", self.name, callback)

    def linkTo(
        self,
        other: CombiNode,
        name: str = "",
        drawCondition: Expression = defaultDrawCondition,
        drawAmount: int = 1,
        releaseAmount: int | str = 1,
    ) -> Link:
        if getattr(other, "type", None) != "COMBI":
            raise TypeError(
                f"Queue '{self.name}' can only link to a COMBI node. "
                + f"Attempted to link to node '{getattr(other, 'name', repr(other))}' of type '{getattr(other, 'type', None)}'."
            )
        return Link(
            source=self,
            target=other,
            name=name,
            drawCondition=drawCondition,
            drawAmount=drawAmount,
            releaseAmount=releaseAmount,
        )


# ============================================================================
# MARK: Combi
# ============================================================================


@dataclass(kw_only=True)
class CombiNode(NodeBase):
    duration: int | float | str = 0
    priority: int = 0
    semaphore: int = 1

    @property
    def type(self) -> NodeType:
        return "COMBI"

    def to_definitions_string(self) -> str:
        return f"{self.type} {self.name};"

    def to_initial_content_string(self) -> str:
        return (
            f"PRIORITY {self.name} {self.priority};\n"
            f"SEMAPHORE {self.name} {self.semaphore};\n"
            f"DURATION {self.name} {self.duration};"
        )

    def _get_callback_variables(self) -> str:
        return f"SimTime,{self.name}.CurInst,{self.name}.TotInst,{self.name}.AveDur"  # ,{self.name}.Duration (Removed because not available durring onBeforeDraws)

    def onBeforeDraws(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("BEFOREDRAWS", self.name, callback)
        else:
            self._register_callback("BEFOREDRAWS", self.name, callback)

    def onStart(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("ONSTART", self.name, callback)
        else:
            self._register_callback("ONSTART", self.name, callback)

    def onBeforeEnd(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("BEFOREEND", self.name, callback)
        else:
            self._register_callback("BEFOREEND", self.name, callback)

    def onEnd(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("ONEND", self.name, callback)
        else:
            self._register_callback("ONEND", self.name, callback)

    def linkTo(
        self,
        other: NodeBase,
        name: str = "",
        drawCondition: Expression = defaultDrawCondition,
        drawAmount: int = 1,
        releaseAmount: int | str = 1,
    ) -> Link:
        return Link(
            source=self,
            target=other,
            name=name,
            drawCondition=drawCondition,
            drawAmount=drawAmount,
            releaseAmount=releaseAmount,
        )


# ============================================================================
# MARK: Activity
# ============================================================================


class ActivityCallbackData(SimCallbackData):
    """Callback data for Activity/Combi nodes."""

    cur_inst: int
    tot_inst: int
    ave_dur: float
    duration: float


@dataclass(kw_only=True)
class ActivityNode(NodeBase):
    duration: int | float | str = 0

    @property
    def type(self) -> NodeType:
        return "NORMAL"

    def to_definitions_string(self) -> str:
        return f"{self.type} {self.name};"

    def to_initial_content_string(self) -> str:
        return f"DURATION {self.name} {self.duration};"

    def _get_callback_variables(self) -> str:
        return f"SimTime,{self.name}.CurInst,{self.name}.TotInst,{self.name}.AveDur,{self.name}.Duration"

    def onStart(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("ONSTART", self.name, callback)
        else:
            self._register_callback("ONSTART", self.name, callback)

    def onBeforeEnd(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("BEFOREEND", self.name, callback)
        else:
            self._register_callback("BEFOREEND", self.name, callback)

    def onEnd(self, callback: Callable[[ActivityCallbackData], None] | Action):
        if isinstance(callback, Action):
            self._register_action("ONEND", self.name, callback)
        else:
            self._register_callback("ONEND", self.name, callback)

    def linkTo(
        self,
        other: NodeBase,
        name: str = "",
        drawCondition: Expression = defaultDrawCondition,
        drawAmount: int = 1,
        releaseAmount: int | str = 1,
    ) -> Link:
        if getattr(other, "type", None) == "COMBI":
            raise TypeError(
                f"Activity '{self.name}' (NORMAL) cannot link to a COMBI node '{other.name}'. "
                + "Combi nodes may only receive from QUEUE nodes."
            )
        return Link(
            source=self,
            target=other,
            name=name,
            drawCondition=drawCondition,
            drawAmount=drawAmount,
            releaseAmount=releaseAmount,
        )


# ============================================================================
# MARK: Link
# ============================================================================


class LinkCallbackData(SimCallbackData):
    """Callback data for Link callbacks."""

    pass  # Links only get sim_time


@dataclass(kw_only=True)
class Link:
    source: NodeBase
    target: NodeBase
    name: str = ""
    drawCondition: Expression = field(default_factory=lambda: defaultDrawCondition)
    drawAmount: int = 1
    releaseAmount: int | str = 1

    def __post_init__(self):
        src_type = getattr(self.source, "type", None)
        tgt_type = getattr(self.target, "type", None)

        if src_type == "QUEUE" and tgt_type != "COMBI":
            raise AssertionError(
                f"Invalid link: Queue '{self.source.name}' may only link to a Combi. "
                + f"Attempted to link to '{self.target.name}' of type '{tgt_type}'."
            )

        if tgt_type == "COMBI" and src_type != "QUEUE":
            raise AssertionError(
                f"Invalid link: Combi '{self.target.name}' may only receive from a Queue. "
                + f"Link source '{self.source.name}' is of type '{src_type}'."
            )

        g = CURRENT_GRAPH.get()
        if g is not None:
            g._register_link(self)

    def to_definitions_string(self) -> str:
        return f"LINK {self.name} {self.source.name} {self.target.name};"

    def to_initial_content_string(self) -> str:
        return textwrap.dedent(f"""
            ENOUGH {self.name} {self.drawCondition.to_jstrx()};
            DRAWUNTIL {self.name} nDraws;
            DRAWAMT {self.name} {self.drawAmount};
            DRAWORDER {self.name} 1;
            DRAWWHERE {self.name} 1;
            RELEASEAMT {self.name} {self.releaseAmount};
            RELEASEUNTIL {self.name} 0;
            RELEASEORDER {self.name} 1;
            RELEASEWHERE {self.name} 1;
            STRENGTH {self.name} 1;
            """)

    def to_position_string(self) -> str:
        return f"POSLINK {self.name} 0 1;"

    def onFlow(self, callback: Callable[[LinkCallbackData], None] | Action):
        g = CURRENT_GRAPH.get()
        if g is not None:
            if isinstance(callback, Action):
                g._register_action(
                    ActionStatement(
                        action=callback,
                        eventName="ONFLOW",
                        expression=self.name,
                        node_type="LINK",
                    )
                )
            else:
                g._register_callback(
                    Callback(
                        eventName="ONFLOW",
                        element_name=self.name,
                        callback=callback,
                        callback_variables="SimTime",
                        node_type="LINK",
                    )
                )


# ============================================================================
# MARK: Actions
# ============================================================================


@dataclass
class Action:
    def to_code_string(self) -> str:
        raise NotImplementedError


@dataclass
class ActionStatement:
    action: Action
    eventName: str
    expression: Expression | str
    node_type: NodeType

    def to_code_string(self) -> str:
        if isinstance(self.expression, str):
            return f"{self.eventName} {self.expression} {self.action.to_code_string()};"
        return f"{self.eventName} {self.expression.to_jstrx()} {self.action.to_code_string()};"


@dataclass
class PrintAction(Action):
    message: str

    @override
    def to_code_string(self) -> str:
        return f'PRINT trace "{self.message}"'


@dataclass
class AssignAction(Action):
    saveValueName: str
    value: Expression

    @override
    def to_code_string(self) -> str:
        return f"ASSIGN {self.saveValueName} {self.value.to_jstrx()}"


@dataclass
class AddToQueueAction(Action):
    queueName: str = field(init=False)
    amount: int = 1

    def __init__(self, queue: str | QueueNode, amount: int = 1):
        super().__init__()
        # Accept either a string name or a QueueNode instance
        if isinstance(queue, str):
            self.queueName = queue
        elif hasattr(queue, "name"):
            self.queueName = queue.name
        else:
            raise TypeError(f"Expected a QueueNode or string for queue, got {type(queue).__name__}")
        self.amount = amount

    @override
    def to_code_string(self) -> str:
        return f"ADDTOQUEUE {self.queueName} {self.amount}"


@dataclass
class RemoveFromQueueAction(Action):
    queueName: str = field(init=False)
    amount: int = 1

    def __init__(self, queue: str | QueueNode, amount: int = 1):
        super().__init__()
        # Accept either a string name or a QueueNode instance
        if isinstance(queue, str):
            self.queueName = queue
        elif hasattr(queue, "name"):
            self.queueName = queue.name
        else:
            raise TypeError(f"Expected a QueueNode or string for queue, got {type(queue).__name__}")
        self.amount = amount

    @override
    def to_code_string(self) -> str:
        return f"REMOVEFROMQUEUE {self.queueName} {self.amount}"


@dataclass
class PostSimStateAction(Action):
    @override
    def to_code_string(self) -> str:
        return "POSTSIMSTATE"


# ---------------- Event registration ----------------
CallbackData = ActivityCallbackData | QueueCallbackData | LinkCallbackData | SimCallbackData

CBDataT = TypeVar("CBDataT", bound=CallbackData)


@dataclass(kw_only=True)
class Callback(Generic[CBDataT]):
    eventName: str
    element_name: str | None
    callback: Callable[[CBDataT], None]
    callback_variables: str
    node_type: NodeType
    callback_id: str = ""

    def __post_init__(self):
        self.callback_id = self.eventName + "-" + self.node_type + "-" + str(uuid.uuid4())

    def to_code_string(self) -> str:
        variables = self.callback_variables.split(",")
        formatted_vars = ",".join(f"{{{var}}}" for var in variables)
        poststring = f'POST "CALLBACK:{self.callback_id}:{formatted_vars}"'

        if self.element_name:
            return f"{self.eventName} {self.element_name} {poststring};"
        else:
            return f"{self.eventName} {poststring};"


# ---------------- Get expressions ----------------
def Get(callback: Callable[[], float]) -> Expression:
    """callback cannot be a lambda function"""
    g = CURRENT_GRAPH.get()
    assert g is not None

    id = f"GET-{str(uuid.uuid4())}"
    g._get_functions[id] = callback
    return GetExpression(id)


NB = TypeVar("NB", bound="NodeBase")


# ---------------- Graph (context manager) ----------------
class JSTRXGenerator:
    def __init__(self):
        self.nodes: list[NodeBase] = []
        self.links: list[Link] = []
        self._next_link_id = 1
        self._token = None  # token returned by ContextVar.set
        self._code_block: str | None = None
        self._callbacks: dict[str, Callback[CallbackData]] = {}
        self._actions: list[ActionStatement] = []
        self._variables: dict[str, Expression] = {}
        self._savevalues: dict[str, Expression] = {}
        self._get_functions: dict[str, Callable[[], float]] = {}

        self.startPos = (100, 100)
        self.positionCount = 0

        self._register_node(QueueNode(name="AAAStartTrigger", initialContent=1))

    # Context manager API
    def __enter__(self) -> JSTRXGenerator:
        self._token = CURRENT_GRAPH.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            CURRENT_GRAPH.reset(self._token)
            self._token = None
        return False

    def _register_node(self, node: NodeBase):
        self.positionCount += 1
        name = node.name[:15] if len(node.name) > 15 else node.name

        existing_names = {n.name for n in self.nodes}
        original_name = name
        counter = 1
        while name in existing_names:
            # Ensure truncated version stays within 15 characters including suffix
            suffix = f"_{counter}"
            name = original_name[: 15 - len(suffix)] + suffix
            counter += 1

        # Assign final unique, truncated name
        node.name = name

        if node not in self.nodes:
            self.nodes.append(node)

    def _register_link(self, link: Link):
        if link.name == "":
            link.name = f"Link{self._next_link_id}"
            self._next_link_id += 1
        if link not in self.links:
            self.links.append(link)

    def add_node(self, node: NodeBase):
        self._register_node(node)

    def add_link(self, link: Link):
        self._register_link(link)

    def setCode(self, code: str):
        self._code_block = code

    def _link_initial_contents(self, link: Link) -> str:
        return link.to_initial_content_string()

    def _register_callback(self, callback: Callback[CallbackData]):
        self._callbacks[callback.callback_id] = callback

    def _register_action(self, action: ActionStatement):
        self._actions.append(action)

    def find_node(self, name: str, node_type: type[NB] = NodeBase) -> NB:
        for node in self.nodes:
            if node.name == name:
                if isinstance(node, node_type):
                    return node
                else:
                    raise TypeError(f"Node {name} is not of type {node_type}")
        raise ValueError(f"Node {name} not found")

    # Variable will evaluate expression on every reference
    def add_variable(self, name: str, expression: Expression):
        self._variables[name] = expression

    # Savevalue evaluates the expression once and stores the result
    def add_savevalue(self, name: str, expression: Expression):
        self._savevalues[name] = expression

    def onBeforeTimeAdvance(self, callback: Callable[[SimCallbackData], None]):
        self._register_callback(
            Callback(
                eventName="BEFORETIMEADVANCE",
                element_name=None,
                callback=callback,
                callback_variables="SimTime",
                node_type="NORMAL",  # Doesn't matter for global callbacks
            )
        )

    def onAfterTimeAdvance(self, callback: Callable[[SimCallbackData], None]):
        self._register_callback(
            Callback(
                eventName="AFTERTIMEADVANCE",
                element_name=None,
                callback=callback,
                callback_variables="SimTime",
                node_type="NORMAL",  # Doesn't matter for global callbacks
            )
        )

    def onIf(self, condition: str | Expression, action: Action):
        self._register_action(
            ActionStatement(
                action=action,
                eventName="IF",
                expression=condition,
                node_type="NORMAL",
            )
        )

    # type "POST" messages
    def _post_callback(self, message: str):
        # Message format: CALLBACK:<callback_id>:<value1>,<value2>,...
        parts = message.split(":", 2)  # Split into max 3 parts
        if parts[0] != "CALLBACK":
            return

        callback_id = parts[1]
        callback = self._callbacks.get(callback_id)
        if not callback:
            return

        # Parse the callback data based on node type
        data_str = parts[2] if len(parts) > 2 else ""
        callback_data = self._parse_callback_data(data_str, callback.node_type, callback.callback_variables)

        return callback.callback(callback_data)

    # type "GET" messages
    def _get_callback(self, message: str):
        # Message format: <get_id>

        callback = self._get_functions.get(message)
        if not callback:
            return
        return str(callback())

    def _parse_callback_data(self, data_str: str, node_type: NodeType, variables: str) -> CallbackData:
        """Parse the comma-separated values into a typed dictionary."""
        values = data_str.split(",")
        var_names = variables.split(",")

        # Create a mapping of variable names to values
        raw_data = {}
        for var_name, value in zip(var_names, values):
            # Extract the field name from dotted notation (e.g., "Resources.CurCount" -> "CurCount")
            field_name = var_name.split(".")[-1] if "." in var_name else var_name
            raw_data[field_name] = value

        # Parse based on node type
        if node_type == "QUEUE":
            return QueueCallbackData(
                sim_time=float(raw_data.get("SimTime", 0)),
                cur_count=int(raw_data.get("CurCount", 0)),
                tot_count=int(raw_data.get("TotCount", 0)),
                ave_wait=float(raw_data.get("AveWait", 0)),
                ave_cont=float(raw_data.get("AveCont", 0)),
            )
        elif node_type in ("COMBI", "NORMAL"):
            return ActivityCallbackData(
                sim_time=float(raw_data.get("SimTime", 0)),
                cur_inst=int(raw_data.get("CurInst", 0)),
                tot_inst=int(raw_data.get("TotInst", 0)),
                ave_dur=float(raw_data.get("AveDur", 0)),
                duration=float(raw_data.get("Duration", 0)),
            )
        elif node_type == "LINK":
            return LinkCallbackData(sim_time=float(raw_data.get("SimTime", 0)))
        else:
            # Global callbacks or unknown types
            return SimCallbackData(sim_time=float(raw_data.get("SimTime", 0)))

    # Generate the output file text
    def generate_jstrx(self) -> str:

        output_parts = [
            "GENTYPE Default;",
            "",
            # nodes definitions
            *(n.to_definitions_string() for n in self.nodes),
            "",
            # link definitions
            *(l.to_definitions_string() for l in self.links),
            "",
            # initial content for nodes
            *(line for n in self.nodes for line in (n.to_initial_content_string(),) if line),
            "",
            # initial content for links (allow generator to control how link bodies are generated)
            *(self._link_initial_contents(l) for l in self.links),
            "",
            "RANDOMSEED true;",
            "SEED 0;",
            "STREAMS 0;",
            "NUMRUNS 1;",
            "ENDCOND SimTime>=1000000;",
            "",
            # optional code section
            "/<CODE>",
            *(
                f"ONENTRY AAAStartTrigger VARIABLE {name} {expression.to_jstrx()};"
                for name, expression in self._variables.items()
            ),
            *(
                f"ONENTRY AAAStartTrigger SAVEVALUE {name} {expression.to_jstrx()};"
                for name, expression in self._savevalues.items()
            ),
            *(c.to_code_string() for c in self._callbacks.values()),
            *(c.to_code_string() for c in self._actions),
            *([self._code_block] if self._code_block else []),
            "/<CODE>",
            "",
            # link positions
            *(l.to_position_string() for l in self.links),
            "",
            # node positions
            *(n.to_position_string() for n in self.nodes),
            "",
            "RANKCOUNT 0;",
        ]
        # filter out empty strings and join
        out = "\n".join(part for part in output_parts if part is not None)
        return out + "\n"

    def write_jstrx(self):
        jstrx = self.generate_jstrx()
        filePath = os.path.join(os.getcwd(), f"generated/generated-{int(time.time())}.jstrx")
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        with open(filePath, "w") as f:
            f.write(jstrx)
        print(f"Simulation file written to: {filePath}")
        return filePath

    def layout(self, orientation: Literal["horizontal", "vertical", "grid"], *nodes: NodeBase):
        if len(nodes) <= 0:
            return
        if orientation == "horizontal":
            anchor = nodes[0]
            for i, node in enumerate(nodes[1:]):
                node.x = anchor.x + (i + 1) * 80
                node.y = anchor.y
        elif orientation == "vertical":
            anchor = nodes[0]
            for i, node in enumerate(nodes[1:]):
                node.x = anchor.x
                node.y = anchor.y + (i + 1) * 80
        elif orientation == "grid":
            anchor = nodes[0]
            width = math.ceil(math.sqrt(len(nodes)))
            for i, node in enumerate(nodes):
                node.x = anchor.x + (i + 1) % width * 80
                node.y = anchor.y + math.floor((i + 1) / width) * 80

    def moveCursor(self, x: int, y: int):
        self.startPos = (x, y)
        self.positionCount = 0
