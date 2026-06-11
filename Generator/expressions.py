from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
import operator
from typing_extensions import override

# if TYPE_CHECKING:
#     from Generator.JSTRXSimPy import RuntimeContext, RuntimeNode


class Expression(ABC):
    """Base class for all logic expressions."""

    def __str__(self) -> str:
        raise Exception("Expressions should not be converted to strings.")

    def to_jstrx(self) -> str:
        """Convert to ConStrobe string format."""
        raise NotImplementedError

    # @abstractmethod
    # def evaluate(self, context: RuntimeContext, node: RuntimeNode | None = None) -> float:
    #     """Execute logic in Python (SimPy backend)."""
    #     raise NotImplementedError

    # --- Operator Overloading ---
    # Comparisons
    def eq(self, other: object) -> BinaryOp:  # type: ignore[override]
        return BinaryOp(self, other, "==", operator.eq)

    def ne(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "!=", operator.ne)

    def __gt__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, ">", operator.gt)

    def __ge__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, ">=", operator.ge)

    def __lt__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "<", operator.lt)

    def __le__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "<=", operator.le)

    # Math
    def __add__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "+", operator.add)

    def __sub__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "-", operator.sub)

    def __mul__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "*", operator.mul)

    def __truediv__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "/", operator.truediv)

    # Logic (Bitwise operators used for boolean logic due to Python limitations)
    def __and__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "&&", lambda a, b: a and b)

    def __or__(self, other: object) -> BinaryOp:
        return BinaryOp(self, other, "||", lambda a, b: a or b)


def wrap(val: float | Expression) -> Expression:
    """Helper to ensure raw numbers/strings become Expression objects."""
    if isinstance(val, Expression):
        return val
    return Literal(val)


@dataclass
class Literal(Expression):
    value: float | Expression

    @override
    def to_jstrx(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        return str(self.value)

    # @override
    # def evaluate(self, context: RuntimeContext, node: RuntimeNode | None = None) -> float:
    #     if isinstance(self.value, Expression):
    #         return self.value.evaluate(context, node)
    #     return self.value


@dataclass
class Var(Expression):
    """Represents a Global Variable, SaveValue, or Attribute."""

    name: str

    @override
    def to_jstrx(self) -> str:
        return self.name

    # @override
    # def evaluate(self, context: RuntimeContext, node: RuntimeNode | None = None) -> float:
    #     # 1. Try Context Variables (Globals)
    #     if self.name in context.variables:
    #         return context.variables[self.name]

    #     # 2. Try Node Attributes (Local scope)
    #     # We assume 'node' has these attributes exposed as properties
    #     if node:
    #         # Map "CurCount" -> node.cur_count
    #         attr = self.name.lower()
    #         if hasattr(node, attr):
    #             return getattr(node, attr)

    #         # Map "QueueName.CurCount" -> context.nodes["QueueName"].cur_count
    #         if "." in self.name:
    #             node_name, attr_name = self.name.split(".", 1)
    #             target_node = context.nodes.get(node_name)
    #             if target_node and hasattr(target_node, attr_name.lower()):
    #                 return getattr(target_node, attr_name.lower())

    #     return 0.0


@dataclass
class GetExpression(Expression):
    # callback: Callable[[], float]
    id: str

    # @override
    # def evaluate(self, context: RuntimeContext, node: RuntimeNode | None = None) -> float:
    #     return self.callback()

    @override
    def to_jstrx(self) -> str:
        return f'GET("{self.id}")'


@dataclass
class BinaryOp(Expression):
    left: Expression
    right: Expression
    symbol: str
    op_func: Callable[[float, float], float]

    def __init__(self, left: Expression, right: object, symbol: str, op_func: Callable[[float, float], float]):
        if isinstance(right, int):
            right = float(right)
        if not isinstance(right, Expression | float):
            raise TypeError(f"Binary Operation expected a float or Expression, got {type(right).__name__}")

        super().__init__()
        self.left = wrap(left)
        self.right = wrap(right)
        self.symbol = symbol
        self.op_func = op_func

    @override
    def to_jstrx(self) -> str:
        return f"{self.left.to_jstrx()}{self.symbol}{self.right.to_jstrx()}"

    # @override
    # def evaluate(self, context: RuntimeContext, node: RuntimeNode | None = None) -> float:
    #     return self.op_func(self.left.evaluate(context, node), self.right.evaluate(context, node))
