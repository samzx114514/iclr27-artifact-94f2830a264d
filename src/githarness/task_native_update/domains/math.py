import ast
from decimal import Decimal, InvalidOperation
import operator

from .base import UpdateDomainContract

CONTRACT = UpdateDomainContract("math", "answer.md", ("calculator",), ("calculator",))

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate_expression(expression: str) -> str:
    """Evaluate bounded public arithmetic without Python evaluation."""

    if len(expression) > 500:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid arithmetic expression") from error

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (
                abs(right) > 20 or abs(left) > Decimal("1e20")
            ):
                raise ValueError("unsafe exponent")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        raise ValueError("calculator accepts numbers and + - * / % ** only")

    try:
        result = evaluate(tree)
    except (ArithmeticError, InvalidOperation) as error:
        raise ValueError(f"arithmetic failed: {error}") from error
    if not result.is_finite():
        raise ValueError("result must be finite")
    text = format(result.normalize(), "f")
    return "0" if text in {"", "-0"} else text
