"""Safe arithmetic calculator. Rejects attribute access and names."""

from __future__ import annotations

import ast
import operator
from typing import Any

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")


_MAX_EXPRESSION_CHARS = 256
_MAX_AST_NODES = 64


def _count_nodes(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def calculator(expression: str) -> dict[str, Any]:
    if not isinstance(expression, str) or not expression.strip():
        return {"ok": False, "error": {"type": "invalid_expression", "message": "expression must be a non-empty string"}}
    if len(expression) > _MAX_EXPRESSION_CHARS:
        return {
            "ok": False,
            "error": {
                "type": "invalid_expression",
                "message": f"expression exceeds {_MAX_EXPRESSION_CHARS} characters",
            },
        }
    try:
        tree = ast.parse(expression, mode="eval")
        if _count_nodes(tree) > _MAX_AST_NODES:
            return {
                "ok": False,
                "error": {
                    "type": "invalid_expression",
                    "message": f"expression exceeds {_MAX_AST_NODES} AST nodes",
                },
            }
        value = _eval(tree)
    except Exception as exc:  # noqa: BLE001 — surface as tool error
        return {"ok": False, "error": {"type": "invalid_expression", "message": str(exc)}}
    if not isinstance(value, (int, float)) or value != value:  # NaN
        return {"ok": False, "error": {"type": "invalid_expression", "message": "non-finite result"}}
    return {"ok": True, "data": {"value": float(value)}}
