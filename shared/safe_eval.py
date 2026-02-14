"""Safe expression evaluator used by planner/runtime conditions.

Supports a constrained Python expression subset with no arbitrary code execution.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping


class SafeExpressionError(ValueError):
    """Raised when an expression contains unsupported/unsafe constructs."""


_ALLOWED_FUNCS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
}


class _SafeEvaluator:
    def __init__(self, context: Mapping[str, Any]):
        self.context = context

    def eval(self, expression: str) -> Any:
        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            name = node.id
            if name in {"True", "False", "None"}:
                return {"True": True, "False": False, "None": None}[name]
            if name in {"true", "false", "null", "none"}:
                return {"true": True, "false": False, "null": None, "none": None}[name]
            return self.context.get(name)

        if isinstance(node, ast.Attribute):
            value = self._eval_node(node.value)
            if isinstance(value, dict):
                return value.get(node.attr)
            return getattr(value, node.attr, None)

        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            index = self._eval_node(node.slice)
            if isinstance(value, (list, tuple)) and isinstance(index, int):
                if 0 <= index < len(value):
                    return value[index]
                return None
            if isinstance(value, dict):
                return value.get(index)
            return None

        if isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(item) for item in node.elts)

        if isinstance(node, ast.Dict):
            return {
                self._eval_node(key): self._eval_node(value)
                for key, value in zip(node.keys, node.values)
            }

        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(bool(item) for item in values)
            if isinstance(node.op, ast.Or):
                return any(bool(item) for item in values)
            raise SafeExpressionError("Unsupported boolean operator")

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise SafeExpressionError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise SafeExpressionError("Unsupported binary operator")

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                result = self._compare(op, left, right)
                if not result:
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            test = bool(self._eval_node(node.test))
            return self._eval_node(node.body if test else node.orelse)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SafeExpressionError("Only direct safe function calls are allowed")
            fn_name = node.func.id
            fn = _ALLOWED_FUNCS.get(fn_name)
            if fn is None:
                raise SafeExpressionError(f"Function '{fn_name}' is not allowed")
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords if kw.arg}
            return fn(*args, **kwargs)

        raise SafeExpressionError(f"Unsupported expression node: {type(node).__name__}")

    def _compare(self, op: ast.cmpop, left: Any, right: Any) -> bool:
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        raise SafeExpressionError(f"Unsupported comparison operator: {type(op).__name__}")


def safe_eval_expression(expression: str, context: Mapping[str, Any], default: Any = None) -> Any:
    """Evaluate a constrained expression safely; return default on failure."""
    expr = str(expression or "").strip()
    if not expr:
        return default
    try:
        evaluator = _SafeEvaluator(context)
        return evaluator.eval(expr)
    except Exception:
        return default


def safe_eval_bool(expression: str, context: Mapping[str, Any], default: bool = False) -> bool:
    """Evaluate expression and coerce to bool with a safe default fallback."""
    value = safe_eval_expression(expression, context, default=default)
    try:
        return bool(value)
    except Exception:
        return default
