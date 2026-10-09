"""Evaluate an arithmetic expression"""
import ast
import operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def _value(node):
    """Evaluate one node of the parsed expression; anything else is rejected."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_value(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, int) and abs(right) > 10_000:
            raise ValueError("integer exponents above 10000 are not allowed")
        return OPS[type(node.op)](left, right)
    raise ValueError("use only numbers, + - * / ** ( ) and scientific notation such as 1.6e-19")


def calc(expr: str) -> str:
    """Safely evaluate a math expression"""
    try:
        return str(_value(ast.parse(expr, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"
