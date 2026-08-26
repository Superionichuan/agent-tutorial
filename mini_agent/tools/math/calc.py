"""Evaluate an expression"""

ALLOWED_CHARS = set("0123456789+-*/.() ")


def calc(expr: str) -> str:
    """Safely evaluate a math expression"""
    try:
        if not all(c in ALLOWED_CHARS for c in expr):
            return "Error: unsafe expression"
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"
