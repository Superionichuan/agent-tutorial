"""Evaluate an expression"""

ALLOWED_CHARS = set("0123456789+-*/.()eE ")  # digits, operators, scientific notation


def calc(expr: str) -> str:
    """Safely evaluate a math expression"""
    try:
        if not all(c in ALLOWED_CHARS for c in expr):
            return "Error: use only numbers, + - * / ** ( ) and scientific notation such as 1.6e-19"
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"
