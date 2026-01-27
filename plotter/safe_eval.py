# Author: TK
# Date: 27-01-2026
# Desc: File responsible for validating math exprs, prevent malicious code exec.
# ensuring expressions are numbers and depends on x.

import ast
import numpy as np

class Math_func:
    """
    Provide function like maths that work with numpy arrays.
    allows users to write familiar math exprs while still supporting
    vectorized evaluation.
    """
    def __getattr__(self, name: str):
        if hasattr(np, name):
            return getattr(np, name)

        if name == "pi":
            return np.pi
        if name == "e":
            return np.e

        # Other input is rejected
        raise AttributeError(f"math.{name} is not allowed")

# Exposed namespace during eval()
ALLOWED_NAMES = {
    "x": None,
    "np": np,
    "math": Math_func(),
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Call,
    ast.Name, ast.Load,
    ast.Constant,
    ast.Attribute,

    #Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)

class SafeExpr(ast.NodeVisitor):
    """
    Walks the parsed syntax tree and rejects anything unsafe.

    Also checks that the expression actually depends on x.
    """

    def __init__(self):

        self.used_x = False

    def visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        return super().visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in ALLOWED_NAMES:
            raise ValueError(f"Unknown name: {node.id}")

        if node.id == "x":
            self.used_x = True

    def visit_Attribute(self, node: ast.Attribute):
        # Prevent access to __dict__, __class__, etc.
        if node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")

        # Only allow math.<fn> or np.<fn>
        if not isinstance(node.value, ast.Name) or node.value.id not in ("np", "math"):
            raise ValueError("Only np.<fn> or math.<fn> allowed")

def normalize_expr(expr: str) -> str:
    """
    Convert caret exponent to Python exponent.

    Example:
    x^2 -> x**2
    """
    return expr.replace("^", "**").strip()


def compile_expr(expr: str):
    """
    Parse and validate an expression.
    Returns compiled code object and normalized string.
    """
    expr = normalize_expr(expr)

    if not expr:
        raise ValueError("Expression is empty")

    tree = ast.parse(expr, mode="eval")
    checker = SafeExpr()
    checker.visit(tree)

    if not checker.used_x:
        raise ValueError("Expression must contain x")

    return compile(tree, "<expr>", "eval"), expr


def eval_on_x(compiled, x: np.ndarray):
    """
    Evaluate the compiled expression safely over x values.
    """
    env = dict(ALLOWED_NAMES)
    env["x"] = x

    y = eval(compiled, {"__builtins__": {}}, env)
    return np.asarray(y, dtype=float)

