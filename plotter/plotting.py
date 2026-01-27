# Author: TK
# Date: 27-01-2026
# Desc: Parses multiple expressions, validates the interval, generates plot, return png bytes

from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use("Agg") # required for server rendering
import matplotlib.pyplot as plt

from .safe_eval import compile_expr, eval_on_x

def parse_expressions(text: str, max_exprs: int):
    """
    Parse one expr per line
    ignore empty line and comments.
    """
    exprs = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startwith("#"):
            continue
        exprs.append(s)

    if not exprs:
        raise ValueError("Please enter at least one expression...")

    if len(exprs) > max_exprs:
        raise ValueError(f"Too many expressions (max: {max_exprs}).")

    return exprs

def validate_interval(a: float, b: float):
    """
    Ensure a valid plotting interval.
    Automatically swaps if a > b.
    """
    if not np.isfinite(a) or not np.isfinite(b):

        raise ValueError("a and b must be finite numbers")

    if a == b:

        raise ValueError("a and b cannot be equal")

    return (a, b) if a < b else (b, a)

def make_plot_png(expr_text, a, b, color, n_points, max_exprs):
    """
    core plotting func which returns png bytes, list of exprs, interval,
    warnings.
    """

    a, b = validate_interval(a, b)
    exprs = parse_expressions(exprs_text, max_exprs)

    compiled = []
    labels = []
    errors = []

    # compile expr one by one for precise reports
    for i, expr in enumerate(exprs, start = 1):
        try:
            c, disp = compile_expr(expr)
            compiled.append(c)
            labels.append(disp)

        except Exception as e:
            errors.append(f"Line {i}: {expr}\n -> {e}")

    if errors:
        raise ValueError("Expression errors:\n" + "\n".join(errors))

    x = np.linspace(a, b, n_points)

    fig, ax = plt.subplots()
    warnings = []

    for c, label in zip(compiled, labels):
        y = eval_on_x(c, x)

        if not np.all(np.isfinite(y)):
            warnings.append(f"Non-finite values in: {label}")

        ax.plot(x, y, color=color, label=label)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if len(labels) > 1:
        ax.legend(fontsize=9)

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)

    buf.seek(0)
    return buf.read(), labels, (a, b), warnings
