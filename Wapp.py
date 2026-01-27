# Author: TK
# Date: 27-01-2026
# Desc: Main web app handles HTTP requests, validates and plots, saves user
# input and manages download function

import base64
from io import BytesIO

from flask import Flask, request, render_template, session, send_file, redirect

from config import (
    COLORS, DEFAULT_EXPRS, DEFAULT_A, DEFAULT_B, DEFAULT_COLOR,
    N_POINTS, MAX_EXPRS
)
from plotter.plotting import make_plot_png
from plotter.llm_mistral import nl_to_exprs


app = Flask(__name__)
app.secret_key = "dev-only-secret"  # required for session storage


@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults (also used to preserve input after POST)
    request_text = "Please plot a sine wave function!"
    exprs = ""
    a = DEFAULT_A
    b = DEFAULT_B
    color = DEFAULT_COLOR


    img_data = None
    error = None
    warnings = None
    exprs_display = []
    a2 = b2 = None

    if request.method == "POST":
        # Read user form input
        request_text = request.form.get("request", "").strip()
        exprs = nl_to_exprs(request_text)
        a = request.form.get("a", DEFAULT_A)
        b = request.form.get("b", DEFAULT_B)
        color = request.form.get("color", DEFAULT_COLOR)

        # validate color against predefined options
        if color not in COLORS:
            color = DEFAULT_COLOR

        try:
            # LLM convert natural lang to expr
            exprs = nl_to_exprs(request_text)

            # Build plot bytes from validated inputs (still safe)
            png, exprs_display, (a2, b2), warns = make_plot_png(
                expr_text=exprs,
                a=float(a),
                b=float(b),
                color=color,
                n_points=N_POINTS,
                max_exprs=MAX_EXPRS,
            )

            # Convert to base64 so we can embed directly in HTML
            img_data = base64.b64encode(png).decode("ascii")

            # Convert list of warnings into a single string for display
            warnings = "\n".join(warns) if warns else None

            # Store last valid plot inputs so /download can regenerate it
            session["last_plot"] = {
                "exprs": exprs,
                "a": a2,
                "b": b2,
                "color": color,
            }

        except Exception as e:
            # Friendly error message shown on the page
            error = str(e)

    # Enable download button only if a successful plot was generated before
    can_download = "last_plot" in session

    # IMPORTANT: Flask views must return a response
    return render_template(
        "index.html",
        colors=COLORS,
        request_text=request_text, # for new UI
        exprs=exprs,
        a=a,
        b=b,
        color=color,
        img_data=img_data,
        error=error,
        warnings=warnings,
        exprs_display=exprs_display,
        a2=a2,
        b2=b2,
        can_download=can_download,
    )


@app.route("/download", methods=["GET"])
def download():
    """
    Recreates the last successful plot and sends it as a PNG file download.
    We regenerate the image (instead of storing the bytes in session)
    to keep session small and simple.
    """
    last = session.get("last_plot")
    if not last:
        return redirect("/")

    png, _, _, _ = make_plot_png(
        expr_text=last["exprs"],
        a=float(last["a"]),
        b=float(last["b"]),
        color=last["color"],
        n_points=N_POINTS,
        max_exprs=MAX_EXPRS,
    )

    return send_file(
        BytesIO(png),
        mimetype="image/png",
        as_attachment=True,
        download_name="plot.png",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

