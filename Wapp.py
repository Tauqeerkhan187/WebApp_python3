# Author: TK
# Date: 27-01-2026
# Desc: Main web app handles HTTP requests, validates and plots, saves user
# input and manages downloading function

import base64
from io import BytesIO
from flask import Flask, request, render_template, session, send_file, redirect

from config import *
from plotter.plotting import make_plot_png

app = Flask(__name__)
app.secret_key = "dev-only-secret"  # required for session storage


@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults (also used to preserve input)
    exprs = DEFAULT_EXPRS
    a = DEFAULT_A
    b = DEFAULT_B
    color = DEFAULT_COLOR

    img_data = None
    error = None
    warnings = None
    exprs_display = []
    a2 = b2 = None

    if request.method == "POST":
        exprs = request.form.get("exprs", DEFAULT_EXPRS)
        a = request.form.get("a", DEFAULT_A)
        b = request.form.get("b", DEFAULT_B)
        color = request.form.get("color", DEFAULT_COLOR)

        try:
            png, exprs_display, (a2, b2), warns = make_plot_png(
                expr_text=exprs,
                a=float(a),
                b=float(b),
                color=color,
                n_points=N_POINTS,
                max_exprs=MAX_EXPRS,
            )

            img_data = base64.b64encode(png).decode("ascii")
            warnings = "\n".join(warns) if warns else None

            # Store last valid plot for download
            session["last_plot"] = {
                "exprs": exprs,
                "a": a2,
                "b": b2,
                "color": color,
            }

        except Exception as e:
            error = str(e)

@app.route("/", methods=["GET", "POST"])
def index():
    # Defaults (also used to preserve input)
    exprs = DEFAULT_EXPRS
    a = DEFAULT_A
    b = DEFAULT_B
    color = DEFAULT_COLOR

    img_data = None
    error = None
    warnings = None
    exprs_display = []
    a2 = b2 = None

    if request.method == "POST":
        exprs = request.form.get("exprs", DEFAULT_EXPRS)
        a = request.form.get("a", DEFAULT_A)
        b = request.form.get("b", DEFAULT_B)
        color = request.form.get("color", DEFAULT_COLOR)

        try:
            png, exprs_display, (a2, b2), warns = make_plot_png(
                expr_text=exprs,
                a=float(a),
                b=float(b),
                color=color,
                n_points=N_POINTS,
                max_exprs=MAX_EXPRS,
            )

            img_data = base64.b64encode(png).decode("ascii")
            warnings = "\n".join(warns) if warns else None

            # Store last valid plot for download
            session["last_plot"] = {
                "exprs": exprs,
                "a": a2,
                "b": b2,
                "color": color,
            }

        except Exception as e:
            error = str(e)

@app.route("/download")
def download():
    """
    Recreates the last successful plot and sends it as PNG.
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
