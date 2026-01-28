# Author: TK
# Date: 28/01/2026
# Desc: Flask Web App

from __future__ import annotations

from flask import Flask, render_template, request, redirect, url_for, flash

from app.service import FriendForTheMomentService


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "dev-secret-change-me"  # used for flash messages

    # Create service once (holds persistence + API client + logger).
    svc = FriendForTheMomentService(store_path="store.json", log_path="app.log")

    @app.route("/", methods=["GET", "POST"])
    def index():
        """
        GET: show form + recent messages
        POST: accept nickname + message, compute recommendations, display results
        """
        if request.method == "POST":
            nickname = request.form.get("nickname", "")
            text = request.form.get("text", "")

            try:
                new_item, top3, recs = svc.add_message_and_recommend(nickname, text)
                return render_template(
                    "index.html",
                    messages=list(reversed(svc.list_messages()))[:20],
                    new_item=new_item,
                    top3=top3,
                    recs=recs,
                )
            except Exception as e:
                flash(str(e), "err")
                return redirect(url_for("index"))

        return render_template(
            "index.html",
            messages=list(reversed(svc.list_messages()))[:20],
            new_item=None,
            top3=[],
            recs=[],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

