from flask import Blueprint, render_template

# Blueprint Configuration
more_bp = Blueprint(
    "more_bp", __name__
)


@more_bp.route("/more", methods=["GET"])
def page():

    return render_template(
        "more.jinja2.html"
    )