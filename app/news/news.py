from flask import Blueprint, render_template

# Blueprint Configuration
news_bp = Blueprint(
    "news_bp", __name__
)


@news_bp.route("/news", methods=["GET"])
def page():

    return render_template(
        "news.jinja2.html"
    )