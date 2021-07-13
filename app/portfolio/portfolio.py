from flask import Blueprint, render_template

# Blueprint Configuration
portfolio_bp = Blueprint(
    "portfolio_bp", __name__
)


@portfolio_bp.route("/portfolio", methods=["GET"])
def page():

    return render_template(
        "portfolio.jinja2.html"
    )