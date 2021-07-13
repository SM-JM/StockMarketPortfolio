"""Initialize Flask app."""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

import pymysql
pymysql.install_as_MySQLdb()


db = SQLAlchemy()


def create_app():
    """Construct the core application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")

    db.init_app(app)

    with app.app_context():
        
        # Import various modules of the application
        from .home            import home
        from .analysis        import analysis
        from .portfolio       import portfolio
        from .news             import news
        from .more             import more
        
        # Register all the blueprints
        app.register_blueprint(home.home_bp)
        app.register_blueprint(analysis.analysis_bp)
        app.register_blueprint(portfolio.portfolio_bp)
        app.register_blueprint(news.news_bp)
        app.register_blueprint(more.more_bp)

        db.create_all()  # Create database tables for our data models

        return app