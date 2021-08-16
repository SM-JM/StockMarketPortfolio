"""Initialize Flask app."""
from flask import Flask

def create_app():
    """Construct the core application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")

    with app.app_context():
        
        # Import various modules of the application
        from .analysis        import analysis
        
        # Register all the blueprints
        app.register_blueprint(analysis.analysis_bp)

        return app