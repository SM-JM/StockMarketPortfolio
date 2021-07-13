from flask import request, render_template, make_response, redirect, Blueprint
from datetime import datetime as dt
from flask import current_app as app
from ..models import db, User

# Blueprint Configuration
home_bp = Blueprint(
    "home_bp", __name__
)

@app.route('/', methods=['GET'])
def user_records():
    """Create a user via query string parameters."""
    username = request.args.get('user')
    email = request.args.get('email')
    if username and email:
        existing_user = User.query.filter(
            User.username == username or User.email == email
        ).first()
        if existing_user:
            return make_response(
                f'{username} ({email}) already created!'
            )
        new_user = User(
            username=username,
            email=email,
            created=dt.now(),
            bio="In West Philadelphia born and raised, \
            on the playground is where I spent most of my days",
            admin=False
        )  # Create an instance of the User class
        db.session.add(new_user)  # Adds new User record to database
        db.session.commit()  # Commits all changes
        redirect('user_records')
    return render_template(
        'home.jinja2.html',
        users=User.query.all(),
        title="Show Users"
    )