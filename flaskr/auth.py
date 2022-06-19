import functools

from flask import(
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from werkzeug.security import check_password_hash, generate_password_hash

# from flaskr.db import get_db
from . user import User
from .extensions import db
from sqlalchemy import exc

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':

        if request.form.get('go_login'):
            return redirect(url_for("auth.login"))

        username = request.form['username']
        password = request.form['password']
        # db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            try:
                user = User(username=username, password=generate_password_hash(
                    password), document_number=0, sens_document_number=0, non_sens_document_number=0, visualisation_method='None', clf_method='LR')
                db.session.add(user)
                db.session.commit()
                # db.execute(
                #     "INSERT INTO user (username, password, document_number, sens_document_number, non_sens_document_number, visualisation_method, clf_method) VALUES (?, ?, 0, 0, 0, ?, ?)",
                #     (username, generate_password_hash(password), 'None', 'LR'),
                # )
                # db.commit()
            except exc.IntegrityError:
                error = f"USER {username} is already registered."
            else:
                session['user_id'] = user.id
                return redirect(url_for('index'))

        flash(error)
    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':

        if request.form.get('go_sign_up'):
            return redirect(url_for("auth.register"))

        username = request.form['username']
        password = request.form['password']
        # db = get_db()
        error = None
        user = User.query.filter_by(username=username).first()
        # user = db.execute(
        #     'SELECT * FROM user WHERE username = ?', (username,)
        # ).fetchone()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user.password, password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('index'))

        flash(error)

    return render_template('auth/login.html')


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = User.query.filter_by(id=user_id).first()


@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view
