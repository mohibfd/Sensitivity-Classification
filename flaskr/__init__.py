import os
from flask import Flask

from .extensions import db


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import classifier
    app.register_blueprint(classifier.bp)
    app.add_url_rule('/', endpoint='index')

    return app
