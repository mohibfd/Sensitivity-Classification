from . import classifier
from . import auth
from . import db
from flask import Flask
import os

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"


# create and configure the app
app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

db.init_app(app)

app.register_blueprint(auth.bp)

app.register_blueprint(classifier.bp)
app.add_url_rule('/', endpoint='index')
