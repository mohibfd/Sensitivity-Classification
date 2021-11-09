import os
from flask import Flask
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


nlp = spacy.load('en_core_web_sm')

# Tokenize


def spacy_tokenize(string):
    tokens = list()
    doc = nlp(string)
    for token in doc:
        tokens.append(token)
    return tokens

# Normalize


def normalize(tokens):
    normalized_tokens = list()
    for token in tokens:
        normalized = token.text.lower().strip()
        if ((token.is_alpha or token.is_digit)):
            # removing stopwords
            if token.is_stop == False:
                normalized_tokens.append(normalized)
    return normalized_tokens

#Tokenize and normalize


def tokenize_normalize(string):
    return normalize(spacy_tokenize(string))


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    nlp = spacy.load('en_core_web_sm')

    # Tokenize

    def spacy_tokenize(string):
        tokens = list()
        doc = nlp(string)
        for token in doc:
            tokens.append(token)
        return tokens

    # Normalize

    def normalize(tokens):
        normalized_tokens = list()
        for token in tokens:
            normalized = token.text.lower().strip()
            if ((token.is_alpha or token.is_digit)):
                # removing stopwords
                if token.is_stop == False:
                    normalized_tokens.append(normalized)
        return normalized_tokens

    #Tokenize and normalize

    def tokenize_normalize(string):
        return normalize(spacy_tokenize(string))

    from . import db
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import blog
    app.register_blueprint(blog.bp)
    app.add_url_rule('/', endpoint='index')

    from . import classifier
    app.register_blueprint(classifier.bp)

    return app
