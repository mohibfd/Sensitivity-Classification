from __future__ import print_function  # In python 2.7
from flask import (
    Blueprint, flash, g, render_template, request
)

from flaskr.auth import login_required
from flaskr.db import get_db

import os
import sys
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from lime.lime_text import LimeTextExplainer

bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model/")

train_data = pd.read_pickle(MODEL_PATH + 'train_data.pkl')
test_data = pd.read_pickle(MODEL_PATH + 'test_data.pkl')

train_labels = train_data['Sensitive']
test_labels = test_data['Sensitive']


@bp.route('/')
def classifier_main_page():
    return render_template('classifier/index.html')


@bp.route('/sensitive-info')
@login_required
def sensitive_info():
    return render_template('classifier/sensitive_info.html')


@bp.route('/non-sensitive-info')
@login_required
def non_sensitive_info():
    return render_template('classifier/non_sensitive_info.html')


@bp.route('/general-sensitivity-info')
@login_required
def general_sensitivity_info():

    model = LogisticExplainer()
    lr_predict = model.predict(test_data['Body'], probability=False)

    prediction = "%.3f" % (f1_score(
        lr_predict, test_labels, average="macro"))

    confusion_matrix_score = confusion_matrix(test_labels, lr_predict)

    return render_template('classifier/general_sensitivity_info.html', prediction=prediction, confusion_matrix_score=confusion_matrix_score)


class LogisticExplainer:
    """Class to explain classification results of a scikit-learn
       Logistic Regression Pipeline. The model is trained within this class.
    """

    def __init__(self) -> None:
        "Input training data path for training Logistic Regression classifier"
        import pandas as pd
        # Read in training data set
        self.train_df = train_data
        self.train_df['truth'] = train_labels
        # Categorical data type for truth labels
        self.train_df['truth'] = self.train_df['truth'].astype(
            int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        "Create sklearn logistic regression model pipeline"
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer(binary=True)),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
            ]
        )
        # Train model
        classifier = pipeline.fit(
            self.train_df['Body'], self.train_df['truth'])
        return classifier

    def predict(self, texts, probability=True) -> np.array([float, ...]):
        """Generate an array of predicted scores (probabilities) from sklearn
        Logistic Regression Pipeline."""
        classifier = self.train()

        if probability:
            probs = classifier.predict_proba(texts)
        else:
            probs = classifier.predict(texts)
        return probs


def explainer(document_number) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""

    model = LogisticExplainer()
    predictor = model.predict

    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        class_names=['Non-Sensitive', 'Sensitive']
    )

    data = test_data['Body']

    first_idx = document_number
    second_idx = first_idx + 1

    specific_data = data[first_idx:second_idx].iloc[0]
    specific_data = specific_data[0:300]

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        specific_data,
        classifier_fn=predictor,
        top_labels=1,
        num_features=4,
    )
    return exp


@bp.route('/single-document-sensitivity-info', methods=('GET', 'POST'))
@login_required
def single_document_sensitivity_info():
    user_id = g.user['id']
    db = get_db()
    document_number = db.execute(
        'SELECT document_number FROM user WHERE id = ?', (user_id,)
    ).fetchone()
    db.commit()

    document_number = document_number[0]

    if request.method == 'POST':

        print('Hello world!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', file=sys.stderr)
        print(document_number, file=sys.stderr)

        if request.form['submit_button'] == 'Previous Document':
            if (document_number == 0):
                flash("There are no previous documents")
            else:
                document_number -= 1
        elif request.form['submit_button'] == 'Next Document':
            document_number += 1
        db.execute(
            'UPDATE user SET document_number = ?'
            ' WHERE id = ?',
            (document_number, user_id)
        )
        db.commit()

    exp = explainer(document_number)
    exp = exp.as_html()

    return render_template('classifier/single_document_sensitivity_info.html', exp=exp)
