from __future__ import print_function  # In python 2.7
import scipy.stats
import sklearn.pipeline
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import fbeta_score
import os
from sklearn.metrics import confusion_matrix
import lime
import sklearn.ensemble
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from pathlib import Path
import sys
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from . import tokenize_normalize

bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_ROOT, "model/")
sensitive_data_df = pd.read_pickle(MODEL_PATH + 'sensitive_data.pkl')


shuffled_sensitive_data_df = sensitive_data_df.reindex(
    np.random.permutation(sensitive_data_df.index))

# 1. Split the data 80/20 train/test
train_split = int(len(shuffled_sensitive_data_df) * 0.8)
tmp_train = shuffled_sensitive_data_df.iloc[:train_split, :]
test_data = shuffled_sensitive_data_df.iloc[train_split:, :]

# 2. Split the train data into a train/validation split that's 80% train, 20% developemnt
validation_split = int(train_split * 0.8)
train_data = tmp_train.iloc[:validation_split, :]
validation_data = tmp_train.iloc[validation_split:, :]

# putting sensitivity column into a variable
train_labels = train_data['Sensitive']
validation_labels = validation_data['Sensitive']
test_labels = test_data['Sensitive']

# train_features = pickle.load(
#     open(MODEL_PATH + 'train_features.sav', 'rb'))
# validation_features = pickle.load(
#     open(MODEL_PATH + 'validation_features.sav', 'rb'))
test_features = pickle.load(
    open(MODEL_PATH + 'test_features.sav', 'rb'))

# load the model from disk
lr_model = pickle.load(open(MODEL_PATH + 'one_hot_lr_model.sav', 'rb'))
lr_predict = lr_model.predict(test_features)


@bp.route('/classifier-main-page')
def classifier_main_page():
    prediction = "%.3f" % (fbeta_score(
        lr_predict, test_labels, beta=1, average="macro"))

    # Note the order here is true, predicted
    confusion_matrix_score = confusion_matrix(test_labels, lr_predict)

    return render_template('classifier/classifier_main_page.html', prediction=prediction, confusion_matrix_score=confusion_matrix_score)


@bp.route('/sensitive-info')
def sensitive_info():
    return render_template('classifier/sensitive_info.html')


@bp.route('/non-sensitive-info')
def non_sensitive_info():
    return render_template('classifier/non_sensitive_info.html')


@bp.route('/general-sensitivity-info')
def general_sensitivity_info():
    return render_template('classifier/general_sensitivity_info.html')


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
                ('clf', LogisticRegression(
                    max_iter=10000,
                )),
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
            return probs
        else:
            probs = classifier.predict(texts)
            return probs


def explainer() -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""

    model = LogisticExplainer()
    predictor = model.predict

    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        class_names=['Non-Sensitive', 'Sensitive']
    )

    data = test_data['Body']

    first_idx = 2
    second_idx = first_idx + 1

    specific_data = data[first_idx:second_idx].iloc[0]
    specific_data = specific_data[0:100]

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        specific_data,
        classifier_fn=predictor,
        top_labels=1,
        num_features=4,
    )
    return exp


@bp.route('/single-document-sensitivity-info')
def single_document_sensitivity_info():
    exp = explainer()

    exp = exp.as_html()

    from sklearn import metrics

    model = LogisticExplainer()
    lr_predict2 = model.predict(test_data['Body'], probability=False)
    prediction2 = [len(test_data), len(test_labels)]
    # prediction2 = (metrics.classification_report(
    #     list(test_labels), lr_predict))

    from sklearn.metrics import f1_score

    prediction2 = "%.3f" % (f1_score(
        lr_predict2, test_labels, average="macro"))

    confusion_matrix_score = confusion_matrix(test_labels, lr_predict2)

    return render_template('classifier/single_document_sensitivity_info.html', exp=exp, prediction=prediction2, confusion_matrix_score=confusion_matrix_score)
