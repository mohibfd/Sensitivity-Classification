from __future__ import print_function  # In python 2.7
from flask import (
    Blueprint, flash, g, render_template, request
)

from flaskr.auth import login_required
from flaskr.db import get_db

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import numpy as np

bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model/")

cross_val_stats = pd.read_pickle(MODEL_PATH + 'cross_val_stats.pkl')
main_data = pd.read_pickle(MODEL_PATH + 'main_data.pkl')


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
    X = main_data['features']
    y = main_data['labels']
    model = main_data['classifier']

    scores = cross_val_score(model, X, y, cv=5, scoring="f1_micro")
    prediction = "f1: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores))

    y_pred = cross_val_predict(model, X, y, cv=5)
    confusion_matrix_score = confusion_matrix(y, y_pred)

    return render_template('classifier/general_sensitivity_info.html', prediction=prediction, confusion_matrix_score=confusion_matrix_score)


def explainer(document_number) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""
    index = 0
    fold1_position = len(cross_val_stats["test_labels_list"][0])
    fold2_position = len(
        cross_val_stats["test_labels_list"][1]) + fold1_position
    fold3_position = len(
        cross_val_stats["test_labels_list"][2]) + fold2_position
    fold4_position = len(
        cross_val_stats["test_labels_list"][3]) + fold3_position

    if document_number < fold1_position:
        index = 0
    elif document_number < fold2_position:
        index = 1
        document_number = document_number - fold1_position
    elif document_number < fold3_position:
        index = 2
        document_number = document_number - fold2_position
    elif document_number < fold4_position:
        index = 3
        document_number = document_number - fold3_position
    else:
        index = 4
        document_number = document_number - fold4_position

    text_data = cross_val_stats["test_features_list"][index]
    one_hot_vectorizer = cross_val_stats["vectorizers"][index]
    model = cross_val_stats["classifiers"][index]

    def predictor(texts):
        feature = one_hot_vectorizer.transform(texts)
        pred = model.predict_proba(feature)
        return pred

    explainer = LimeTextExplainer(class_names=['Non-Sensitive', 'Sensitive'])

    first_idx = document_number
    second_idx = first_idx + 1

    specific_data = text_data[first_idx:second_idx].iloc[0]
    specific_data = specific_data[0:300]

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        specific_data,
        classifier_fn=predictor,
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

        if request.form['submit_button'] == 'Previous Document':
            if (document_number == 0):
                flash("There are no previous documents")
            else:
                document_number -= 1
        elif request.form['submit_button'] == 'Next Document':
            if (document_number == 3800):
                flash("There are no more documents")
            else:
                document_number += 1
        db.execute(
            'UPDATE user SET document_number = ?'
            ' WHERE id = ?',
            (document_number, user_id)
        )
        db.commit()

    exp = explainer(document_number)
    exp = exp.as_html()

    return render_template('classifier/single_document_sensitivity_info.html', exp=exp, document_number=document_number+1)
