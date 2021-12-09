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
folds = len(cross_val_stats["classifiers"])


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

    f1_micro_scores = cross_val_score(
        model, X, y, cv=folds, scoring="f1_micro")
    f1_macro_scores = cross_val_score(
        model, X, y, cv=folds, scoring="f1_macro")
    accuracy_scores = cross_val_score(
        model, X, y, cv=folds, scoring="accuracy")
    precision_scores = cross_val_score(
        model, X, y, cv=folds, scoring="precision")

    f1_micro_prediction = "%0.2f (+/- %0.2f)" % (
        np.mean(f1_micro_scores), np.std(f1_micro_scores))
    f1_macro_prediction = "%0.2f (+/- %0.2f)" % (
        np.mean(f1_macro_scores), np.std(f1_macro_scores))
    accuracy_prediction = " %0.2f (+/- %0.2f)" % (
        np.mean(accuracy_scores), np.std(accuracy_scores))
    precision_prediction = "%0.2f (+/- %0.2f)" % (
        np.mean(precision_scores), np.std(precision_scores))

    predictions = {"f1_micro": f1_micro_prediction, "f1_macro": f1_macro_prediction,
                   "accuracy": accuracy_prediction, "precision": precision_prediction}

    y_pred = cross_val_predict(model, X, y, cv=folds)
    conf_mat = confusion_matrix(y, y_pred)

    return render_template('classifier/general_sensitivity_info.html', predictions=predictions, confusion_matrix_score=conf_mat)


def explainer(document_number) -> LimeTextExplainer:
    index = 0
    fold_length = len(cross_val_stats["test_labels_list"][0])

    for i in range(folds):
        if document_number < fold_length*(i+1):
            index = i
            document_number -= fold_length*i
            break

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
    specific_data = specific_data[0:1000]

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

    max_documents = len(main_data['labels'])

    if request.method == 'POST':

        if request.form['submit_button'] == 'Previous Document':
            if (document_number == 0):
                flash("There are no previous documents")
            else:
                document_number -= 1
        elif request.form['submit_button'] == 'Next Document':
            if (document_number == max_documents-1):
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

    return render_template('classifier/single_document_sensitivity_info.html', exp=exp, document_number=document_number+1, max_documents=max_documents)
