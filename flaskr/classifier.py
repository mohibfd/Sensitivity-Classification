from __future__ import print_function  # In python 2.7
from flask import (
    Blueprint, flash, g, render_template, request
)

from flaskr.auth import login_required
from flaskr.db import get_db

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from lime.lime_text import LimeTextExplainer
import numpy as np
import shap as shap
import eli5 as eli5

bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model/")

cross_val_stats = pd.read_pickle(MODEL_PATH + 'cross_val_stats.pkl')
main_data = pd.read_pickle(MODEL_PATH + 'main_data.pkl')
folds = len(cross_val_stats["classifiers"])
target_names = ['Non-Sensitive', 'Sensitive']


def get_doc_num(database="") -> int:
    user_id = g.user['id']
    db = get_db()
    document_number = 0

    if database == 0:
        document_number = db.execute(
            'SELECT non_sens_document_number FROM user WHERE id = ?', (
                user_id,)
        ).fetchone()[0]
    elif database == 1:
        document_number = db.execute(
            'SELECT sens_document_number FROM user WHERE id = ?', (user_id,)
        ).fetchone()[0]
    else:
        document_number = db.execute(
            'SELECT document_number FROM user WHERE id = ?', (user_id,)
        ).fetchone()[0]

    return document_number


def change_doc(document_number: int, max_documents: int, database="") -> int:
    user_id = g.user['id']
    db = get_db()

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

    if database == 0:
        db.execute(
            'UPDATE user SET non_sens_document_number = ?'
            ' WHERE id = ?',
            (document_number, user_id)
        )
    elif database == 1:
        db.execute(
            'UPDATE user SET sens_document_number = ?'
            ' WHERE id = ?',
            (document_number, user_id)
        )
    else:
        db.execute(
            'UPDATE user SET document_number = ?'
            ' WHERE id = ?',
            (document_number, user_id)
        )

    db.commit()

    return document_number


def get_specific_sens(sens: int) -> pd:
    test_data = cross_val_stats["test_features_list"].copy()
    test_labels = cross_val_stats["test_labels_list"].copy()

    extra_indexs = [0 for _ in range(folds)]
    indexs_counter = 0

    average_length = int(main_data["labels"].value_counts()[sens]/folds)
    for i in range(len(test_data)):
        data = {'Body': test_data[i], 'Sensitive': test_labels[i]}
        test_df = pd.DataFrame(data)

        sensitive_df = test_df.loc[test_df['Sensitive'] == sens]

        if len(sensitive_df) > average_length:
            indexs_counter += 1
            extra_indexs[i] = indexs_counter

        test_data[i] = sensitive_df['Body']
        test_labels[i] = sensitive_df['Sensitive']

    return test_data, test_labels, extra_indexs


def explainers(document_index: int, test_data: pd, test_labels: pd, extra_indexs: list) -> LimeTextExplainer:
    index = 0
    fold_length = len(test_data[0])

    # find which cross validation index to choose from
    while document_index > fold_length * (index+1) + extra_indexs[index] - 1:
        index += 1

    document_index -= fold_length * index + extra_indexs[index]

    test_data = test_data[index]
    vectorizer = cross_val_stats["vectorizers"][index]
    model = cross_val_stats["classifiers"][index]

    def lime_explain():
        def predictor(texts):
            feature = vectorizer.transform(texts)
            pred = model.predict_proba(feature)
            return pred

        lime_explainer = LimeTextExplainer(
            class_names=target_names)

        lime_data = test_data.iloc[document_index][0:200]

        lime_values = lime_explainer.explain_instance(
            lime_data,
            classifier_fn=predictor,
        )
        return lime_values.as_html()

    def shap_explain():
        train_data = cross_val_stats["train_features_list"][index]
        train_features = vectorizer.transform(train_data).toarray()
        test_features = vectorizer.transform(test_data).toarray()

        shap_explainer = shap.Explainer(
            model, train_features, feature_names=vectorizer.get_feature_names())
        shap_values = shap_explainer(test_features)

        force_plot = shap.plots.force(
            shap_values[document_index], matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return shap_html

    def eli5_explain():
        eli5_html = eli5.show_prediction(
            model, test_data.iloc[document_index], vec=vectorizer, target_names=target_names)
        return eli5_html

    lime_html = lime_explain()
    shap_plot = shap_explain()
    eli5_html = eli5_explain()

    test_labels = test_labels[index]
    isSensitive = (
        "Sensitive" if test_labels.iloc[document_index] else "Non-Sensitive")

    return lime_html, shap_plot, eli5_html, isSensitive


@bp.route('/')
def classifier_main_page():
    return render_template('classifier/index.html')


@bp.route('/sensitive-info', methods=('GET', 'POST'))
@login_required
def sensitive_info():
    sensitivity = 1

    test_data, test_labels, extra_indexs = get_specific_sens(sensitivity)

    document_number = get_doc_num(sensitivity)
    # document_number = 490

    max_documents = main_data["labels"].value_counts()[sensitivity]

    if request.method == 'POST':
        document_number = change_doc(
            document_number, max_documents, sensitivity)

    lime_html, shap_plot, eli5_html, isSensitive = explainers(
        document_number, test_data, test_labels, extra_indexs)

    return render_template('classifier/sensitive_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, lime_html=lime_html, shap_plot=shap_plot,
                           eli5_html=eli5_html)


@bp.route('/non-sensitive-info', methods=('GET', 'POST'))
@login_required
def non_sensitive_info():
    sensitivity = 0

    test_data, test_labels, extra_indexs = get_specific_sens(sensitivity)

    document_number = get_doc_num(sensitivity)

    max_documents = main_data["labels"].value_counts()[sensitivity]

    if request.method == 'POST':
        document_number = change_doc(
            document_number, max_documents, sensitivity)

    lime_html, shap_plot, eli5_html, isSensitive = explainers(
        document_number, test_data, test_labels, extra_indexs)

    return render_template('classifier/non_sensitive_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, lime_html=lime_html, shap_plot=shap_plot,
                           eli5_html=eli5_html)


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

    vec = main_data['vectorizer']
    model.fit(X, y)
    eli5_general = eli5.show_weights(model, vec=vec, top=10,
                                     target_names=target_names)

    return render_template('classifier/general_sensitivity_info.html', predictions=predictions, confusion_matrix_score=conf_mat,
                           eli5_general=eli5_general)


@bp.route('/single-document-sensitivity-info', methods=('GET', 'POST'))
@login_required
def single_document_sensitivity_info():
    document_number = get_doc_num()

    max_documents = len(main_data['labels'])

    extra_indexs = [0 for _ in range(folds)]

    if request.method == 'POST':
        document_number = change_doc(document_number, max_documents)

    test_data = cross_val_stats["test_features_list"]
    test_labels = cross_val_stats["test_labels_list"]

    lime_html, shap_plot, eli5_html, isSensitive = explainers(
        document_number, test_data, test_labels, extra_indexs)

    return render_template('classifier/single_document_sensitivity_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, lime_html=lime_html, shap_plot=shap_plot,
                           eli5_html=eli5_html)
