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

LR_cross_val_stats = pd.read_pickle(MODEL_PATH + 'cross_val_stats.pkl')
main_data = pd.read_pickle(MODEL_PATH + 'main_data.pkl')
folds = len(LR_cross_val_stats["classifiers"])
target_names = ['Non-Sensitive', 'Sensitive']
vis_options = ["LIME", "ELI5"]
clf_options = ["LR", "RF", "SVM"]


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


def get_visualisation() -> str:
    user_id = g.user['id']
    db = get_db()

    visual = db.execute(
        'SELECT visualisation_method FROM user WHERE id = ?', (user_id,)
    ).fetchone()[0]

    return visual


def get_clf() -> str:
    user_id = g.user['id']
    db = get_db()

    clf = db.execute(
        'SELECT clf_method FROM user WHERE id = ?', (user_id,)
    ).fetchone()[0]

    return clf


def change_visual(visual: str) -> str:
    user_id = g.user['id']
    db = get_db()

    db.execute(
        'UPDATE user SET visualisation_method = ?'
        ' WHERE id = ?',
        (visual, user_id)
    )

    db.commit()

    return visual


def change_clf(clf: str) -> str:
    user_id = g.user['id']
    db = get_db()

    db.execute(
        'UPDATE user SET clf_method = ?'
        ' WHERE id = ?',
        (clf, user_id)
    )

    db.commit()

    return clf


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


def get_specific_sens(sens: int, cross_val_stats: dict) -> pd:
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


def get_clf_stats(clf: str) -> dict:
    if clf == 'LR':
        return LR_cross_val_stats
    elif clf == 'RF':
        return LR_cross_val_stats
    elif clf == 'SVM':
        return LR_cross_val_stats


def explainers(document_index: int, test_data: pd, test_labels: pd, extra_indexs: list, visual: str, cross_val_stats: dict) -> LimeTextExplainer:
    index = 0
    fold_length = len(test_data[0])

    # find which cross validation index to choose from
    while document_index > fold_length * (index+1) + extra_indexs[index] - 1:
        index += 1

    document_index -= fold_length * index + extra_indexs[index]

    test_data = test_data[index]
    vectorizer = cross_val_stats["vectorizers"][index]
    model = cross_val_stats["classifiers"][index]

    def lime_explain(text=True):
        def predictor(texts):
            feature = vectorizer.transform(texts)
            pred = model.predict_proba(feature)
            return pred

        lime_explainer = LimeTextExplainer(
            class_names=target_names)

        lime_data = test_data.iloc[document_index][0:500]

        lime_values = lime_explainer.explain_instance(
            lime_data,
            classifier_fn=predictor,
        )

        if text:
            return lime_values.as_html(predict_proba=False, specific_predict_proba=False)
        else:
            return lime_values.as_html(text=False)

    def shap_explain():
        shap_values = cross_val_stats["shap_values"][index]

        force_plot = shap.plots.force(
            shap_values[document_index], matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return shap_html

    def eli5_explain():
        eli5_html = eli5.show_prediction(
            model, test_data.iloc[document_index], vec=vectorizer, target_names=target_names)
        return eli5_html

    shap_html = shap_explain()

    vis_html = None
    lime_html = lime_explain(text=False)
    if visual == 'ELI5':
        vis_html = eli5_explain()

    elif visual == 'LIME':
        vis_html = lime_explain()

    test_labels = test_labels[index]
    isSensitive = (
        "Sensitive" if test_labels.iloc[document_index] else "Non-Sensitive")

    return shap_html, lime_html, vis_html, isSensitive


def reset_options(visual: str, clf: str) -> None:
    # ensure that chosen options shows to user
    vis_options.remove(visual)
    vis_options.append(visual)

    clf_options.remove(clf)
    clf_options.append(clf)


def get_visual_html(sensitivity: int, document_number: int, visual: str, clf: str) -> LimeTextExplainer:

    cross_val_stats = get_clf_stats(clf)

    test_data, test_labels, extra_indexs = get_specific_sens(
        sensitivity, cross_val_stats)

    shap_html, lime_probas_html, visual_html, isSensitive = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return shap_html, lime_probas_html, visual_html


@bp.route('/')
def classifier_main_page():
    return render_template('classifier/index.html')


@bp.route('/sensitive-info', methods=('GET', 'POST'))
@login_required
def sensitive_info():
    sensitivity = 1

    document_number = get_doc_num(sensitivity)

    max_documents = main_data["labels"].value_counts()[sensitivity]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_options')

        if chosen_vis == None:
            document_number = change_doc(
                document_number, max_documents, sensitivity)
        else:
            visual = change_visual(chosen_vis)
            chosen_clf = request.form.get('clf_options')
            clf = change_clf(chosen_clf)

    reset_options(visual, clf)

    shap_html, lime_probas_html, visual_html = get_visual_html(
        sensitivity, document_number, visual, clf)

    return render_template('classifier/sensitive_info.html', document_number=document_number+1, max_documents=max_documents,
                           vis_options=vis_options, visual_html=visual_html, class_options=clf_options,
                           shap_html=shap_html, lime_probas_html=lime_probas_html)


@bp.route('/non-sensitive-info', methods=('GET', 'POST'))
@login_required
def non_sensitive_info():
    sensitivity = 0

    document_number = get_doc_num(sensitivity)

    max_documents = main_data["labels"].value_counts()[sensitivity]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_options')

        if chosen_vis == None:
            document_number = change_doc(
                document_number, max_documents, sensitivity)
        else:
            visual = change_visual(chosen_vis)
            chosen_clf = request.form.get('clf_options')
            clf = change_clf(chosen_clf)

    reset_options(visual, clf)

    shap_html, lime_probas_html, visual_html = get_visual_html(
        sensitivity, document_number, visual, clf)

    return render_template('classifier/non_sensitive_info.html', document_number=document_number+1, max_documents=max_documents,
                           vis_options=vis_options, visual_html=visual_html, class_options=clf_options, shap_html=shap_html,
                           lime_probas_html=lime_probas_html)


@bp.route('/single-document-sensitivity-info', methods=('GET', 'POST'))
@login_required
def single_document_sensitivity_info():

    document_number = get_doc_num()

    max_documents = len(main_data['labels'])

    extra_indexs = [0 for _ in range(folds)]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_options')

        if chosen_vis == None:
            document_number = change_doc(document_number, max_documents)
        else:
            visual = change_visual(chosen_vis)
            chosen_clf = request.form.get('clf_options')
            clf = change_clf(chosen_clf)

    reset_options(visual, clf)

    cross_val_stats = get_clf_stats(clf)

    test_data = cross_val_stats["test_features_list"]
    test_labels = cross_val_stats["test_labels_list"]

    shap_html, lime_probas_html, visual_html, isSensitive = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return render_template('classifier/single_document_sensitivity_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, vis_options=vis_options,
                           visual_html=visual_html, class_options=clf_options, lime_probas_html=lime_probas_html,
                           shap_html=shap_html)


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
