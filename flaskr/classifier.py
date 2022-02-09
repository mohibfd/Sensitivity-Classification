from flask import (
    Blueprint, flash, g, render_template, request, Flask
)

from flaskr.auth import login_required
from flaskr.db import get_db

import os
import pandas as pd
from lime.lime_text import LimeTextExplainer
import numpy as np
import shap as shap
import eli5 as eli5


bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model/")

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER


LR_cross_val_stats = pd.read_pickle(MODEL_PATH + 'cross_val_stats.pkl')
XGB_cross_val_stats = pd.read_pickle(MODEL_PATH + 'XGB_cross_val_stats.pkl')
SVM_cross_val_stats = pd.read_pickle(MODEL_PATH + 'SVM_cross_val_stats.pkl')

LR_main_data = pd.read_pickle(MODEL_PATH + 'main_data.pkl')
XGB_main_data = pd.read_pickle(MODEL_PATH + 'XGB_main_data.pkl')

folds = len(LR_cross_val_stats["classifiers"])
target_names = ['Non-Sensitive', 'Sensitive']
vis_options = ["LIME", "ELI5"]
clf_options = ["LR", "XGB", "SVM"]


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


def change_visual(visual: str):
    user_id = g.user['id']
    db = get_db()

    db.execute(
        'UPDATE user SET visualisation_method = ?'
        ' WHERE id = ?',
        (visual, user_id)
    )

    db.commit()


def change_clf(clf: str):
    user_id = g.user['id']
    db = get_db()

    db.execute(
        'UPDATE user SET clf_method = ?'
        ' WHERE id = ?',
        (clf, user_id)
    )

    db.commit()


def change_doc(document_number: int, max_documents: int, database="") -> int:
    user_id = g.user['id']
    db = get_db()

    if request.form['submit_button'] == "Prev":
        if (document_number == 0):
            flash("There are no previous documents")
        else:
            document_number -= 1

    elif request.form['submit_button'] == 'Next':
        if (document_number == max_documents-1):
            flash("There are no more documents")
        else:
            document_number += 1

    else:
        document_number = int(request.form['submit_button'])-1

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

    average_length = int(LR_main_data["labels"].value_counts()[sens]/folds)
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
    elif clf == 'XGB':
        return XGB_cross_val_stats
    elif clf == 'SVM':
        return LR_cross_val_stats
    else:
        return XGB_cross_val_stats


def explainers(document_index: int, test_data: pd, test_labels: pd, extra_indexs: list, visual: str, cross_val_stats: dict) -> LimeTextExplainer:
    index = 0
    fold_length = len(test_data[0])

    # find which cross validation index to choose from
    while document_index > fold_length * (index+1) + extra_indexs[index] - 1:
        index += 1

    document_index -= fold_length * index + extra_indexs[index]

    test_data = test_data[index]
    specific_test = test_data.iloc[document_index]
    vectorizer = cross_val_stats["vectorizers"][index]
    model = cross_val_stats["classifiers"][index]

    def lime_explain(text=True):
        def proba_predictor(texts):
            feature = vectorizer.transform(texts)
            pred = model.predict_proba(feature)
            return pred

        lime_explainer = LimeTextExplainer(
            class_names=target_names)

        lime_data = specific_test[0:1000]
        # lime_data = specific_test

        lime_values = lime_explainer.explain_instance(
            lime_data,
            classifier_fn=proba_predictor,
        )

        if text:
            return lime_values.as_html(predict_proba=False, specific_predict_proba=False)
        else:
            return lime_values.as_html(text=False)

    def shap_explain():
        shap_values = cross_val_stats["shap_values"][index]

        force_plot = None
        if get_clf() == 'LR':
            force_plot = shap.plots.force(
                shap_values[document_index], matplotlib=False)
        else:
            explainer = shap.TreeExplainer(model)
            vec = cross_val_stats["vectorizers"][index]

            force_plot = shap.plots.force(
                explainer.expected_value, shap_values[document_index], feature_names=vec.get_feature_names(), matplotlib=False)
            # explainer.expected_value, shap_values[ind], feature_names=vec.get_feature_names_out()

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return shap_html

    def eli5_explain():
        eli5_html = eli5.show_prediction(
            model, specific_test, vec=vectorizer, target_names=target_names, top=10)
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

    def predictor(texts):
        predictions = []
        feature = vectorizer.transform(texts)
        models = {LR_cross_val_stats["classifiers"][index]: 'LR',
                  XGB_cross_val_stats["classifiers"][index]: 'XGB'}

        for model, name in models.items():
            pred = model.predict(feature)
            if pred == True:
                predictions.append({name: 'Sensitive'})
            else:
                predictions.append({name: 'Non-Sensitive'})

        return predictions

    prediction = predictor([specific_test])
    return shap_html, lime_html, vis_html, isSensitive, prediction


def get_visual_html(sensitivity: int, document_number: int, visual: str, clf: str) -> LimeTextExplainer:

    cross_val_stats = get_clf_stats(clf)

    test_data, test_labels, extra_indexs = get_specific_sens(
        sensitivity, cross_val_stats)

    shap_html, lime_probas_html, visual_html, _, prediction = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return shap_html, lime_probas_html, visual_html, prediction


@bp.route('/')
def classifier_main_page():
    return render_template('classifier/index.html')


@bp.route('/sensitive-info', methods=('GET', 'POST'))
@login_required
def sensitive_info():
    sensitivity = 1

    document_number = get_doc_num(sensitivity)

    max_documents = LR_main_data["labels"].value_counts()[sensitivity]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_option')

        chosen_clf = request.form.get('clf_option')

        if chosen_vis == None and chosen_clf == None:
            document_number = change_doc(
                document_number, max_documents, sensitivity)
        elif chosen_clf == None:
            visual = chosen_vis
            change_visual(visual)
        else:
            clf = chosen_clf
            change_clf(clf)

    shap_html, lime_probas_html, visual_html, prediction = get_visual_html(
        sensitivity, document_number, visual, clf)

    return render_template('classifier/sensitive_info.html', document_number=document_number+1, max_documents=max_documents,
                           curr_vis=visual, visual_html=visual_html, curr_clf=clf, shap_html=shap_html,
                           lime_probas_html=lime_probas_html, prediction=prediction)


@bp.route('/non-sensitive-info', methods=('GET', 'POST'))
@login_required
def non_sensitive_info():
    sensitivity = 0

    document_number = get_doc_num(sensitivity)

    max_documents = LR_main_data["labels"].value_counts()[sensitivity]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_option')

        chosen_clf = request.form.get('clf_option')

        if chosen_vis == None and chosen_clf == None:
            document_number = change_doc(
                document_number, max_documents, sensitivity)
        elif chosen_clf == None:
            visual = chosen_vis
            change_visual(visual)
        else:
            clf = chosen_clf
            change_clf(clf)

    shap_html, lime_probas_html, visual_html = get_visual_html(
        sensitivity, document_number, visual, clf)

    return render_template('classifier/non_sensitive_info.html', document_number=document_number+1, max_documents=max_documents,
                           curr_vis=visual, visual_html=visual_html, curr_clf=clf, shap_html=shap_html,
                           lime_probas_html=lime_probas_html)


@bp.route('/single-document-sensitivity-info', methods=('GET', 'POST'))
@login_required
def single_document_sensitivity_info():

    document_number = get_doc_num()

    max_documents = len(LR_main_data['labels'])

    extra_indexs = [0 for _ in range(folds)]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_option')

        chosen_clf = request.form.get('clf_option')

        if chosen_vis == None and chosen_clf == None:
            document_number = change_doc(
                document_number, max_documents)
        elif chosen_clf == None:
            visual = chosen_vis
            change_visual(visual)
        else:
            clf = chosen_clf
            change_clf(clf)

    cross_val_stats = get_clf_stats(clf)

    test_data = cross_val_stats["test_features_list"]
    test_labels = cross_val_stats["test_labels_list"]

    shap_html, lime_probas_html, visual_html, isSensitive = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return render_template('classifier/single_document_sensitivity_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, vis_options=vis_options,
                           curr_vis=visual, curr_clf=clf, lime_probas_html=lime_probas_html,
                           shap_html=shap_html, visual_html=visual_html)


@bp.route('/general-sensitivity-info', methods=('GET', 'POST'))
@login_required
def general_sensitivity_info():

    def get_shap_images():
        for i in range(1, 6):
            url = f"{clf}/shap{i}.png"
            shap_images.append(os.path.join(
                app.config['IMAGES_FOLDER'], url))

    clf = ""
    if request.method == 'POST':
        clf = request.form.get('clf_option')
        change_clf(clf)
    else:
        clf = get_clf()

    main_data = {}
    shap_images = []
    conf_mat_png = os.path.join(
        app.config['IMAGES_FOLDER'], clf+'/conf_mat.png')

    if clf == 'LR':
        main_data = LR_main_data
        get_shap_images()

    else:
        main_data = XGB_main_data
        get_shap_images()

    predictions = main_data["predictions"]

    eli5_general = main_data["eli5_general"]

    return render_template('classifier/general_sensitivity_info.html', predictions=predictions, eli5_general=eli5_general,
                           conf_mat_png=conf_mat_png, curr_clf=clf, shap_images=shap_images)
