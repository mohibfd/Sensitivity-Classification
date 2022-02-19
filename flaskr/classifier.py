import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import TweetTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import unpack
import numpy as np
# import eli5 as eli5
import pickle
import shap as shap
from lime.lime_text import LimeTextExplainer
import pandas as pd
import os
from tensorflow.keras.preprocessing import sequence
from flaskr.db import get_db
from flaskr.auth import login_required
from flask import (
    Blueprint, flash, g, render_template, request, Flask
)
tf.compat.v1.disable_v2_behavior()

bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model/")

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER


LR_cross_val_stats = pd.read_pickle(MODEL_PATH + 'LR_cross_val_stats.pkl')
XGB_cross_val_stats = pd.read_pickle(MODEL_PATH + 'XGB_cross_val_stats.pkl')
data_labels = pd.read_pickle(MODEL_PATH + 'data_labels.pkl')

target_names = ['Non-Sensitive', 'Sensitive']
folds = len(LR_cross_val_stats["classifiers"])
doc_length = np.sum(
    [len(LR_cross_val_stats["test_features_list"][i]) for i in range(folds)])


def decontract(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


lemmatizer = WordNetLemmatizer()


def process_text(text):

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = str(re.sub("\S*\d\S*", "", text).strip())
    text = decontract(text)
    # tokenize texts
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tokens = tokenizer.tokenize(text)

    texts_clean = []
    for word in tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation+'...'):  # remove punctuation
            #
            stem_word = lemmatizer.lemmatize(
                word, "v")  # Lemmatizing word
            texts_clean.append(stem_word)

    return " ".join(texts_clean)


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

    average_length = int(data_labels.value_counts()[sens]/folds)
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
    elif clf == 'LSTM':
        LSTM_cross_val_stats = pd.read_pickle(
            MODEL_PATH + 'LSTM_cross_val_stats.pkl')
        return LSTM_cross_val_stats


def explainers(document_index: int, test_data: pd, test_labels: pd, extra_indexs: list, visual: str, cross_val_stats: dict) -> LimeTextExplainer:
    index = 0
    fold_length = len(test_data[0])

    # find which cross validation index to choose from
    while document_index > fold_length * (index+1) + extra_indexs[index] - 1:
        index += 1

    document_index -= fold_length * index + extra_indexs[index]
    specific_test = test_data[index].iloc[document_index]
    vectorizer = cross_val_stats["vectorizers"][index]
    model = cross_val_stats["classifiers"][index]
    clf_name = get_clf()
    max_len = 150

    def lime_explain(text=True):
        # lime_data = specific_test
        lime_data = specific_test[0:1000]

        lime_explainer = LimeTextExplainer(
            class_names=target_names)

        proba_predictor_func = None

        if clf_name == 'LSTM':

            def proba_predictor(arr):
                processed = []
                for i in arr:
                    processed.append(process_text(i))
                sequences = vectorizer.texts_to_sequences(processed)
                Ex = sequence.pad_sequences(sequences, maxlen=max_len)
                pred = model.predict(Ex)
                returnable = []
                for i in pred:
                    temp = i[0]
                    returnable.append(np.array([1-temp, temp]))
                return np.array(returnable)

            proba_predictor_func = proba_predictor

        else:
            def proba_predictor(texts):
                feature = vectorizer.transform(texts)
                pred = model.predict_proba(feature)
                return pred

            proba_predictor_func = proba_predictor

        lime_values = lime_explainer.explain_instance(
            lime_data,
            classifier_fn=proba_predictor_func,
        )

        if text:
            return lime_values.as_html(predict_proba=False, specific_predict_proba=False)
        else:
            return lime_values.as_html(text=False)

    def shap_explain():
        shap_values = cross_val_stats["shap_values"][index]

        force_plot = None
        if clf_name == 'LR':
            force_plot = shap.plots.force(
                shap_values[document_index], matplotlib=False)
        elif clf_name == 'XGB':
            explainer = shap.TreeExplainer(model)
            force_plot = shap.plots.force(
                explainer.expected_value, shap_values[document_index], feature_names=vectorizer.get_feature_names(), matplotlib=False)
        else:
            X_train = cross_val_stats["train_features_list"][index]
            X_test = cross_val_stats["test_features_list"][index]

            sequences = vectorizer.texts_to_sequences(X_train)
            sequences_matrix = sequence.pad_sequences(
                sequences, maxlen=max_len)

            processed = []
            for i in X_test:
                processed.append(process_text(i))

            test_sequences = vectorizer.texts_to_sequences(processed)
            test_sequences_matrix = sequence.pad_sequences(
                test_sequences, maxlen=max_len)

            explainer = shap.DeepExplainer(model, sequences_matrix)

            words = vectorizer.word_index
            num2word = {}
            for w in words.keys():
                num2word[words[w]] = w
            x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(
                x, "NONE"), test_sequences_matrix[i]))) for i in range(len(shap_values[0]))])

            force_plot = shap.plots.force(
                explainer.expected_value[0], shap_values[0][document_index], x_test_words[document_index])

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

        return shap_html

    shap_html = shap_explain()
    lime_html = lime_explain(text=False)
    eli5_weights = pickle.load(open(MODEL_PATH + "ELI5_weights.pkl", 'rb'))
    eli5_html = eli5_weights[document_index]

    vis_html = None
    highlighting = True
    if visual == 'LIME':
        vis_html = lime_explain()
    elif visual == 'ELI5':
        eli5_predictions = pickle.load(
            open(MODEL_PATH + "ELI5_predictions.pkl", 'rb'))
        vis_html = eli5_predictions[document_index]
    else:
        vis_html = specific_test.lower()
        highlighting = False

    test_labels = test_labels[index]
    isSensitive = (
        "Sensitive" if test_labels.iloc[document_index] else "Non-Sensitive")

    def predictor(texts):
        sens_clfs = []
        non_sens_clfs = []

        LSTM_cross_val_stats = pd.read_pickle(
            MODEL_PATH + 'LSTM_cross_val_stats.pkl')

        lstm_tok = LSTM_cross_val_stats["vectorizers"][index]
        lstm_model = LSTM_cross_val_stats["classifiers"][index]

        test_sequences = lstm_tok.texts_to_sequences(texts)
        test_sequences_matrix = sequence.pad_sequences(
            test_sequences, maxlen=max_len)

        y_pred = ''
        for i in lstm_model.predict(test_sequences_matrix):
            if i > 0.5:
                y_pred = 'Sensitive'
                sens_clfs.append('LSTM')
            else:
                y_pred = 'Non-Sensitive'
                non_sens_clfs.append('LSTM')

        predictions = [{'LSTM': y_pred}]

        vec = LR_cross_val_stats["vectorizers"][index]
        feature = vec.transform(texts)
        models = {LR_cross_val_stats["classifiers"][index]: 'LR',
                  XGB_cross_val_stats["classifiers"][index]: 'XGB'}

        for model, name in models.items():
            pred = model.predict(feature)
            if pred == True:
                predictions.append({name: 'Sensitive'})
                sens_clfs.append(name)
            else:
                predictions.append({name: 'Non-Sensitive'})
                non_sens_clfs.append(name)

        outlier = None

        if sens_clfs != 0 and non_sens_clfs != 0:
            if len(sens_clfs) == 1:
                outlier = sens_clfs[0]
            else:
                outlier = non_sens_clfs[0]

        return predictions, outlier

    prediction, outlier = predictor([specific_test])

    return shap_html, lime_html, vis_html, isSensitive, prediction, highlighting, eli5_html, outlier


def get_visual_html(sensitivity: int, document_number: int, visual: str, clf: str) -> LimeTextExplainer:

    cross_val_stats = get_clf_stats(clf)

    test_data, test_labels, extra_indexs = get_specific_sens(
        sensitivity, cross_val_stats)

    shap_html, lime_probas_html, visual_html, _, prediction, highlighting, eli5_html, outlier = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return shap_html, lime_probas_html, visual_html, prediction, highlighting, eli5_html, outlier


@bp.route('/')
def classifier_main_page():
    return render_template('classifier/index.html')


@bp.route('/sensitive-info', methods=('GET', 'POST'))
@login_required
def sensitive_info():
    sensitivity = 1

    document_number = get_doc_num(sensitivity)

    max_documents = data_labels.value_counts()[sensitivity]

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

    shap_html, lime_probas_html, visual_html, prediction, highlighting, eli5_html, outlier = get_visual_html(
        sensitivity, document_number, visual, clf)

    return render_template('classifier/sensitive_info.html', document_number=document_number+1, max_documents=max_documents,
                           curr_vis=visual, visual_html=visual_html, curr_clf=clf, shap_html=shap_html,
                           lime_probas_html=lime_probas_html, prediction=prediction, highlighting=highlighting,
                           eli5_html=eli5_html, outlier=outlier)


@bp.route('/non-sensitive-info', methods=('GET', 'POST'))
@login_required
def non_sensitive_info():
    sensitivity = 0

    document_number = get_doc_num(sensitivity)

    max_documents = data_labels.value_counts()[sensitivity]

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

    shap_html, lime_probas_html, visual_html, prediction, highlighting, eli5_html, outlier = get_visual_html(
        sensitivity, document_number, visual, clf)

    return render_template('classifier/non_sensitive_info.html', document_number=document_number+1, max_documents=max_documents,
                           curr_vis=visual, visual_html=visual_html, curr_clf=clf, shap_html=shap_html,
                           lime_probas_html=lime_probas_html, prediction=prediction, highlighting=highlighting,
                           eli5_html=eli5_html, outlier=outlier)


@bp.route('/single-document-sensitivity-info', methods=('GET', 'POST'))
@login_required
def single_document_sensitivity_info():

    document_number = get_doc_num()

    max_documents = len(data_labels)

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

    shap_html, lime_probas_html, visual_html, isSensitive, prediction, highlighting, eli5_html, outlier = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return render_template('classifier/single_document_sensitivity_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, curr_vis=visual, curr_clf=clf,
                           lime_probas_html=lime_probas_html, shap_html=shap_html, visual_html=visual_html,
                           prediction=prediction, highlighting=highlighting, eli5_html=eli5_html,
                           outlier=outlier)


@bp.route('/general-sensitivity-info', methods=('GET', 'POST'))
@login_required
def general_sensitivity_info():

    shap_images = []

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

    conf_mat_png = os.path.join(
        app.config['IMAGES_FOLDER'], clf+'/conf_mat.png')

    predictions = {}
    eli5_general = None
    if clf == 'LR':
        predictions = LR_cross_val_stats["predictions"]
    elif clf == 'XGB':
        predictions = XGB_cross_val_stats["predictions"]
    else:
        LSTM_cross_val_stats = pd.read_pickle(
            MODEL_PATH + 'LSTM_cross_val_stats.pkl')

        predictions = LSTM_cross_val_stats["predictions"]
    get_shap_images()

    return render_template('classifier/general_sensitivity_info.html', predictions=predictions, eli5_general=eli5_general,
                           conf_mat_png=conf_mat_png, curr_clf=clf, shap_images=shap_images)
