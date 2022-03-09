from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import TweetTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# import unpack
import numpy as np
# import eli5 as eli5
import pickle
import shap as shap
from lime.lime_text import LimeTextExplainer
import pandas as pd
import os
# from flaskr.db import get_db
from flaskr.auth import login_required
from flask import (
    Blueprint, flash, g, render_template, request, Flask
)
from . user import User
from .extensions import db

# import tensorflow as tf
# from tensorflow.keras.preprocessing import sequence
# tf.compat.v1.disable_v2_behavior()

bp = Blueprint('classifier', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "model/")

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER

#
survey = True
#

LR_cross_val_stats = []
XGB_cross_val_stats = []

survey_documents = pickle.load(
    open(MODEL_PATH + "survey_documents.pkl", 'rb'))


if survey:
    LR_cross_val_stats = None
else:
    LR_cross_val_stats = pd.read_pickle(MODEL_PATH + 'LR_cross_val_stats.pkl')

if survey:
    XGB_cross_val_stats = None
else:
    XGB_cross_val_stats = pd.read_pickle(
        MODEL_PATH + 'XGB_cross_val_stats.pkl')

data_labels = []
if not survey:
    data_labels = pd.read_pickle(MODEL_PATH + 'data_labels.pkl')


target_names = ['Non-Sensitive', 'Sensitive']

folds = 5
if not survey:
    folds = len(LR_cross_val_stats["classifiers"])

# doc_length = np.sum(
#     [len(LR_cross_val_stats["test_features_list"][i]) for i in range(folds)])


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
    user_id = g.user.id
    # db =()
    document_number = 0
    user = User.query.filter_by(id=user_id).first()

    if database == 0:
        document_number = user.non_sens_document_number
        # document_number = db.execute(
        #     'SELECT non_sens_document_number FROM user WHERE id = ?', (
        #         user_id,)
        # ).fetchone()[0]
    elif database == 1:
        document_number = user.sens_document_number
    else:
        document_number = user.document_number

    return document_number


def get_visualisation() -> str:
    user_id = g.user.id
    # db = get_db()

    user = User.query.filter_by(id=user_id).first()
    visual = user.visualisation_method

    return visual


def get_clf() -> str:
    user_id = g.user.id
    # db = get_db()

    user = User.query.filter_by(id=user_id).first()
    clf = user.clf_method
    # clf = db.execute(
    #     'SELECT clf_method FROM user WHERE id = ?', (user_id,)
    # ).fetchone()[0]

    return clf


def change_visual(visual: str):
    user_id = g.user.id
    # db = get_db()

    user = User.query.filter_by(id=user_id).first()
    user.visualisation_method = visual
    db.session.commit()

    # db.execute(
    #     'UPDATE user SET visualisation_method = ?'
    #     ' WHERE id = ?',
    #     (visual, user_id)
    # )


def change_clf(clf: str):
    user_id = g.user.id
    # db = get_db()

    user = User.query.filter_by(id=user_id).first()
    user.clf_method = clf
    db.session.commit()

    # db.execute(
    #     'UPDATE user SET clf_method = ?'
    #     ' WHERE id = ?',
    #     (clf, user_id)
    # )

    # db.commit()


def change_doc(document_number: int, max_documents: int, database="") -> int:
    user_id = g.user.id
    # db = get_db()
    user = User.query.filter_by(id=user_id).first()

    if request.form['submit_button'] == "Prev":
        if (document_number == 0):
            flash("There are no previous documents")
        else:
            document_number -= 1

    elif request.form['submit_button'] == 'Next':
        if (document_number == max_documents-1 and not survey):
            flash("There are no more documents")
        else:
            document_number += 1

    else:
        if survey and (database == 'change'):
            document_number = int(request.form['submit_button'])
        else:
            document_number = int(request.form['submit_button'])-1

    if database == 0:
        user.non_sens_document_number = document_number

    elif database == 1:
        user.sens_document_number = document_number

    else:
        user.document_number = document_number

    # db.commit()
    db.session.commit()

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
        if survey:
            return None
        return LR_cross_val_stats
    elif clf == 'XGB':
        if survey:
            return None
        return XGB_cross_val_stats
    elif clf == 'LSTM':
        if survey:
            return None
        LSTM_cross_val_stats = pd.read_pickle(
            MODEL_PATH + 'LSTM_cross_val_stats.pkl')
        return LSTM_cross_val_stats


def explainers(document_index: int, test_data: pd, test_labels: pd, extra_indexs: list, visual: str, cross_val_stats: dict) -> LimeTextExplainer:
    index = 0
    # fold_length = len(test_data[0])

    # # find which cross validation index to choose from
    # while document_index > fold_length * (index+1) + extra_indexs[index] - 1:
    #     index += 1

    # document_index -= fold_length * index + extra_indexs[index]

    specific_test = ''
    if survey:
        # user eval
        index = document_index
        specific_test = test_data[index]
    else:
        specific_test = test_data[index].iloc[document_index]

    clf_name = get_clf()
    vectorizer = None
    model = None
    if not survey:
        vectorizer = cross_val_stats["vectorizers"][index]
        model = cross_val_stats["classifiers"][index]

    max_len = 150

    def lime_explain(text_only=False, probas_only=False, lime_values=None):

        # lime_data = specific_test
        lime_data = specific_test[0:2000]

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

        if not lime_values:
            lime_values = lime_explainer.explain_instance(
                lime_data,
                classifier_fn=proba_predictor_func,
            )

        if text_only:
            return lime_values.as_html(predict_proba=False, specific_predict_proba=False)
        elif probas_only:
            return lime_values.as_html(text=False,  specific_predict_proba=False), lime_values
        else:
            return lime_values.as_html(text=False, predict_proba=False)

    def shap_explain():
        shap_values = None
        if not survey:
            shap_values = cross_val_stats["shap_values"][index]

        force_plot = None
        if clf_name == 'LR':
            if survey:
                force_plot = pickle.load(
                    open(MODEL_PATH + "LR_shap_surveys.pkl", 'rb'))[index]
            else:
                force_plot = shap.plots.force(
                    shap_values[document_index], matplotlib=False)

        elif clf_name == 'XGB':
            if survey:
                force_plot = pickle.load(
                    open(MODEL_PATH + "XGB_shap_surveys.pkl", 'rb'))[index]
            else:
                explainer = shap.TreeExplainer(model)
                force_plot = shap.plots.force(
                    explainer.expected_value, shap_values[document_index], feature_names=vectorizer.get_feature_names(), matplotlib=False)

        else:
            if survey:
                force_plot = pickle.load(
                    open(MODEL_PATH + "LSTM_shap_surveys.pkl", 'rb'))[index]
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

    def get_eli5_weights():
        eli5_weights = []
        if clf_name == 'LR':
            if survey:
                eli5_weights = pickle.load(
                    open(MODEL_PATH + "LR_eli5_explanations_survey.pkl", 'rb'))["weights"]
            else:
                eli5_weights = pickle.load(
                    open(MODEL_PATH + "LR_ELI5_explanations.pkl", 'rb'))["weights"]
        elif clf_name == 'XGB':
            if survey:
                eli5_weights = pickle.load(
                    open(MODEL_PATH + "XGB_eli5_explanations_survey.pkl", 'rb'))["weights"]
            else:
                eli5_weights = pickle.load(
                    open(MODEL_PATH + "XGB_ELI5_explanations.pkl", 'rb'))["weights"]
        else:
            if survey:
                eli5_weights = pickle.load(
                    open(MODEL_PATH + "LSTM_eli5_explanations_survey.pkl", 'rb'))["weights"]
            else:
                eli5_weights = pickle.load(
                    open(MODEL_PATH + "LSTM_ELI5_explanations.pkl", 'rb'))["weights"]

        return eli5_weights

    shap_html = shap_explain()
    eli5_weights = get_eli5_weights()
    eli5_html = None
    if survey:
        eli5_html = eli5_weights[index]
    else:
        eli5_html = eli5_weights[index][document_index]

    lime_probas = None
    lime_values = None
    lime_html = None
    if survey:
        lime_pickle = pickle.load(
            open(MODEL_PATH + f"survey_lime_{clf_name}.pkl", 'rb'))
        lime_probas = lime_pickle[index]['proba']
        lime_html = lime_pickle[index]['graph']

    else:
        lime_probas, lime_values = lime_explain(probas_only=True)
        lime_html = lime_explain(lime_values=lime_values)

    vis_html = None
    highlighting = True
    if visual == 'LIME':
        if survey:
            vis_html = pickle.load(
                open(MODEL_PATH + f"survey_lime_{clf_name}.pkl", 'rb'))[index]['highlighting']
        else:
            vis_html = lime_explain(text_only=True, lime_values=lime_values)
    elif visual == 'ELI5':
        eli5_predictions = []
        if clf_name == 'LR':
            if survey:
                eli5_predictions = pickle.load(
                    open(MODEL_PATH + "LR_eli5_explanations_survey.pkl", 'rb'))["predictions"]
            else:
                eli5_predictions = pickle.load(
                    open(MODEL_PATH + "LR_ELI5_explanations.pkl", 'rb'))["predictions"]
        elif clf_name == 'XGB':
            if survey:
                eli5_predictions = pickle.load(
                    open(MODEL_PATH + "XGB_eli5_explanations_survey.pkl", 'rb'))["predictions"]
            else:
                eli5_predictions = pickle.load(
                    open(MODEL_PATH + "XGB_ELI5_explanations.pkl", 'rb'))["predictions"]
        else:
            if survey:
                eli5_predictions = pickle.load(
                    open(MODEL_PATH + "LSTM_eli5_explanations_survey.pkl", 'rb'))["predictions"]
            else:
                eli5_predictions = pickle.load(
                    open(MODEL_PATH + "LSTM_ELI5_explanations.pkl", 'rb'))["predictions"]

        if survey:
            vis_html = eli5_predictions[index]
        else:
            vis_html = eli5_predictions[index][document_index]

    else:
        vis_html = specific_test.lower()
        highlighting = False

    # test_labels = test_labels[index]
    # user eval
    # isSensitive = (
    # "Sensitive" if test_labels.iloc[document_index] else "Non-Sensitive")
    isSensitive = None

    def predictor(texts):
        sens_clfs = []
        non_sens_clfs = []
        predictions = []

        if survey:
            def sens_pred(name):
                sens_clfs.append(name)
                predictions.append({name: 'Sensitive'})

            def non_sens_pred(name):
                non_sens_clfs.append(name)
                predictions.append({name: 'Non-Sensitive'})

            if specific_test == survey_documents[0]:
                non_sens_pred('LSTM')
                sens_pred('LR')
                non_sens_pred('XGB')

            elif specific_test == survey_documents[1]:
                sens_pred('LSTM')
                sens_pred('LR')
                sens_pred('XGB')

            elif specific_test == survey_documents[2]:
                non_sens_pred('LSTM')
                sens_pred('LR')
                sens_pred('XGB')

            elif specific_test == survey_documents[3]:
                non_sens_pred('LSTM')
                non_sens_pred('LR')
                non_sens_pred('XGB')

            elif specific_test == survey_documents[4]:
                non_sens_pred('LSTM')
                sens_pred('LR')
                sens_pred('XGB')

        # The maths behind this needs fixing since this is wrong
        # else:
        #     LSTM_cross_val_stats = pd.read_pickle(
        #         MODEL_PATH + 'LSTM_cross_val_stats.pkl')

        #     lstm_tok = LSTM_cross_val_stats["vectorizers"][index]
        #     lstm_model = LSTM_cross_val_stats["classifiers"][index]

        #     test_sequences = lstm_tok.texts_to_sequences(texts)
        #     test_sequences_matrix = sequence.pad_sequences(
        #         test_sequences, maxlen=max_len)

        #     y_pred = ''
        #     for i in lstm_model.predict(test_sequences_matrix):
        #         if i > 0.5:
        #             y_pred = 'Sensitive'
        #             sens_clfs.append('LSTM')
        #         else:
        #             y_pred = 'Non-Sensitive'
        #             non_sens_clfs.append('LSTM')

        #     predictions = [{'LSTM': y_pred}]

        #     vec = LR_cross_val_stats["vectorizers"][index]
        #     feature = vec.transform(texts)
        #     models = {LR_cross_val_stats["classifiers"][index]: 'LR',
        #               XGB_cross_val_stats["classifiers"][index]: 'XGB'}

        #     for model, name in models.items():
        #         pred = model.predict(feature)
        #         if pred == True:
        #             predictions.append({name: 'Sensitive'})
        #             sens_clfs.append(name)
        #         else:
        #             predictions.append({name: 'Non-Sensitive'})
        #             non_sens_clfs.append(name)

        outlier = None
        common_classifiers = None
        if len(sens_clfs) != 0 and len(non_sens_clfs) != 0:
            if len(sens_clfs) == 1:
                outlier = sens_clfs[0]
                common_classifiers = non_sens_clfs[0] + ', ' + non_sens_clfs[1]
            else:
                outlier = non_sens_clfs[0]
                common_classifiers = sens_clfs[0] + ', ' + sens_clfs[1]

        return predictions, outlier, common_classifiers

    prediction, outlier, common_classifiers = predictor([specific_test])

    return shap_html, lime_html, vis_html, isSensitive, prediction, highlighting, eli5_html, outlier, lime_probas, common_classifiers


def get_visual_html(sensitivity: int, document_number: int, visual: str, clf: str) -> LimeTextExplainer:

    cross_val_stats = get_clf_stats(clf)

    test_data = []
    test_labels = None
    extra_indexs = 0

    if survey:
        survey_documents = pickle.load(
            open(MODEL_PATH + "survey_documents.pkl", 'rb'))

        if sensitivity == 1:
            test_data = [survey_documents[0], survey_documents[1],
                         survey_documents[2], survey_documents[4]]
        else:
            test_data = [survey_documents[0], survey_documents[2],
                         survey_documents[3], survey_documents[4]]
    else:
        test_data, test_labels, extra_indexs = get_specific_sens(
            sensitivity, cross_val_stats)

    shap_html, lime_probas_html, visual_html, _, prediction, highlighting, eli5_html, outlier, lime_probas, common_classifiers = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return shap_html, lime_probas_html, visual_html, prediction, highlighting, eli5_html, outlier, lime_probas, common_classifiers


@bp.route('/')
def classifier_main_page():
    return render_template('classifier/index.html')


@bp.route('/sensitive-info', methods=('GET', 'POST'))
@login_required
def sensitive_info():
    sensitivity = 1

    # document_number = 0
    document_number = get_doc_num()
    if survey:
        # user_id = g.user['id']
        # db = get_db()
        # document_number = db.execute(
        #     'SELECT document_number FROM user WHERE id = ?', (user_id,)
        # ).fetchone()[0]
        if document_number == 4:
            document_number = 3
        # elif document_number == 0:
            # document_number = 1

    # else:
    #     document_number = get_doc_num(sensitivity)

    # user eval
    max_documents = 0
    if survey:
        max_documents = 4
    else:
        max_documents = data_labels.value_counts()[sensitivity]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_option')
        chosen_clf = request.form.get('clf_option')
        change_docs = request.form.get('submit_button')

        if change_docs:
            if survey:
                document_number = change_doc(
                    document_number, max_documents)
            else:
                document_number = change_doc(
                    document_number, max_documents, sensitivity)
        elif chosen_vis:
            visual = chosen_vis
            change_visual(visual)
        elif chosen_clf:
            clf = chosen_clf
            change_clf(clf)

    shap_html, lime_probas_html, visual_html, prediction, highlighting, eli5_html, outlier, lime_probas, common_classifiers = get_visual_html(
        sensitivity, document_number, visual, clf)

    # if not survey:
    document_number += 1

    return render_template('classifier/sensitive_info.html', document_number=document_number, max_documents=max_documents,
                           curr_vis=visual, visual_html=visual_html, curr_clf=clf, shap_html=shap_html,
                           lime_probas_html=lime_probas_html, prediction=prediction, highlighting=highlighting,
                           eli5_html=eli5_html, outlier=outlier, lime_probas=lime_probas, common_classifiers=common_classifiers)


@bp.route('/non-sensitive-info', methods=('GET', 'POST'))
@login_required
def non_sensitive_info():
    sensitivity = 0

    # document_number = 0
    document_number = get_doc_num()
    if survey:
        # user_id = g.user['id']
        # db = get_db()
        # document_number = db.execute(
        #     'SELECT document_number FROM user WHERE id = ?', (user_id,)
        # ).fetchone()[0]
        if document_number == 4:
            document_number = 3
        # elif document_number == 0:
            # document_number = 1

    # else:
    #     document_number = get_doc_num(sensitivity)

    # user eval
    max_documents = 0
    if survey:
        max_documents = 4
    else:
        max_documents = data_labels.value_counts()[sensitivity]

    visual = get_visualisation()

    clf = get_clf()

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_option')
        chosen_clf = request.form.get('clf_option')
        change_docs = request.form.get('submit_button')

        if change_docs:
            if survey:
                document_number = change_doc(
                    document_number, max_documents)
            else:
                document_number = change_doc(
                    document_number, max_documents, sensitivity)

        elif chosen_vis:
            visual = chosen_vis
            change_visual(visual)
        elif chosen_clf:
            clf = chosen_clf
            change_clf(clf)

    shap_html, lime_probas_html, visual_html, prediction, highlighting, eli5_html, outlier, lime_probas, common_classifiers = get_visual_html(
        sensitivity, document_number, visual, clf)

    # if not survey:
    document_number += 1

    return render_template('classifier/non_sensitive_info.html', document_number=document_number, max_documents=max_documents,
                           curr_vis=visual, visual_html=visual_html, curr_clf=clf, shap_html=shap_html,
                           lime_probas_html=lime_probas_html, prediction=prediction, highlighting=highlighting,
                           eli5_html=eli5_html, outlier=outlier, lime_probas=lime_probas, common_classifiers=common_classifiers)


@bp.route('/single-document-sensitivity-info', methods=('GET', 'POST'))
@login_required
def single_document_sensitivity_info():

    document_number = get_doc_num()

    # user eval
    max_documents = 0
    if survey:
        max_documents = 5
    else:
        max_documents = len(data_labels)

    extra_indexs = [0 for _ in range(folds)]

    visual = get_visualisation()

    clf = get_clf()

    # user_id = g.user['id']
    # db = get_db()
    # posts = db.execute(
    #     'SELECT s.document_number, classifiers_chosen, feature1, feature2, feature3, feature4, feature5'
    #     ' FROM survey s JOIN user u ON s.author_id = u.id'
    #     ' WHERE u.id = ?'
    #     ' ORDER BY s.document_number DESC',
    #     (user_id,)
    # ).fetchall()

    # import sys
    # # accessing first document survey
    # print(len(posts), file=sys.stderr)
    # print(posts[0][0], file=sys.stderr)
    # print(posts[0][1], file=sys.stderr)
    # print(posts[0][2], file=sys.stderr)
    # for i in posts[0]:
    #     print(i, file=sys.stderr)
    # for i in posts[1]:
    #     print(i, file=sys.stderr)
    # print(posts[0], file=sys.stderr)
    # print(posts[1], file=sys.stderr)

    # # accessing second document survey
    # print(posts[1][0], file=sys.stderr)
    # print(posts[1][1], file=sys.stderr)
    # print(posts[1][2], file=sys.stderr)
    # # accessing third document survey
    # print(posts[2][0], file=sys.stderr)
    # print(posts[2][1], file=sys.stderr)
    # print(posts[2][2], file=sys.stderr)

    if request.method == 'POST':
        chosen_vis = request.form.get('vis_option')
        chosen_clf = request.form.get('clf_option')
        change_docs = request.form.get('submit_button')

        if change_docs:
            if survey:
                visual = None
                change_visual('None')

            document_number = change_doc(
                document_number, max_documents)
        elif chosen_vis:
            visual = chosen_vis
            change_visual(visual)
        elif chosen_clf:
            clf = chosen_clf
            change_clf(clf)
        else:
            user_features = [request.form.get(
                f'feature{i}') for i in range(1, 6)]
            radio_option_clf = request.form.get('inlineRadioOptions')
            error = None

            for i in user_features:
                if i == '':
                    error = "Please enter 5 features"
                    flash(error)
                    break

            outlier = request.form.get('outlier_name')
            if radio_option_clf == None and outlier != 'None':
                error = "Please choose one of the classifiers"
                flash(error)

            # if error is None:
            #     db = get_db()
            #     db.execute(
            #         "INSERT INTO survey (author_id, document_number, feature1, feature2, feature3, feature4, feature5, classifiers_chosen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            #         (g.user['id'], document_number, user_features[0], user_features[1],
            #          user_features[2], user_features[3], user_features[4], radio_option_clf),
            #     )
            #     db.commit()

    cross_val_stats = get_clf_stats(clf)

    test_data = []
    test_labels = []
    if survey:
        test_data = pickle.load(
            open(MODEL_PATH + "survey_documents.pkl", 'rb'))
    else:
        test_data = cross_val_stats["test_features_list"]
        test_labels = cross_val_stats["test_labels_list"]

    shap_html, lime_probas_html, visual_html, isSensitive, prediction, highlighting, eli5_html, outlier, lime_probas, common_classifiers = explainers(
        document_number, test_data, test_labels, extra_indexs, visual, cross_val_stats)

    return render_template('classifier/single_document_sensitivity_info.html', document_number=document_number+1,
                           max_documents=max_documents, isSensitive=isSensitive, curr_vis=visual, curr_clf=clf,
                           lime_probas_html=lime_probas_html, shap_html=shap_html, visual_html=visual_html,
                           prediction=prediction, highlighting=highlighting, eli5_html=eli5_html,
                           outlier=outlier, lime_probas=lime_probas, common_classifiers=common_classifiers)


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
        if survey:
            predictions = pd.read_pickle(
                MODEL_PATH + 'LR_predictions.pkl')
        else:
            predictions = LR_cross_val_stats["predictions"]

    elif clf == 'XGB':
        if survey:
            predictions = pd.read_pickle(
                MODEL_PATH + 'XGB_predictions.pkl')
        else:
            predictions = XGB_cross_val_stats["predictions"]

    else:
        if survey:
            predictions = pd.read_pickle(
                MODEL_PATH + 'LSTM_predictions.pkl')
        else:
            predictions = pd.read_pickle(
                MODEL_PATH + 'LSTM_cross_val_stats.pkl')['predictions']

    get_shap_images()

    return render_template('classifier/general_sensitivity_info.html', predictions=predictions, eli5_general=eli5_general,
                           conf_mat_png=conf_mat_png, curr_clf=clf, shap_images=shap_images)
