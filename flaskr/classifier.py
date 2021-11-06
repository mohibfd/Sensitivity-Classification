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


@bp.route('/single-document-sensitivity-info')
def single_document_sensitivity_info():

    # c = make_pipeline(one_hot_vectorizer, lr_model)

    # class_names = ['Non-Sensitive', 'Sensitive']

    # explainer = LimeTextExplainer(class_names=class_names)

    # data = test_data['Body']

    # target = test_data['Sensitive']

    # first_idx = 2
    # second_idx = first_idx + 1

    # specific_data = data[first_idx:second_idx].iloc[0]

    # specific_target = target[first_idx:second_idx].iloc[0]
    return render_template('classifier/single_document_sensitivity_info.html')
