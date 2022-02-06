import pytest
from flaskr.db import get_db

paths = (
    '/sensitive-info',
    '/non-sensitive-info',
    '/general-sensitivity-info',
    '/single-document-sensitivity-info',
)


@pytest.mark.parametrize('path', paths)
def test_paths_no_login(client, path):
    assert client.get(path).status_code == 302


@pytest.mark.parametrize('path', paths)
def test_paths_logged_in(client, path, auth):
    auth.login()
    assert client.get(path).status_code == 200


prev_action = "Prev"
next_action = 'Next'
documents_visibility_list = [
    (0, b'There are no previous documents', prev_action),
]

all_docs_visib = documents_visibility_list.copy()
all_docs_visib.append(
    (543, b'543', prev_action))
all_docs_visib.append((543, b'544', '543'))
all_docs_visib.append((543, b'545', next_action))
all_docs_visib.append((3800, b'There are no more documents', next_action))


@pytest.mark.parametrize(('document_num', 'message', 'button_action'), all_docs_visib)
def test_single_documents_visibility(client, app, auth, document_num, message, button_action):
    auth.login()
    with app.app_context():
        db = get_db()
        db.execute('UPDATE user SET document_number = ? WHERE id = 1',
                   (document_num,)
                   )
        db.commit()

    response = client.post(
        '/single-document-sensitivity-info',
        data={'submit_button': button_action}
    )

    assert message in response.data


sens_docs_visib = documents_visibility_list.copy()
sens_docs_visib.append((71, b'71', prev_action))
sens_docs_visib.append((71, b'72', '71'))
sens_docs_visib.append((71, b'73', next_action))
sens_docs_visib.append((501, b'There are no more documents', next_action))


@pytest.mark.parametrize(('document_num', 'message', 'button_action'), sens_docs_visib)
def test_sensitive_documents_visibility(client, app, auth, document_num, message, button_action):
    auth.login()
    with app.app_context():
        db = get_db()
        db.execute('UPDATE user SET sens_document_number = ? WHERE id = 1',
                   (document_num,)
                   )
        db.commit()

    response = client.post(
        '/sensitive-info',
        data={'submit_button': button_action}
    )

    assert message in response.data


non_sens_docs_visib = documents_visibility_list.copy()
non_sens_docs_visib.append(
    (471, b'471', prev_action))
non_sens_docs_visib.append((471, b'472', '471'))
non_sens_docs_visib.append(
    (471, b'473', next_action))
non_sens_docs_visib.append(
    (3298, b'There are no more documents', next_action))


@pytest.mark.parametrize(('document_num', 'message', 'button_action'), non_sens_docs_visib)
def test_non_sensitive_documents_visibility(client, app, auth, document_num, message, button_action):
    auth.login()
    with app.app_context():
        db = get_db()
        db.execute('UPDATE user SET non_sens_document_number = ? WHERE id = 1',
                   (document_num,)
                   )
        db.commit()

    response = client.post(
        '/non-sensitive-info',
        data={'submit_button': button_action}
    )

    assert message in response.data


eli5 = 'ELI5'
lime = 'LIME'
lr = 'LR'
svm = 'SVM'
rf = 'RF'

visuals = [eli5, lime]
classifiers = [lr, svm, rf]
clf_types = ['non-sensitive-info', 'sensitive-info',
             'single-document-sensitivity-info']

diff_options = []

for i in clf_types:
    for j in visuals:
        for k in classifiers:
            if j == lime:
                diff_options.append((i, j, k, b'Text with highlighted words'))
            elif j == eli5:
                diff_options.append((i, j, k, b'top features'))


@pytest.mark.parametrize(('clf_type', 'visual', 'classifier', 'message'), diff_options)
def test_dropdown_options(client, auth, clf_type,  visual, classifier, message):
    auth.login()

    response = client.post(
        '/' + clf_type,
        data={'clf_options': classifier, 'vis_options': visual,
              'options_button': 'Submit'}
    )

    assert message in response.data
