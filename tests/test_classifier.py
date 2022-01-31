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


prev_action = 'Previous'
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

sens_docs_visib = documents_visibility_list.copy()
sens_docs_visib.append((71, b'71', prev_action))
sens_docs_visib.append((71, b'72', '71'))
sens_docs_visib.append((71, b'73', next_action))
sens_docs_visib.append((501, b'There are no more documents', next_action))

non_sens_docs_visib = documents_visibility_list.copy()
non_sens_docs_visib.append(
    (471, b'471', prev_action))
non_sens_docs_visib.append((471, b'472', '471'))
non_sens_docs_visib.append(
    (471, b'473', next_action))
non_sens_docs_visib.append(
    (3298, b'There are no more documents', next_action))


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
