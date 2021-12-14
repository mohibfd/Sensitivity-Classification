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


documents_visibility_list = [
    (0, b'There are no previous documents', 'Previous Document'),
]

all_docs_visib = documents_visibility_list.copy()
all_docs_visib.append(
    (543, b'Document number =\n    543', 'Previous Document'))
all_docs_visib.append((543, b'Document number =\n    544', ''))
all_docs_visib.append((543, b'Document number =\n    545', 'Next Document'))
all_docs_visib.append((3800, b'There are no more documents', 'Next Document'))

sens_docs_visib = documents_visibility_list.copy()
sens_docs_visib.append((71, b'Document number =\n    71', 'Previous Document'))
sens_docs_visib.append((71, b'Document number =\n    72', ''))
sens_docs_visib.append((71, b'Document number =\n    73', 'Next Document'))
sens_docs_visib.append((501, b'There are no more documents', 'Next Document'))

non_sens_docs_visib = documents_visibility_list.copy()
non_sens_docs_visib.append(
    (471, b'Document number =\n    471', 'Previous Document'))
non_sens_docs_visib.append((471, b'Document number =\n    472', ''))
non_sens_docs_visib.append(
    (471, b'Document number =\n    473', 'Next Document'))
non_sens_docs_visib.append(
    (3298, b'There are no more documents', 'Next Document'))


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
