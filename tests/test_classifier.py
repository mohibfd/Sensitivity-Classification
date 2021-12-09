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


@pytest.mark.parametrize(('document_num', 'message', 'button_action'), (
    (0, b'There are no previous documents', 'Previous Document'),
    (500, b'document number =\n  500', 'Previous Document'),
    (500, b'document number =\n  501', ''),
    (500, b'document number =\n  502', 'Next Document'),
    (3800, b'There are no more documents', 'Next Document'),
))
def test_documents_visibility(client, app, auth, document_num, message, button_action):
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
