from flask.cli import cli
import pytest
from flask import g, session


@pytest.mark.parametrize('path', (
    '/',
    '/classifier-main-page',
    '/sensitive-info',
    '/non-sensitive-info',
    '/general-sensitivity-info',
    '/single-document-sensitivity-info',

))
def test_paths(client, path):
    assert client.get(path).status_code == 200
