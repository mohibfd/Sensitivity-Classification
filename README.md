# Use this command to install the necessary packages

pip install -r requirements.txt

# Use this command to initialise the database

flask init-db

# Use this command to start the project:

export FLASK_APP=flaskr && export FLASK_ENV=development && flask run

# Use this command for running the tests

pytest

# Note

The Lime file needs to be modified to work.

`explanation.py` is a file modified from the lime library's codebase. For the flask project to show the lime explanations in a good format, you must replace lime's explanation.py file with the `explanations.py` file in this directory. After installing all packages using the `requirements.txt` file, you can do this.
