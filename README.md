# Installed required packages

pip install -r requirements.txt

# Initialise database

flask init-db

# Use this command to start project:

export FLASK_APP=flaskr && export FLASK_ENV=development && flask run

# test command

pytest

# Note

lime file needs to be modified to work.

`explanation.py` is a file modified from the lime library's codebase. For the flask project to show the lime explanations in a good format, you must replace lime's explanation.py file with the `explanations.py` file in this directory. You can do this after installing all packages using the `requirements.txt`.
