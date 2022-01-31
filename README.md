# Use this command to start project:

export FLASK_APP=flaskr && export FLASK_ENV=development && flask run

# test commands

## ensure you are in sens_class_webs dir

_test coverage:_ coverage run -m pytest
_html report:_ coverage html
_view:_ open htmlcov/index.html

# Command to print in terminal:

import sys
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', file=sys.stderr)
print(document_index, file=sys.stderr)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', file=sys.stderr)

# Commands for pickle:

pickle.dump(one_hot_vectorizer, open(directory_path+filename, 'wb'))
temp = pickle.load(open(directory_path+filename, 'rb'))

# Hide input

%%model

# Change database

flask init-db

# PS

lime file needs to be modified to work.
