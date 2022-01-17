# Use this command to start project:

export FLASK_APP=flaskr && export FLASK_ENV=development && flask run

# test commands

test coverage: coverage run -m pytest
html report: coverage html

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
