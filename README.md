Use this command to start project:
export FLASK_APP=flaskr && export FLASK_ENV=development && flask run

test coverage: coverage run -m pytest
html report: coverage html

command to print in terminal:
import sys
print('Hello world!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', file=sys.stderr)
print(specific_data, file=sys.stderr)

commands for pickle:
pickle.dump(one_hot_vectorizer, open(directory_path+filename, 'wb'))
temp = pickle.load(open(directory_path+filename, 'rb'))

%%model
this will hide input
