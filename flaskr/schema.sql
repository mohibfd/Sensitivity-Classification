DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS survey;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  document_number INTEGER NOT NULL,
  sens_document_number INTEGER NOT NULL,
  non_sens_document_number INTEGER NOT NULL,
  visualisation_method VARCHAR NOT NUlL,
  clf_method VARCHAR NOT NUlL
);

CREATE TABLE survey (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  document_number INTEGER NOT NULL,
  feature1 VARCHAR NOT NUlL,
  feature2 VARCHAR NOT NUlL,
  feature3 VARCHAR NOT NUlL,
  feature4 VARCHAR NOT NUlL,
  feature5 VARCHAR NOT NUlL,
  classifiers_chosen VARCHAR,
  FOREIGN KEY (author_id) REFERENCES user (id)
);
  