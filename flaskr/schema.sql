DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;

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