from .extensions import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))
    password = db.Column(db.String(50))
    document_number = db.Column(db.Integer)
    sens_document_number = db.Column(db.Integer)
    non_sens_document_number = db.Column(db.Integer)
    visualisation_method = db.Column(db.String(10))
    clf_method = db.Column(db.String(10))
