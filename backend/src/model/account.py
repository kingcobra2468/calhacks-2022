import sqlalchemy as sa

from model import db


class Account(db.Model):
    __tablename__ = "account"
    account_id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String)
    password = sa.Column(sa.String)
