from sqlalchemy import Column, ForeignKey, Integer, Table
import sqlalchemy as sa

from . import db


class Meal(db.Model):
    __tablename__ = "meal"
    meal_id = sa.Column(sa.Integer, primary_key=True)
    account_id = sa.Column(Integer, ForeignKey("account.account_id"))
    note = sa.Column(sa.String)
    type = sa.Column(sa.String)
