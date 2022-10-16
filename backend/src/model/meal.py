from sqlalchemy import Column, ForeignKey, Integer, Table
import sqlalchemy as sa

from . import db


class Meal(db.Model):
    __tablename__ = "meal"
    meal_id = sa.Column(sa.Integer, primary_key=True)
    account_id = sa.Column(Integer, ForeignKey("account.account_id"))
    notes = sa.Column(sa.String)
    image_path = sa.Column(sa.String)
    healthy_food = sa.Column(sa.Boolean)
    food_prodicted = sa.Column(sa.String)
    calories = sa.Column(sa.Float)
    carbs = sa.Column(sa.Float)
    protein = sa.Column(sa.Float)
    fat = sa.Column(sa.Float)

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
