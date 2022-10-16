from email.mime import image
import random
import json
import math
import os

from flask import Blueprint, jsonify, request

from model.account import Account
from model.meal import Meal
from model import db
from util.image import save_image_base64
import calorie_counter

meals_blueprint = Blueprint('meals', __name__)


@meals_blueprint.route('/meals/<account_id>', methods=['POST'])
def add_meal(account_id):
    account = account = Account.query.filter_by(account_id=account_id).first()
    if not account:
        return {
            'status': 'fail',
            'data': {
                'msg': 'Account does not exist'
            }
        }, 401

    image_b64 = request.json.get('image_b64')
    notes = request.json.get('notes')
    image_id = random.randint(0, 1000000)
    meal = Meal(notes=notes, account_id=account_id,
                image_path=os.path.join('/static/', f'{image_id}.png'))

    image_path = save_image_base64(image_b64, image_id)
    _, meta = calorie_counter.count_calories(image_path, 100)

    meal.carbs = float(meta['carb'])
    meal.healthy_food = meta['healthy_food']
    meal.food_prodicted = meta['food_pred']
    meal.calories = float(meta['cal'])
    meal.protein = float(meta['prot'])
    meal.fat = float(meta['fat'])

    db.session.add(meal)
    db.session.commit()

    #save_image_base64(image_b64, meal.meal_id)

    return {
        'status': 'success',
        'data': None
    }


@meals_blueprint.route('/meals/<account_id>', methods=['GET'])
def get_meals(account_id):
    account = account = Account.query.filter_by(account_id=account_id).first()
    if not account:
        return {
            'status': 'fail',
            'data': {
                'msg': 'Account does not exist'
            }
        }, 401

    meals = [meal.as_dict() for meal in list(
        Meal.query.filter_by(account_id=account_id))]

    return {
        'status': 'success',
        'data': {'meals': meals}
    }
