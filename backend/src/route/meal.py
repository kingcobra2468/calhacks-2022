from email.mime import image
import json
import os

from flask import Blueprint, jsonify, request

from model.account import Account
from model.meal import Meal
from model import db

from util.image import save_image_base64

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

    meal = Meal(notes=notes, type='healthy', account_id=account_id,
                image_path=os.path.join('/static/', f'{account_id}.png'))
    db.session.add(meal)
    db.session.commit()

    save_image_base64(image_b64, meal.meal_id)

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

    meals = [meal.as_dict() for meal in list(Meal.query.filter_by(account_id=account_id))]
    
    return {
        'status': 'success',
        'data': {'meals': meals}
    }
