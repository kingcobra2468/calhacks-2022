import json
from flask import Blueprint, jsonify


meals_blueprint = Blueprint('meals', __name__)


@meals_blueprint.route('/meals/<account_id>', methods=['POST'])
def add_meal(account_id):
    return {
        'status': 'success',
        'data': None
    }


@meals_blueprint.route('/meals/<account_id>', methods=['GET'])
def get_meals(account_id):
    return {
        'status': 'success',
        'data': {'meals': [
            {'image_path': '/stub',
             'type': 'healthy',
             'date': 1665897302,
             'notes': 'good'
             }, {'image_path': '/stub',
                 'type': 'unhealthy',
                 'date': 1666897302,
                 'notes': 'did not like'
                 }
        ]}
    }
