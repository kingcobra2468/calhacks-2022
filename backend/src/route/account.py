from flask import Blueprint


accounts_blueprint = Blueprint('accounts', __name__)


@accounts_blueprint.route('/accounts/login', methods=['POST'])
def login():
    return {
        'status': 'success',
        'data': None
    }


@accounts_blueprint.route('/accounts/register', methods=['POST'])
def register():
    return {
        'status': 'success',
        'data': None
    }
