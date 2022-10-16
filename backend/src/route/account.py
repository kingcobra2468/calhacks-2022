from flask import Blueprint, request

from model.account import Account
from model import db

accounts_blueprint = Blueprint('accounts', __name__)


@accounts_blueprint.route('/accounts/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    account = Account.query.filter_by(
        username=username).first()
    if not account:
        return {
            'status': 'fail',
            'data': {
                'msg': 'Account does not exist'
            }
        }, 401

    return {
        'status': 'success',
        'data': {
            'id': account.account_id
        }
    }


@accounts_blueprint.route('/accounts/registration', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')

    account = Account.query.filter_by(
        username=username).first()

    if account:
        return {
            'status': 'fail',
            'data': {
                'msg': 'Account does not exist'
            }
        }, 403

    account = Account(username=username, password=password)
    db.session.add(account)
    db.session.commit()

    return {
        'status': 'success',
        'data': {
            'id': account.account_id
        }
    }
