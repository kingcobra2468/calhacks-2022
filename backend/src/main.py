import os

from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import create_engine

from model import db
from route import accounts_blueprint, meals_blueprint


load_dotenv()

app = Flask(__name__)

conn = os.environ['COCKROACH_DB_URI']
engine = create_engine(conn)

app.config['SQLALCHEMY_DATABASE_URI'] = conn
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    #db.drop_all()
    db.create_all()

app.register_blueprint(accounts_blueprint, url_prefix='/api')
app.register_blueprint(meals_blueprint, url_prefix='/api')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7200, debug=False)
