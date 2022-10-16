from flask import Flask

from route import accounts_blueprint, meals_blueprint

app = Flask(__name__)

app.register_blueprint(accounts_blueprint, url_prefix='/api')
app.register_blueprint(meals_blueprint, url_prefix='/api')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7200, debug=False)
