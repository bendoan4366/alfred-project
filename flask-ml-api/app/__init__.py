from flask_api import FlaskAPI

app = FlaskAPI(__name__)

from app import routes