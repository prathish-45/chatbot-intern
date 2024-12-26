from flask import Flask

app = Flask(__name__)

from app import main  # Import the main module to set up routes and functionality