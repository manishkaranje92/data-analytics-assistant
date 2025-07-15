# app/__init__.py
import os
from flask import Flask
import duckdb
import spacy

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for multiple tables
db_connection = duckdb.connect(database=':memory:', read_only=False)
loaded_tables = []

# Load spaCy model once
try:
    nlp_model = spacy.load("en_core_web_sm")
    print("spaCy NLP model loaded successfully.")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_model = spacy.load("en_core_web_sm")

# Import routes after app initialization to avoid circular imports
from app import routes
