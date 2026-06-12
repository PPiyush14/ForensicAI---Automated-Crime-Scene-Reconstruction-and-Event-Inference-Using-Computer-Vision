"""
Crime Scene AI — Reconstruction & Event Inference System
Flask entry point
"""
from flask import Flask
from routes import register_routes
from config import Config
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    os.makedirs(app.config['UPLOAD_FOLDER'],  exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    register_routes(app)
    return app

if __name__ == '__main__':
    app = create_app()
    print("\n" + "="*60)
    print("  Crime Scene AI — Reconstruction & Event Inference")
    print("="*60)
    print("  Running at: http://localhost:5000")
    print("  AI Engine:  Groq LLaMA-4 Scout Vision (Free)")
    print("  Detection:  YOLOv8m + Custom blood/weapon models")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
