"""
Configuration Settings for Crime Scene AI
"""
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'crime-scene-ai-secret-2026')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    UPLOAD_FOLDER  = os.path.join('static', 'uploads')
    RESULTS_FOLDER = os.path.join('static', 'results')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

    # API Keys
    GROQ_API_KEY      = os.environ.get('GROQ_API_KEY', '')
    ROBOFLOW_API_KEY  = os.environ.get('ROBOFLOW_API_KEY', '')

    # YOLO — use medium model for much better accuracy (still free & local)
    YOLO_MODEL_PATH      = os.environ.get('YOLO_MODEL_PATH', 'yolov8m.pt')
    BLOOD_MODEL_PATH     = os.path.join('models', 'blood_best.pt')
    WEAPON_MODEL_PATH    = os.path.join('models', 'weapon_best.pt')

    DETECTION_CONFIDENCE = 0.25
    DETECTION_IOU        = 0.45

    @staticmethod
    def init_app():
        os.makedirs(Config.UPLOAD_FOLDER,  exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
