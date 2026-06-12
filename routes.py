"""
Flask Routes for Crime Scene AI
"""
from flask import render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os, uuid, logging
from pathlib import Path

logger = logging.getLogger(__name__)

_detector = None
_inference_engine = None


def get_detector():
    global _detector
    if _detector is None:
        from models.detector import CrimeSceneDetector
        from config import Config
        _detector = CrimeSceneDetector(
            model_path=Config.YOLO_MODEL_PATH,
            confidence=Config.DETECTION_CONFIDENCE,
            iou=Config.DETECTION_IOU
        )
    return _detector


def get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        from models.inference_engine import ForensicInferenceEngine
        _inference_engine = ForensicInferenceEngine()
    return _inference_engine


def allowed_file(filename, exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts


def register_routes(app):

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/analyze', methods=['POST'])
    def analyze():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'error': 'Invalid file type. Supported: JPG, PNG, WEBP, BMP'}), 400

        filename     = secure_filename(file.filename)
        unique_id    = uuid.uuid4().hex[:8]
        case_id      = unique_id.upper()
        saved_name   = f"{Path(filename).stem}_{unique_id}{Path(filename).suffix}"
        upload_path  = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_name)
        file.save(upload_path)

        try:
            from utils.preprocessor import (
                preprocess_image, save_annotated_image,
                generate_heatmap, generate_evidence_grid
            )

            pre_path = preprocess_image(upload_path, output_dir=current_app.config['UPLOAD_FOLDER'])

            detector = get_detector()
            detections, annotated = detector.detect(pre_path)

            annotated_path = save_annotated_image(
                annotated, current_app.config['RESULTS_FOLDER'], saved_name
            )
            heatmap_path = generate_heatmap(pre_path, detections, current_app.config['RESULTS_FOLDER']) if detections else None
            grid_path    = generate_evidence_grid(pre_path, detections, current_app.config['RESULTS_FOLDER']) if detections else None

            engine      = get_inference_engine()
            full_report = engine.analyze_scene(pre_path, detections)

            def web_path(p):
                if p is None: return None
                return '/' + p.replace(os.sep, '/')

            return jsonify({
                'success':        True,
                'case_id':        case_id,
                'original_image': f"/static/uploads/{saved_name}",
                'annotated_image': web_path(annotated_path),
                'heatmap_image':   web_path(heatmap_path),
                'evidence_grid':   web_path(grid_path),
                'report':          full_report,
            })

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/health')
    def health():
        has_groq      = bool(os.environ.get('GROQ_API_KEY'))
        has_roboflow  = bool(os.environ.get('ROBOFLOW_API_KEY'))
        return jsonify({
            'status':       'running',
            'ai_engine':    'Groq LLaMA-4 Scout Vision (Free)',
            'groq_key':     'configured' if has_groq     else 'MISSING — add to .env (free at console.groq.com)',
            'roboflow_key': 'configured' if has_roboflow else 'not set (optional)',
            'base_model':   os.environ.get('YOLO_MODEL_PATH', 'yolov8m.pt'),
        })
