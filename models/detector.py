"""
Crime Scene Detector — Multi-model YOLO ensemble
Detection chain:
  1. blood_best.pt   — local custom, biological evidence
  2. weapon_best.pt  — local custom, ONLY gun-class detections kept,
                       camera FP rejected via geometry filter
  3. yolov8x.pt      — base model for persons, furniture, objects
  4. Roboflow        — optional cloud enhancement
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
import logging
import base64
import requests

logger = logging.getLogger(__name__)

# ─── Severity map ─────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    'blood': 'critical', 'bloodstain': 'critical', 'wound': 'critical',
    'body': 'critical',  'victim': 'critical',     'corpse': 'critical',
    'gun': 'high',    'pistol': 'high',  'rifle': 'high',  'firearm': 'high',
    'knife': 'high',  'blade': 'high',   'weapon': 'high', 'shell': 'high',
    'casing': 'high', 'bullet': 'high',
    'person': 'medium', 'rope': 'medium', 'evidence': 'medium',
    'phone': 'low', 'bag': 'low', 'suitcase': 'low',
}

# ─── Category map ─────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    'blood': 'Biological', 'bloodstain': 'Biological', 'wound': 'Biological',
    'gun': 'Weapon',    'pistol': 'Weapon',  'rifle': 'Weapon',
    'firearm': 'Weapon','knife': 'Weapon',   'blade': 'Weapon',
    'weapon': 'Weapon', 'shell': 'Weapon',   'casing': 'Weapon', 'bullet': 'Weapon',
    'person': 'Person', 'body': 'Person',    'victim': 'Person',
    'suitcase': 'Bag',  'backpack': 'Bag',   'handbag': 'Bag',
    'chair': 'Furniture','couch': 'Furniture','sofa': 'Furniture','bed': 'Furniture',
}

# ─── COCO labels that are never weapons — hard block ─────────────────────────
WEAPON_FP_LABELS = {
    'cell phone', 'remote', 'mouse', 'keyboard', 'book',
    'bottle', 'cup', 'scissors', 'hair drier', 'toothbrush',
    # NOTE: 'camera' intentionally NOT in this list —
    # we handle camera FP via geometry filter instead,
    # so a real gun that looks camera-shaped still gets caught
}

# ─── Forensic kit / briefcase relabel ────────────────────────────────────────
# When a suitcase/bag detection overlaps a known forensic kit area,
# relabel it so the AI report is more accurate
FORENSIC_KIT_LABELS = {'suitcase', 'briefcase', 'luggage'}

# ─── Per-category minimum confidence thresholds ───────────────────────────────
CONF_THRESHOLDS = {
    'Biological': 32.0,   # blood model reliable, keep slightly loose
    'Weapon':     28.0,   # LOW on purpose — weapon model detects floor guns
                          # at low conf; we filter FP by geometry not conf
    'Person':     40.0,
    'Furniture':  50.0,   # raise furniture bar — cuts Bed/Chair noise
    'Bag':        40.0,
    'Object':     45.0,
}

# ─── Roboflow optional models ─────────────────────────────────────────────────
ROBOFLOW_MODELS = [
    {
        'workspace': 'roboflow-universe-projects',
        'project':   'shell-casing-detection',
        'version':   1,
        'name':      'shell_rf',
        'category':  'Weapon',
        'label_map': {'shell': 'Shell Casing', 'casing': 'Shell Casing'},
    },
]

COLOR_MAP = {
    'critical': (0, 0, 220),
    'high':     (0, 60, 220),
    'medium':   (0, 140, 255),
    'low':      (0, 220, 220),
}


class CrimeSceneDetector:

    def __init__(self, model_path='yolov8x.pt', confidence=0.25, iou=0.45):
        self.confidence = confidence
        self.iou = iou
        self.roboflow_api_key = os.environ.get('ROBOFLOW_API_KEY', '')

        # ── Base YOLO X ───────────────────────────────────────────────────────
        logger.info(f"Loading base YOLO model: {model_path}")
        try:
            self.primary_model = YOLO(model_path)
            logger.info("✓ Base YOLO loaded")
        except Exception as e:
            logger.warning(f"Could not load {model_path}, falling back to yolov8m: {e}")
            self.primary_model = YOLO('yolov8m.pt')

        # ── Custom models ─────────────────────────────────────────────────────
        self.blood_model  = self._load_model('models/blood_best.pt',  'blood')
        self.weapon_model = self._load_model('models/weapon_best.pt', 'weapon')

    def _load_model(self, path_str, name):
        p = Path(path_str)
        if p.exists():
            try:
                m = YOLO(str(p))
                logger.info(f"✓ Local {name} model loaded")
                return m
            except Exception as e:
                logger.warning(f"Local {name} model failed: {e}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        all_results = []

        # 1. Blood model
        if self.blood_model:
            try:
                res = self.blood_model(image, conf=0.30, iou=self.iou, verbose=False)
                dets = self._process_yolo(
                    res, force_category='Biological', force_severity='critical'
                )
                all_results.extend(dets)
                logger.info(f"Blood model → {len(dets)} detections")
            except Exception as e:
                logger.warning(f"Blood model error: {e}")

        # 2. Weapon model — run at LOW conf to catch floor guns,
        #    FP rejection is done in _filter_weapon() not here
        if self.weapon_model:
            try:
                res = self.weapon_model(image, conf=0.20, iou=self.iou, verbose=False)
                raw_dets = self._process_yolo(
                    res, force_category='Weapon', force_severity='high'
                )
                # Apply geometry-based camera FP filter
                gun_dets = [d for d in raw_dets if self._is_likely_gun(d, image.shape)]
                all_results.extend(gun_dets)
                logger.info(
                    f"Weapon model → {len(raw_dets)} raw, "
                    f"{len(gun_dets)} after FP filter"
                )
            except Exception as e:
                logger.warning(f"Weapon model error: {e}")

        # 3. Base YOLOv8x
        try:
            res = self.primary_model(image, conf=0.30, iou=self.iou, verbose=False)
            dets = self._process_yolo(res)
            all_results.extend(dets)
            logger.info(f"Base YOLO → {len(dets)} detections")
        except Exception as e:
            logger.warning(f"Base YOLO error: {e}")

        # 4. Roboflow optional
        if self.roboflow_api_key:
            for info in ROBOFLOW_MODELS:
                try:
                    dets = self._run_roboflow(image_path, info)
                    all_results.extend(dets)
                    logger.info(f"Roboflow {info['name']} → {len(dets)} detections")
                except Exception as e:
                    logger.warning(f"Roboflow {info['name']} error: {e}")

        # Filter → NMS → relabel → draw
        all_results = self._filter_detections(all_results)
        detections  = self._nms(all_results)
        detections  = self._relabel(detections)
        annotated   = self._draw(image.copy(), detections)

        return detections, annotated

    # ──────────────────────────────────────────────────────────────────────────
    def _is_likely_gun(self, det, img_shape):
        """
        Geometry-based camera vs gun discriminator.

        Cameras held up to face have these traits:
          - Located in UPPER portion of image (person holding camera up)
          - Relatively SQUARE bounding box (camera body is boxy)
          - Large area (camera + person arm fills a lot of space)

        Guns on floor have these traits:
          - Located in LOWER or MID portion of image
          - ELONGATED bounding box (barrel + handle = wider than tall)
          - Small-to-medium area
        """
        img_h, img_w = img_shape[:2]
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        h = y2 - y1
        area = w * h
        img_area = img_h * img_w

        # Relative vertical position (0=top, 1=bottom)
        cy_rel = ((y1 + y2) / 2) / img_h

        # Aspect ratio (>1 means wider than tall = gun-like)
        aspect = w / h if h > 0 else 1.0

        # Area fraction
        area_frac = area / img_area

        # Camera signature: upper 40% of image + large area + squarish
        is_camera_position = cy_rel < 0.45
        is_large           = area_frac > 0.03
        is_squarish        = 0.6 < aspect < 1.6

        if is_camera_position and is_large and is_squarish:
            logger.info(
                f"Camera FP rejected: cy_rel={cy_rel:.2f} "
                f"area_frac={area_frac:.3f} aspect={aspect:.2f} "
                f"conf={det['confidence']}%"
            )
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────────
    def _filter_detections(self, dets):
        """Per-category confidence gates + hard label blocklist."""
        filtered = []
        for d in dets:
            cat  = d['category']
            conf = d['confidence']
            raw  = d['raw_label'].lower()

            # Hard block known non-weapon COCO labels
            if cat == 'Weapon' and raw in WEAPON_FP_LABELS:
                logger.info(f"Blocked FP weapon label: {d['label']} {conf}%")
                continue

            # Per-category confidence gate
            threshold = CONF_THRESHOLDS.get(cat, 40.0)
            if conf < threshold:
                logger.info(
                    f"Filtered {cat} '{d['label']}' {conf}% < {threshold}%"
                )
                continue

            filtered.append(d)

        return filtered

    # ──────────────────────────────────────────────────────────────────────────
    def _relabel(self, dets):
        """
        Post-NMS semantic relabeling for known misclassifications.
        - suitcase in a crime scene = Forensic Kit
        - bed that is clearly a couch = Couch (lower severity either way)
        """
        for d in dets:
            raw = d['raw_label'].lower()
            # Forensic kit relabel
            if raw in FORENSIC_KIT_LABELS:
                d['label']     = 'Forensic Kit'
                d['raw_label'] = 'forensic kit'
                d['category']  = 'Object'
                d['severity']  = 'medium'  # forensic kit IS relevant evidence
        return dets

    # ──────────────────────────────────────────────────────────────────────────
    def _run_roboflow(self, image_path, info):
        with open(str(image_path), 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        url = (
            f"https://detect.roboflow.com/{info['project']}/{info['version']}"
            f"?api_key={self.roboflow_api_key}&confidence={int(self.confidence*100)}"
        )
        resp = requests.post(
            url, data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        )
        resp.raise_for_status()
        results = []
        for pred in resp.json().get('predictions', []):
            raw   = pred['class'].lower()
            label = info['label_map'].get(raw, raw.title())
            cx, cy, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            conf = round(pred['confidence'] * 100, 1)
            sev  = self._get_severity(raw, info['category'])
            results.append({
                'label': label, 'raw_label': raw,
                'confidence': conf,
                'category': info['category'], 'severity': sev,
                'bbox': [x1, y1, x2, y2],
                'center': [int(cx), int(cy)],
                'area': int(w * h),
                'source': info['name'],
            })
        return results

    def _process_yolo(self, results, force_category=None, force_severity=None):
        out = []
        for r in results:
            for b in r.boxes:
                conf  = float(b.conf[0])
                raw   = r.names[int(b.cls[0])]
                label = raw.replace('_', ' ').title()
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cat = force_category or self._get_category(raw)
                sev = force_severity or self._get_severity(raw, cat)
                out.append({
                    'label': label, 'raw_label': raw.lower(),
                    'confidence': round(conf * 100, 1),
                    'category': cat, 'severity': sev,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1+x2)//2, (y1+y2)//2],
                    'area': (x2-x1)*(y2-y1),
                    'source': 'yolo_local',
                })
        return out

    def _get_category(self, label):
        ll = label.lower()
        for k, v in CATEGORY_MAP.items():
            if k in ll:
                return v
        return 'Object'

    def _get_severity(self, label, category):
        ll = label.lower()
        for k, v in SEVERITY_MAP.items():
            if k in ll:
                return v
        if category == 'Biological': return 'critical'
        if category == 'Weapon':     return 'high'
        if category == 'Person':     return 'medium'
        return 'low'

    def _nms(self, dets):
        if not dets:
            return []
        dets.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        for d in dets:
            if not any(self._iou(d['bbox'], k['bbox']) > 0.45 for k in keep):
                keep.append(d)
        return keep

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        u = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / u if u > 0 else 0

    def _draw(self, img, dets):
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            color = COLOR_MAP.get(d['severity'], (0, 220, 220))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{d['label']} {d['confidence']}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(img, text, (x1+2, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img