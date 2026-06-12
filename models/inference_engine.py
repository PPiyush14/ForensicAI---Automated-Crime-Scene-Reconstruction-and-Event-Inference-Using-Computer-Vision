"""
Forensic Inference Engine
AI backend: Groq API (FREE) — llama-4-scout-17b-16e-instruct with vision
Get a free key at: https://console.groq.com (no credit card needed)

Falls back gracefully to rule-based analysis if no key is set.
"""
import os
import base64
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_rule_based_report(detections):
    """Rule-based forensic report — always runs as foundation."""
    labels_lower = [d['raw_label'].lower() for d in detections]
    has_blood  = any('blood' in l or 'stain' in l or 'wound' in l for l in labels_lower)
    has_gun    = any(kw in l for l in labels_lower for kw in ['gun', 'pistol', 'rifle', 'firearm', 'weapon'])
    has_knife  = any(kw in l for l in labels_lower for kw in ['knife', 'blade', 'sharp'])
    has_person = any('person' in l or 'body' in l or 'victim' in l for l in labels_lower)
    has_rope   = any('rope' in l or 'cord' in l for l in labels_lower)
    has_shell  = any('shell' in l or 'casing' in l or 'bullet' in l for l in labels_lower)

    events = []
    if has_blood and has_gun:
        events.append({'event': 'Shooting incident',   'confidence': 85, 'evidence': ['BLOOD', 'GUN']})
    if has_shell:
        events.append({'event': 'Gunshot discharged',  'confidence': 80, 'evidence': ['SHELL_CASING']})
    if has_blood and has_knife:
        events.append({'event': 'Stabbing incident',   'confidence': 80, 'evidence': ['BLOOD', 'KNIFE']})
    if has_blood and not has_gun and not has_knife:
        events.append({'event': 'Assault with injury', 'confidence': 55, 'evidence': ['BLOOD']})
    if has_gun and not has_blood:
        events.append({'event': 'Armed confrontation', 'confidence': 60, 'evidence': ['GUN']})
    if has_rope and has_person:
        events.append({'event': 'Restraint / assault', 'confidence': 65, 'evidence': ['ROPE', 'PERSON']})
    if has_person and has_blood:
        events.append({'event': 'Violent assault',     'confidence': 70, 'evidence': ['PERSON', 'BLOOD']})
    if not events and detections:
        events.append({'event': 'Suspicious scene',    'confidence': 35,
                       'evidence': [d['label'].upper() for d in detections[:3]]})

    sev_set = {d['severity'] for d in detections}
    if 'critical' in sev_set:
        threat = 'CRITICAL'
    elif 'high' in sev_set:
        threat = 'HIGH'
    elif detections:
        threat = 'MEDIUM'
    else:
        threat = 'LOW'

    return {'threat_level': threat, 'possible_events': events}


def _image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _get_media_type(image_path):
    ext = Path(image_path).suffix.lower()
    return {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.webp': 'image/webp',
            '.bmp': 'image/bmp'}.get(ext, 'image/jpeg')


class ForensicInferenceEngine:
    """
    Forensic AI engine powered by Groq (free tier).
    Model: llama-4-scout-17b-16e-instruct — supports image vision.
    Fallback: rule-based analysis when no key configured.
    """

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if self.api_key:
            logger.info(f"Groq forensic engine ready (model: {self.GROQ_MODEL})")
        else:
            logger.warning("GROQ_API_KEY not set — rule-based only. Free key: https://console.groq.com")

    def analyze_scene(self, image_path, detections):
        rule_report = _build_rule_based_report(detections)

        ai_result = {'analysis': None, 'success': False, 'error': None, 'model': None}

        if self.api_key:
            try:
                ai_result = self._groq_analyze(image_path, detections)
            except Exception as e:
                logger.error(f"Groq analysis error: {e}")
                ai_result = {'analysis': None, 'success': False, 'error': str(e), 'model': self.GROQ_MODEL}
        else:
            ai_result['error'] = (
                "AI analysis not configured. Add GROQ_API_KEY to .env — "
                "free key at https://console.groq.com (no credit card needed)."
            )

        stats = {
            'total_objects':  len(detections),
            'critical_count': sum(1 for d in detections if d['severity'] == 'critical'),
            'high_count':     sum(1 for d in detections if d['severity'] == 'high'),
            'medium_count':   sum(1 for d in detections if d['severity'] == 'medium'),
            'low_count':      sum(1 for d in detections if d['severity'] == 'low'),
            'categories':     list({d['category'] for d in detections}),
        }

        return {
            'rule_based_assessment': rule_report,
            'ai_analysis':           ai_result,
            'statistics':            stats,
            'detections':            detections,
        }

    def _groq_analyze(self, image_path, detections):
        img_b64    = _image_to_base64(image_path)
        media_type = _get_media_type(image_path)

        det_text = "\n".join(
            f"  - {d['label']} ({d['confidence']}% conf) | Severity: {d['severity'].upper()} | Category: {d['category']}"
            for d in detections
        ) or "  None detected"

        prompt = f"""You are a senior forensic pathologist and crime scene reconstruction expert with 20 years of experience.
Analyze this crime scene image and provide a detailed professional forensic report.

YOLO object detection found:
{det_text}

Provide your analysis using EXACTLY these section headers:

## SCENE OVERVIEW
Brief description of what you observe (2-3 sentences).

## EVIDENCE ASSESSMENT
Assess accuracy of each detected item. Flag likely false positives (e.g. cameras misidentified as guns).

## RECONSTRUCTION
Likely sequence of events. Be specific about timing and causality.

## WOUND PATTERN ANALYSIS
If blood/wounds visible: describe pattern, location, cause. If none visible, state so.

## INVESTIGATIVE LEADS
5 specific immediate next steps for investigators (numbered list).

## PHYSICAL EVIDENCE PRIORITY
Top 3 evidence items to process first and why.

Be technical, professional, concise. Use forensic terminology."""

        payload = {
            "model": self.GROQ_MODEL,
            "max_tokens": 1500,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        resp = requests.post(self.GROQ_API_URL, json=payload, headers=headers, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            return {
                'analysis': data['choices'][0]['message']['content'],
                'success':  True,
                'error':    None,
                'model':    self.GROQ_MODEL
            }
        else:
            err = resp.json().get('error', {}).get('message', resp.text)
            raise RuntimeError(f"Groq API {resp.status_code}: {err}")
