from typing import List, Dict
from app.schemas.defect import DefectModel
from app.core.config import settings


class DefectService:
    """Generates defect list based on analysis results."""

    def detect_defects(self, data: bytes, doc_info: Dict, quality: Dict, ocr_res: Dict = None) -> List[Dict]:
        defects = []

        # Blur / sharpness
        sharpness_score = quality.get("sharpness_score", 0.0)
        if sharpness_score < 30:
            defects.append({
                "code": "BLUR",
                "name": "Blurred image",
                "severity": "HIGH",
                "confidence": 0.9,
                "message": "Document image is too blurry",
                "recommendation": "Rescan the document with better focus",
            })

        # Brightness
        brightness = quality.get("brightness_score", 0.0)
        if brightness < 30:
            defects.append({
                "code": "LOW_BRIGHTNESS",
                "name": "Low brightness",
                "severity": "MEDIUM",
                "confidence": 0.85,
                "message": "Image appears too dark",
                "recommendation": "Increase lighting or adjust exposure",
            })
        elif brightness > 90:
            defects.append({
                "code": "HIGH_BRIGHTNESS",
                "name": "High brightness",
                "severity": "MEDIUM",
                "confidence": 0.8,
                "message": "Image appears too bright/overexposed",
                "recommendation": "Reduce lighting or exposure",
            })

        # Noise
        noise_level = quality.get("noise_score", 100.0)
        if noise_level < 30:
            defects.append({
                "code": "NOISE",
                "name": "Image noise",
                "severity": "LOW",
                "confidence": 0.7,
                "message": "Minor image noise detected",
                "recommendation": "Apply denoising",
            })

        # Document not detected
        if not doc_info.get("detected", False):
            defects.append({
                "code": "DOCUMENT_NOT_FOUND",
                "name": "Document not found",
                "severity": "HIGH",
                "confidence": 0.95,
                "message": "Could not detect a document region in the image",
                "recommendation": "Ensure the document is fully visible and try again",
            })

        # Low resolution
        if quality.get("resolution_score", 0.0) < 20:
            defects.append({
                "code": "LOW_RESOLUTION",
                "name": "Low resolution",
                "severity": "MEDIUM",
                "confidence": 0.7,
                "message": "Image resolution is low",
                "recommendation": "Scan at higher DPI",
            })

        # OCR low confidence
        if ocr_res:
            conf = ocr_res.get("confidence", 0.0)
            # conf is 0-1 inside OCRService
            if conf and conf < 0.5:
                defects.append({
                    "code": "OCR_LOW_CONFIDENCE",
                    "name": "Low OCR confidence",
                    "severity": "MEDIUM",
                    "confidence": float(conf),
                    "message": "OCR produced low confidence results",
                    "recommendation": "Improve image quality or use better OCR model",
                })

        return defects

