import io
from typing import Dict
from PIL import Image
import numpy as np
import cv2
from app.core.config import settings


class QualityService:
    """Analyzes image quality and returns scores and level."""

    def analyze_bytes(self, data: bytes, doc_info: Dict) -> Dict:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)

            # resolution score (based on pixels)
            h, w = arr.shape[:2]
            megapixels = (w * h) / 1e6
            res_score = min(100.0, megapixels / 2.0 * 100.0)

            # brightness score
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            mean_brightness = float(gray.mean())
            # map 0-255 to 0-100
            brightness_score = (mean_brightness / 255.0) * 100.0

            # contrast score (std of grayscale)
            contrast_score = float(gray.std()) / 128.0 * 100.0
            contrast_score = max(0.0, min(100.0, contrast_score))

            # sharpness via Laplacian variance
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(lap.var())
            # normalize roughly
            sharpness_score = max(0.0, min(100.0, (sharpness / settings.BLUR_THRESHOLD) * 100.0))

            # noise: simplistic estimation (high frequency content)
            noise = float(np.mean(np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0))))
            noise_score = max(0.0, min(100.0, 100.0 - (noise / 50.0 * 100.0)))

            # document detection score
            doc_score = doc_info.get("confidence", 0.0) * 100.0

            # combine with weights
            score = (0.25 * res_score + 0.2 * brightness_score + 0.15 * contrast_score + 0.25 * sharpness_score + 0.15 * noise_score) / 1.0
            score = max(0.0, min(100.0, score))

            # level
            if score >= 90:
                level = "EXCELLENT"
            elif score >= 75:
                level = "GOOD"
            elif score >= 50:
                level = "FAIR"
            elif score >= 25:
                level = "POOR"
            else:
                level = "VERY_POOR"

            return {
                "score": float(score),
                "level": level,
                "resolution_score": res_score,
                "brightness_score": brightness_score,
                "contrast_score": contrast_score,
                "sharpness_score": sharpness_score,
                "noise_score": noise_score,
            }
        except Exception:
            return {"score": 0.0, "level": "VERY_POOR"}

