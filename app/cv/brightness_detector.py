import cv2
import numpy as np
from app.core.config import settings


def analyze_brightness(image: np.ndarray) -> dict:
    """Return brightness score 0-100 and category"""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    mean = float(np.mean(gray))
    score = (mean / 255.0) * 100.0
    # Return human-friendly categories to match tests: 'dark', 'normal', 'bright'
    low_thr = getattr(settings, "LOW_BRIGHTNESS_THRESHOLD", 70)
    high_thr = getattr(settings, "HIGH_BRIGHTNESS_THRESHOLD", 200)
    if mean < low_thr:
        category = "dark"
    elif mean > high_thr:
        category = "bright"
    else:
        category = "normal"
    return {"score": score, "mean": mean, "category": category}

