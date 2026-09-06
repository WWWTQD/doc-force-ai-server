import cv2
import numpy as np


def detect_skew(image: np.ndarray) -> dict:
    """Detect skew angle (in degrees) using Hough or minAreaRect on edges."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size == 0:
        return {"angle": 0.0, "is_skewed": False, "confidence": 0.0}
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # convert to more intuitive angle
    if angle < -45:
        angle = 90 + angle
    is_skewed = abs(angle) > 1.0
    confidence = min(1.0, max(0.0, abs(angle) / 45.0))
    return {"angle": float(angle), "is_skewed": is_skewed, "confidence": confidence}

