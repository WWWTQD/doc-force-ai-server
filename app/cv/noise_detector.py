import cv2
import numpy as np


def estimate_noise(image: np.ndarray) -> dict:
    """A simple noise estimation using local variance."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # compute Laplacian var as proxy for noise
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = float(np.std(lap))
    # map to LOW/MEDIUM/HIGH
    if score < 5:
        level = "LOW"
    elif score < 20:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return {"score": score, "level": level}

