import cv2
import numpy as np


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> dict:
    """Detect blur using variance of Laplacian.

    image: numpy array (BGR or grayscale)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurred = var < threshold
    # confidence: map distance from threshold to 0-1
    conf = max(0.0, min(1.0, 1.0 - abs(var - threshold) / max(threshold, 1.0)))
    return {"is_blurred": is_blurred, "score": var, "threshold": threshold, "confidence": conf}

