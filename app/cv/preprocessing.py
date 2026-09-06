import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    (h, w) = img.shape[:2]
    if width and not height:
        r = width / float(w)
        dim = (width, int(h * r))
    elif height and not width:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        return img
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) if img.ndim == 3 else cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

