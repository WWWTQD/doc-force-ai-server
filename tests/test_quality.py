import numpy as np
from app.cv.blur_detector import detect_blur
from app.cv.brightness_detector import analyze_brightness


def test_blur_detector():
    # create a sharp image (checkerboard)
    img = np.zeros((200, 200), dtype="uint8")
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    res = detect_blur(img, threshold=10.0)
    assert isinstance(res.get("score"), float)


def test_brightness_detector():
    bright = np.full((100, 100, 3), 240, dtype="uint8")
    res = analyze_brightness(bright)
    assert res.get("category") == "bright"

