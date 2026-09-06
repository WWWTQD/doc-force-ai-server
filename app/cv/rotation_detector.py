import pytesseract
from PIL import Image


def detect_rotation(pil_image: Image.Image) -> dict:
    """Try to detect orientation via pytesseract's osd if available."""
    try:
        osd = pytesseract.image_to_osd(pil_image)
        # parse angle: rotate: 90
        angle = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                angle = int(line.split(":")[-1].strip())
        confidence = 0.9
        return {"rotation": angle, "confidence": confidence}
    except Exception:
        return {"rotation": 0, "confidence": 0.0}

