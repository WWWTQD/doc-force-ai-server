import cv2
import numpy as np
import io
from PIL import Image
from typing import Dict
from app.cv import rotation_detector


class DocumentService:
    """Detects document region and provides helper to extract/correct perspective."""

    def detect_document_from_bytes(self, data: bytes) -> Dict:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    # order points consistently
                    ordered = self._order_points(pts)
                    # naive confidence based on area
                    area = cv2.contourArea(approx)
                    img_area = arr.shape[0] * arr.shape[1]
                    confidence = min(1.0, max(0.0, area / img_area))
                    # bounding box
                    x, y, w_box, h_box = cv2.boundingRect(approx)
                    bounding_box = {"x": int(x), "y": int(y), "width": int(w_box), "height": int(h_box)}
                    # attempt rotation detection via pytesseract if available
                    rotation_info = rotation_detector.detect_rotation(Image.fromarray(arr))
                    rotation = int(rotation_info.get("rotation", 0)) if rotation_info else 0
                    return {
                        "detected": True,
                        "confidence": float(confidence),
                        "corners": ordered.tolist(),
                        "bounding_box": bounding_box,
                        "rotation": rotation,
                        "skew_angle": 0.0,
                    }
            return {"detected": False, "confidence": 0.0, "corners": [], "rotation": 0, "skew_angle": 0.0}
        except Exception:
            return {"detected": False, "confidence": 0.0, "corners": [], "rotation": 0, "skew_angle": 0.0}

    def extract_document_bytes(self, data: bytes, doc_info: Dict) -> bytes:
        # If corners available, perform perspective transform and return JPEG bytes
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)
            if doc_info.get("detected") and doc_info.get("corners"):
                pts = np.array(doc_info["corners"], dtype="float32")
                # order points: top-left, top-right, bottom-right, bottom-left
                rect = self._order_points(pts)
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))
                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(arr, M, (maxWidth, maxHeight))
                pil = Image.fromarray(warped)
            else:
                pil = img
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            return buf.getvalue()
        except Exception:
            return data

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

