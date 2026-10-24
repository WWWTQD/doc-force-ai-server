import cv2
import numpy as np
import io
import logging
from PIL import Image
from typing import Dict, Any
from app.cv import rotation_detector
from app.services.yolo_service import YOLOService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Detects document region using deep learning YOLO model and performs perspective extraction.
    """

    def __init__(self):
        self.yolo_service = YOLOService()

    def detect_document_from_bytes(self, data: bytes) -> Dict[str, Any]:
        """
        Detects document using YOLO model as primary detector.
        If YOLO does not detect document, returns detected=False without OpenCV fallback.
        """
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)

            # Primary detector: YOLO model
            yolo_result = self.yolo_service.detect_document(img)

            if yolo_result.get("detected") and yolo_result.get("corners"):
                corners = yolo_result["corners"]
                bounding_box = yolo_result["bounding_box"]
                confidence = yolo_result["confidence"]

                # Rotation detection
                rotation = 0
                try:
                    rotation_info = rotation_detector.detect_rotation(img)
                    rotation = int(rotation_info.get("rotation", 0)) if rotation_info else 0
                except Exception as ex:
                    logger.debug("Rotation detection skipped: %s", ex)

                return {
                    "detected": True,
                    "confidence": float(confidence),
                    "corners": corners,
                    "bounding_box": bounding_box,
                    "rotation": rotation,
                    "skew_angle": 0.0,
                    "model": "YOLOv8-Doc",
                    "checkpoint": yolo_result.get("checkpoint"),
                    "inference_time_ms": yolo_result.get("inference_time_ms", 0.0),
                }

            # No document detected by YOLO
            logger.info("Document not detected by YOLO in uploaded image.")
            return {
                "detected": False,
                "confidence": 0.0,
                "corners": [],
                "bounding_box": None,
                "rotation": 0,
                "skew_angle": 0.0,
                "model": "YOLOv8-Doc",
                "checkpoint": yolo_result.get("checkpoint"),
                "inference_time_ms": yolo_result.get("inference_time_ms", 0.0),
            }

        except Exception as e:
            logger.exception("Error during document detection: %s", e)
            return {
                "detected": False,
                "confidence": 0.0,
                "corners": [],
                "bounding_box": None,
                "rotation": 0,
                "skew_angle": 0.0,
            }

    def extract_document_bytes(self, data: bytes, doc_info: Dict) -> bytes:
        """
        Crops and applies perspective transformation on the document region detected by YOLO.
        """
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)

            if doc_info.get("detected") and doc_info.get("corners"):
                pts = np.array(doc_info["corners"], dtype="float32")
                rect = self._order_points(pts)
                (tl, tr, br, bl) = rect

                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))

                if maxWidth > 10 and maxHeight > 10:
                    dst = np.array(
                        [
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1],
                        ],
                        dtype="float32",
                    )
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(arr, M, (maxWidth, maxHeight))
                    pil = Image.fromarray(warped)
                else:
                    pil = img
            else:
                pil = img

            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=95)
            return buf.getvalue()

        except Exception as e:
            logger.warning("Failed to extract document bytes: %s. Returning original image.", e)
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
