import os
import time
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class YOLOService:
    """
    Singleton service managing the YOLO document detection model.
    Loads checkpoint into memory once and provides inference for document detection.
    """
    _instance: Optional["YOLOService"] = None
    _model = None
    _model_path: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOService, cls).__new__(cls)
        return cls._instance

    def _resolve_model_path(self) -> str:
        candidates = [
            settings.YOLO_MODEL_PATH,
            os.path.join(os.path.dirname(__file__), "..", "..", settings.YOLO_MODEL_PATH),
            os.path.join(os.getcwd(), settings.YOLO_MODEL_PATH),
            "worker/Model/best.pt",
            "models/best.pt",
            os.path.join(os.path.dirname(__file__), "..", "..", "worker", "Model", "best.pt"),
        ]
        for path in candidates:
            norm = os.path.normpath(os.path.abspath(path))
            if os.path.exists(norm) and os.path.isfile(norm):
                return norm
        return settings.YOLO_MODEL_PATH

    def load_model(self):
        if self._model is not None:
            return self._model

        model_path = self._resolve_model_path()
        self._model_path = model_path

        if not os.path.exists(model_path):
            logger.error("[YOLO] Checkpoint not found at: %s", model_path)
            raise FileNotFoundError(f"YOLO checkpoint not found at: {model_path}")

        try:
            from ultralytics import YOLO
            logger.info("[YOLO] Loading document detection model from: %s", model_path)
            t0 = time.perf_counter()
            self._model = YOLO(model_path)
            load_time = (time.perf_counter() - t0) * 1000.0
            logger.info("[YOLO] Model loaded successfully in %.1fms. Task: %s, Names: %s", 
                        load_time, getattr(self._model, "task", "unknown"), getattr(self._model, "names", {}))
            return self._model
        except Exception as e:
            logger.exception("[YOLO] Failed to load YOLO model: %s", e)
            self._model = None
            raise

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect_document(self, img: Image.Image | np.ndarray) -> Dict[str, Any]:
        """
        Run YOLO inference to detect the document bounding box.
        Returns structured detection result.
        """
        if self._model is None:
            self.load_model()

        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
            width, height = pil_img.size
        elif isinstance(img, Image.Image):
            pil_img = img
            width, height = pil_img.size
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        t_start = time.perf_counter()
        try:
            results = self._model.predict(
                pil_img, 
                conf=settings.YOLO_CONF_THRESHOLD, 
                verbose=False
            )
            inference_time_ms = (time.perf_counter() - t_start) * 1000.0

            valid_detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    valid_detections.append({
                        "class_id": cls_id,
                        "class_name": self._model.names.get(cls_id, "document"),
                        "confidence": conf,
                        "xyxy": xyxy
                    })

            # Sort by confidence descending
            valid_detections.sort(key=lambda d: d["confidence"], reverse=True)

            if not valid_detections:
                logger.info("[YOLO] Inference completed in %.1fms: 0 document detected (threshold=%.2f). Checkpoint=%s",
                            inference_time_ms, settings.YOLO_CONF_THRESHOLD, self._model_path)
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "bounding_box": None,
                    "corners": [],
                    "inference_time_ms": round(inference_time_ms, 2),
                    "num_detections": 0,
                    "checkpoint": self._model_path,
                }

            best = valid_detections[0]
            x1, y1, x2, y2 = best["xyxy"]
            # Clamp to image boundaries
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            w_box = x2 - x1
            h_box = y2 - y1

            bounding_box = {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "width": int(round(w_box)),
                "height": int(round(h_box)),
            }

            corners = [
                [int(round(x1)), int(round(y1))],  # Top-left
                [int(round(x2)), int(round(y1))],  # Top-right
                [int(round(x2)), int(round(y2))],  # Bottom-right
                [int(round(x1)), int(round(y2))],  # Bottom-left
            ]

            logger.info("[YOLO] Document detected! Checkpoint=%s, Box=%s, Conf=%.4f, Total detections=%d, Time=%.1fms",
                        self._model_path, bounding_box, best["confidence"], len(valid_detections), inference_time_ms)

            return {
                "detected": True,
                "confidence": round(best["confidence"], 4),
                "bounding_box": bounding_box,
                "corners": corners,
                "inference_time_ms": round(inference_time_ms, 2),
                "num_detections": len(valid_detections),
                "checkpoint": self._model_path,
            }

        except Exception as e:
            inference_time_ms = (time.perf_counter() - t_start) * 1000.0
            logger.exception("[YOLO] Inference error after %.1fms: %s", inference_time_ms, e)
            return {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
                "corners": [],
                "inference_time_ms": round(inference_time_ms, 2),
                "num_detections": 0,
                "error": str(e),
                "checkpoint": self._model_path,
            }
