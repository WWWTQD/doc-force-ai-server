"""
Image Preprocessor Module cho Doc Forge AI Worker.

Tiền xử lý ảnh tài liệu quét/chụp:
  1. YOLO Document Detection & Crop ( worker/Model/best.pt )
  2. Deskew (xoay thẳng văn bản bị nghiêng)
  3. Denoise & CLAHE (tùy chọn)
"""

import os
import logging
import cv2
import numpy as np

LOG = logging.getLogger("preprocessor")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    LOG.warning("Thư viện ultralytics chưa cài đặt. Bỏ qua YOLO Document Detection.")


class ImagePreprocessor:
    """Pipeline tiền xử lý ảnh trước khi đưa vào U-Net và OCR."""

    def __init__(
        self,
        enable_yolo: bool = True,
        enable_deskew: bool = True,
        enable_denoise: bool = False,
        enable_clahe: bool = False,
        yolo_model_path: str = "Model/best.pt",
    ):
        self.enable_yolo = enable_yolo
        self.enable_deskew = enable_deskew
        self.enable_denoise = enable_denoise
        self.enable_clahe = enable_clahe
        self.yolo_model = None

        if self.enable_yolo and YOLO_AVAILABLE:
            candidates = [
                yolo_model_path,
                os.path.join(os.path.dirname(__file__), yolo_model_path),
                os.path.join(os.path.dirname(__file__), "Model", "best.pt"),
                os.path.join(os.path.dirname(__file__), "..", "worker", "Model", "best.pt"),
                "worker/Model/best.pt",
                "models/best.pt",
            ]
            for p in candidates:
                if os.path.isfile(p):
                    try:
                        self.yolo_model = YOLO(p)
                        LOG.info("Đã tải YOLO Document Detection từ: %s", p)
                        break
                    except Exception as e:
                        LOG.warning("Không thể tải YOLO từ %s: %s", p, e)

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Chuyển byte array thành OpenCV image (BGR)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể decode ảnh từ byte stream")
        return img

    def encode_image(self, img: np.ndarray, format: str = ".jpg") -> bytes:
        """Mã hóa OpenCV image thành bytes."""
        success, encoded_img = cv2.imencode(format, img)
        if not success:
            raise ValueError("Không thể encode ảnh thành bytes")
        return encoded_img.tobytes()

    def detect_and_crop(self, img: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """
        Phát hiện khung toàn trang tài liệu bằng YOLOv8 và crop trang.
        Trả về: (cropped_img, [x1, y1, x2, y2])
        Nếu không phát hiện hoặc không có model, trả về (img, [0, 0, w, h]).
        """
        h, w = img.shape[:2]
        if self.yolo_model is None:
            return img, [0, 0, w, h]

        try:
            results = self.yolo_model.predict(img, conf=0.35, verbose=False)
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                b = box.xyxy[0].tolist()
                x1 = max(0, min(w - 1, int(round(b[0]))))
                y1 = max(0, min(h - 1, int(round(b[1]))))
                x2 = max(x1 + 10, min(w, int(round(b[2]))))
                y2 = max(y1 + 10, min(h, int(round(b[3]))))

                # Chỉ crop nếu khung chiếm ít nhất 30% diện tích ảnh
                if (x2 - x1) * (y2 - y1) >= 0.3 * (w * h):
                    LOG.info(
                        "YOLO phát hiện document [%d, %d, %d, %d] (conf=%.2f) — tiến hành crop",
                        x1, y1, x2, y2, float(box.conf[0])
                    )
                    cropped = img[y1:y2, x1:x2].copy()
                    return cropped, [x1, y1, x2, y2]
        except Exception as e:
            LOG.warning("Lỗi trong quá trình YOLO detect: %s", e)

        return img, [0, 0, w, h]

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Phát hiện góc nghiêng và xoay thẳng văn bản.
        Sử dụng Minimum Bounding Rectangle trên ảnh nhị phân tạm thời.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        pts = cv2.findNonZero(thresh)
        if pts is None or len(pts) < 10:
            return img

        rect = cv2.minAreaRect(pts)
        angle = rect[-1]

        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        if abs(angle) < 0.5 or abs(angle) > 30.0:
            return img

        LOG.info("Phát hiện góc nghiêng %.2f độ — tiến hành xoay thẳng", angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Khử nhiễu giữ cạnh sắc nét bằng Bilateral Filter."""
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def enhance_clahe(self, img: np.ndarray) -> np.ndarray:
        """Tăng cường tương phản vùng bằng CLAHE trên kênh L (LAB)."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process(self, image_bytes: bytes) -> tuple[np.ndarray, list[int]]:
        """
        Pipeline tiền xử lý:
        1. Decode bytes -> BGR image
        2. YOLO Detect & Crop document (worker/Model/best.pt)
        3. Deskew xoay thẳng
        Trả về: (processed_cv_img, crop_bbox)
        """
        img = self.decode_image(image_bytes)
        crop_bbox = [0, 0, img.shape[1], img.shape[0]]

        if self.enable_yolo:
            try:
                img, crop_bbox = self.detect_and_crop(img)
            except Exception as e:
                LOG.warning("Lỗi khi crop bằng YOLO: %s", e)

        if self.enable_deskew:
            try:
                img = self.deskew(img)
            except Exception as e:
                LOG.warning("Lỗi khi deskew ảnh: %s", e)

        if self.enable_denoise:
            try:
                img = self.denoise(img)
            except Exception as e:
                LOG.warning("Lỗi khi denoise ảnh: %s", e)

        if self.enable_clahe:
            try:
                img = self.enhance_clahe(img)
            except Exception as e:
                LOG.warning("Lỗi khi CLAHE ảnh: %s", e)

        return img, crop_bbox
