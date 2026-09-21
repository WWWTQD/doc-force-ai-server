"""
Image Preprocessor Module cho Doc Forge AI Worker.

Tiền xử lý ảnh tài liệu quét/chụp:
  1. Deskew (xoay thẳng văn bản bị nghiêng)
  2. Denoise (khử nhiễu muỗi/nếp gấp)
  3. CLAHE (tăng tương phản thích ứng)
"""

import logging
import math
import cv2
import numpy as np

LOG = logging.getLogger("preprocessor")


class ImagePreprocessor:
    """Pipeline tiền xử lý ảnh trước khi đưa vào OCR."""

    def __init__(
        self,
        enable_deskew: bool = True,
        enable_denoise: bool = True,
        enable_clahe: bool = True,
    ):
        self.enable_deskew = enable_deskew
        self.enable_denoise = enable_denoise
        self.enable_clahe = enable_clahe

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

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Phát hiện góc nghiêng và xoay thẳng văn bản.
        Sử dụng Hough Lines / Minimum Bounding Rectangle.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarize bằng Otsu
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Lấy tất cả các điểm pixel đen (chữ)
        pts = cv2.findNonZero(thresh)
        if pts is None or len(pts) < 10:
            return img

        # Tìm góc nghiêng của minAreaRect
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]

        # Chuẩn hóa góc angle về khoảng [-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        # Chỉ xoay nếu góc nghiêng đủ lớn (> 0.5 độ và < 30 độ)
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

    def process(self, image_bytes: bytes) -> tuple[np.ndarray, bytes]:
        """
        Thực hiện toàn bộ pipeline tiền xử lý:
        Decode -> Deskew -> Denoise -> CLAHE -> Encode bytes
        Trả về (cv2_image, processed_bytes)
        """
        img = self.decode_image(image_bytes)

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

        processed_bytes = self.encode_image(img)
        return img, processed_bytes
