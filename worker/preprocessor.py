"""
Image Preprocessor Module cho Doc Forge AI Worker.

Tiền xử lý ảnh tài liệu quét/chụp:
  1. Deskew (xoay thẳng văn bản bị nghiêng)
  2. Denoise (khử nhiễu muỗi/nếp gấp)
  3. Magic Color (khử bóng đổ, làm trắng nền giấy) — mặc định bật
     hoặc CLAHE (tăng tương phản thích ứng) khi tắt Magic Color
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
        enable_magic: bool = True,
    ):
        self.enable_deskew = enable_deskew
        self.enable_denoise = enable_denoise
        self.enable_clahe = enable_clahe
        self.enable_magic = enable_magic

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

    def deskew_with_matrix(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Phát hiện góc nghiêng và xoay thẳng văn bản.
        Sử dụng Hough Lines / Minimum Bounding Rectangle.
        Trả về (ảnh, ma trận 3x3 đã áp dụng) để truy vết toạ độ.
        """
        identity = np.eye(3, dtype=np.float64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarize bằng Otsu
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Lấy tất cả các điểm pixel đen (chữ)
        pts = cv2.findNonZero(thresh)
        if pts is None or len(pts) < 10:
            return img, identity

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
            return img, identity

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
        return rotated, np.vstack([M, [0.0, 0.0, 1.0]])

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """Phát hiện góc nghiêng và xoay thẳng văn bản (chỉ trả về ảnh)."""
        return self.deskew_with_matrix(img)[0]

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

    def magic_enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Khử bóng đổ, làm trắng nền giấy và làm chữ đậm nét (tương tự CamScanner).

        Ước lượng nền sáng của giấy bằng phép giãn (dilate) kênh L với kernel lớn hơn nét chữ,
        sau đó chuẩn hoá ảnh theo nền -> bóng tay/bóng đèn bị triệt tiêu, nền thành trắng.
        """
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Ước lượng nền trên ảnh thu nhỏ (nhanh gấp 100 lần, nền vẫn mượt mà chính xác)
        scale = 0.25
        small_l = cv2.resize(l_channel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw = small_l.shape[:2]
        k = max(7, int(min(sh, sw) / 60)) | 1

        bg_small = cv2.dilate(small_l, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        bg_small = cv2.medianBlur(bg_small, min(k, 51))
        background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Chia cho nền: pixel bằng nền -> 255 (trắng), chữ tối hơn nền -> giữ tương phản
        normalized = cv2.divide(l_channel, background, scale=255)

        # Kéo giãn: nền gần trắng -> trắng hẳn, chữ đậm hơn
        _, truncated = cv2.threshold(normalized, 235, 255, cv2.THRESH_TRUNC)
        stretched = cv2.normalize(truncated, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Giảm sắc độ màu nền (ám vàng/xanh) về trung tính
        a_neutral = cv2.addWeighted(a_channel, 0.5, np.full_like(a_channel, 128), 0.5, 0)
        b_neutral = cv2.addWeighted(b_channel, 0.5, np.full_like(b_channel, 128), 0.5, 0)

        result = cv2.merge([stretched, a_neutral, b_neutral])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def enhance(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Deskew -> Denoise -> Magic Color (hoặc CLAHE).
        Trả về (ảnh đã xử lý, ma trận 3x3 của phép biến đổi hình học đã áp dụng).
        """
        transform = np.eye(3, dtype=np.float64)

        if self.enable_deskew:
            try:
                img, transform = self.deskew_with_matrix(img)
            except Exception as e:
                LOG.warning("Lỗi khi deskew ảnh: %s", e)

        if self.enable_denoise:
            try:
                img = self.denoise(img)
            except Exception as e:
                LOG.warning("Lỗi khi denoise ảnh: %s", e)

        if self.enable_magic:
            try:
                img = self.magic_enhance(img)
            except Exception as e:
                LOG.warning("Lỗi khi magic enhance ảnh: %s", e)
        elif self.enable_clahe:
            try:
                img = self.enhance_clahe(img)
            except Exception as e:
                LOG.warning("Lỗi khi CLAHE ảnh: %s", e)

        return img, transform

    def process(self, image_bytes: bytes) -> tuple[np.ndarray, bytes]:
        """
        Thực hiện toàn bộ pipeline tiền xử lý:
        Decode -> Deskew -> Denoise -> Magic Color/CLAHE -> Encode bytes
        Trả về (cv2_image, processed_bytes)
        """
        img = self.decode_image(image_bytes)
        img, _ = self.enhance(img)
        processed_bytes = self.encode_image(img)
        return img, processed_bytes
