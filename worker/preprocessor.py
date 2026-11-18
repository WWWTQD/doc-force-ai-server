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
        """Mã hóa OpenCV image thành bytes (JPEG quality 98 để giữ nguyên độ sắc nét gốc)."""
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 98] if format.lower() in (".jpg", ".jpeg") else []
        success, encoded_img = cv2.imencode(format, img, params)
        if not success:
            raise ValueError("Không thể encode ảnh thành bytes")
        return encoded_img.tobytes()

    def clean_handwriting_and_book_gutter(self, img: np.ndarray) -> np.ndarray:
        """
        1. Khử chữ viết tay bằng bút bi xanh/tím/đỏ, dấu ngoặc nháp và ghi chú lề ("chữ nháp ghi vào")
           bằng cách so sánh độ lệch sắc độ (chromaticity) với nền giấy cục bộ, giữ nguyên chữ in đen/nâu trung tính.
           Không chạy xóa mực bút bi trên ảnh nhỏ (min(h, w) < 900) để bảo vệ tuyệt đối nét chữ in trên giấy cổ.
        2. Phát hiện rãnh gáy sách (book spine gutter) khi chụp trang sách mở và làm trắng phần chữ bị cắt của trang kế bên.
        """
        if img is None or img.ndim != 3:
            return img
        out = img.copy()
        h, w = out.shape[:2]

        # Bảo vệ tuyệt đối ảnh kích thước nhỏ / tài liệu lịch sử độ phân giải thấp (min(h, w) < 900)
        if min(h, w) < 900:
            return out

        # Bước 1: Ước lượng nền giấy cục bộ theo tỉ lệ thích ứng theo độ phân giải ảnh
        scale = min(0.5, 450.0 / float(min(h, w)))
        small = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        k_med = max(15, min(51, int(round(61 * scale)) | 1))
        bg_small = cv2.medianBlur(small, k_med)
        bg_small = cv2.GaussianBlur(bg_small, (15, 15), 0)
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

        b_ch = out[:, :, 0].astype(np.int16)
        g_ch = out[:, :, 1].astype(np.int16)
        r_ch = out[:, :, 2].astype(np.int16)
        bg_b = bg[:, :, 0].astype(np.int16)
        bg_g = bg[:, :, 1].astype(np.int16)
        bg_r = bg[:, :, 2].astype(np.int16)

        # Độ hấp thụ ánh sáng so với nền giấy cục bộ (chữ tối hơn nền -> giá trị dương)
        drop_b = bg_b - b_ch
        drop_g = bg_g - g_ch
        drop_r = bg_r - r_ch
        luma_drop = 0.114 * drop_b + 0.587 * drop_g + 0.299 * drop_r

        # Chỉ xóa mực bút bi khi sắc độ xanh dương/đỏ/tím thực sự rõ rệt;
        # tuyệt đối không xóa nhầm mực in ngả nâu/vàng (có R và G gần nhau) trên nền giấy cổ.
        blue_pen = (drop_r >= 14) & ((drop_r - drop_b) >= 14) & ((b_ch - r_ch) >= 14)
        purple_pen = (drop_g >= 15) & ((drop_g - drop_b) >= 14) & ((b_ch - g_ch) >= 13)
        red_pen = (drop_b >= 24) & ((drop_b - drop_r) >= 20) & ((r_ch - b_ch) >= 26) & ((r_ch - g_ch) >= 18)

        # Bảo vệ tuyệt đối mọi pixel chữ in đen/xám/nâu trung tính (kể cả vùng chữ in hơi mờ ở rìa trang)
        min_rgb = np.minimum(np.minimum(b_ch, g_ch), r_ch)
        neutral_print = (
            ((drop_b >= 25) & (drop_g >= 25) & (drop_r >= 25) & (np.abs(drop_r - drop_b) <= 13))
            | ((min_rgb < 65) & (np.abs(r_ch - b_ch) < 20))
        )
        pen_mask = ((blue_pen | purple_pen | red_pen) & (~neutral_print)).astype(np.uint8) * 255

        if int(pen_mask.sum()) > 0:
            dil_k = 5 if min(h, w) >= 1600 else 3
            pen_dilated = cv2.dilate(pen_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k)), iterations=1)
            replace_mask = (pen_dilated > 0) & (~neutral_print)
            out[replace_mask] = bg[replace_mask]

        # Bước 2: Phát hiện và làm trắng phần trang sách kế bên qua rãnh gáy sách (x ∈ [6%..21%] chiều rộng)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        ink = ((bg_gray.astype(np.int16) - gray.astype(np.int16)) >= 28).astype(np.uint8)
        y0, y1 = int(h * 0.12), int(h * 0.88)
        band = ink[y0:y1, :]
        col_density = band.mean(axis=0)  # tỉ lệ pixel mực trên mỗi cột X

        # Tìm rãnh trắng dọc (gutter valley) trong khoảng [8% .. 23%] nơi có chữ của trang bên trái (x < 9%)
        # Chỉ áp dụng khi mép trái thực sự là nền giấy trang sách kề bên (cả median bg lẫn percentile 25 ảnh xám thô đều sáng, tránh nền bàn/vải tối)
        left_slice_col = col_density[int(w * 0.01):int(w * 0.09)]
        left_edge_ink = float(left_slice_col.mean()) if w >= 650 else 0.0
        left_bg_med = float(np.median(bg_gray[y0:y1, int(w * 0.01):int(w * 0.09)])) if w >= 650 else 0.0
        left_raw_p25 = float(np.percentile(gray[y0:y1, int(w * 0.01):int(w * 0.09)], 25)) if w >= 650 else 0.0
        if 0.012 <= left_edge_ink <= 0.30 and left_bg_med >= 120.0 and left_raw_p25 >= 105.0:
            g_lo, g_hi = int(w * 0.08), int(w * 0.23)
            if g_hi > g_lo + 20:
                win = max(15, int(w * 0.015)) | 1
                smooth_col = np.convolve(col_density, np.ones(win) / win, mode="same")
                gutter_x = g_lo + int(np.argmin(smooth_col[g_lo:g_hi]))
                if smooth_col[gutter_x] <= 0.012:
                    strip_p25 = float(np.percentile(gray[y0:y1, :max(1, gutter_x - 8)], 25))
                    right_body_ink = float(col_density[gutter_x + 20:min(w, gutter_x + int(w * 0.45))].mean())
                    if strip_p25 >= 105.0 and right_body_ink >= 0.025:
                        out[:, :gutter_x] = bg[:, :gutter_x]
                        LOG.info("  ✓ Đã lọc sạch lề trang kế bên tại gáy sách (x=0..%d)", gutter_x)

        return out

    def deskew_with_matrix(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Giữ nguyên hệ trục khung trang gốc (ma trận đơn vị 3x3) vì bộ cắt dòng tứ giác _rotate_crop
        của OCREngine đã nắn thẳng từng dòng chữ riêng biệt với độ sắc nét tối đa.
        """
        return img, np.eye(3, dtype=np.float64)

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """Phát hiện góc nghiêng và xoay thẳng văn bản (chỉ trả về ảnh)."""
        return self.deskew_with_matrix(img)[0]

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Khử nhiễu giữ cạnh sắc nét bằng Bilateral Filter tối ưu tốc độ (d=5)."""
        return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    def enhance_clahe(self, img: np.ndarray) -> np.ndarray:
        """Tăng cường tương phản vùng bằng CLAHE trên kênh L (LAB)."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def neutralize_highlighter_luma(img_or_gray: np.ndarray) -> np.ndarray:
        """
        Khử màu bút dạ quang (vàng, xanh lá, hồng, cam, xanh dương nhạt) có độ bão hòa màu cao
        về mức kênh sáng nhất trước khi chuyển sang Grayscale, giữ nguyên chữ in đen/xám trung tính.
        """
        if img_or_gray is None:
            return img_or_gray
        if img_or_gray.ndim == 2:
            return img_or_gray
        b, g, r = cv2.split(img_or_gray)
        max_c = np.maximum(np.maximum(b, g), r).astype(np.float32)
        min_c = np.minimum(np.minimum(b, g), r).astype(np.float32)
        sat = max_c - min_c
        luma = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)
        # Vùng bút dạ quang màu có độ bão hòa (sat >= 16) và kênh sáng nhất (max_c >= 120)
        hl_weight = np.clip((sat - 16.0) / 32.0, 0.0, 1.0) * np.clip((max_c - 120.0) / 50.0, 0.0, 1.0)
        gray = np.clip(luma + hl_weight * (max_c - luma), 0.0, 255.0).astype(np.uint8)
        return gray

    @staticmethod
    def estimate_paper_background(gray_or_l: np.ndarray, mode: str = "dilate") -> np.ndarray:
        """
        Ước lượng nền sáng đa tỉ lệ (Multi-scale Background Estimation):
        - Chỉ thu nhỏ khi min(h, w) > 1000 (ảnh nhỏ giữ nguyên 100% độ phân giải gốc).
        - Kết hợp:
          1. Macro background (~131-171px tương đương) cho vùng đổ bóng lớn / biến thiên ánh sáng.
          2. Tight stroke-scale closing (k_tx ~ 27px, k_ty ~ 9px + erode viền 3x5) chỉ đóng qua nét chữ mảnh
             nhưng bám sát nền của từng dải bút dạ quang (1 dòng hoặc 6 dòng liên tiếp) kể cả trên ảnh chụp
             sách photocopy/grayscale, giúp phép chia nền biến vùng tô dạ quang thành trắng tinh 255.
        """
        if gray_or_l.ndim == 3:
            gray_or_l = ImagePreprocessor.neutralize_highlighter_luma(gray_or_l)
        h, w = gray_or_l.shape[:2]
        min_dim = min(h, w)
        scale = 1.0 if min_dim <= 1000 else max(0.35, 1000.0 / float(min_dim))
        if scale < 1.0:
            small = cv2.resize(gray_or_l, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = gray_or_l

        # 1. Macro background kernel (~131-171px quy đổi)
        k_macro_orig = max(45, min(171, (min_dim // 10) | 1))
        k_macro = max(25, int(round(k_macro_orig * scale)) | 1)
        se_macro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_macro, k_macro))
        if mode == "close":
            bg_macro = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se_macro)
        else:
            bg_macro = cv2.dilate(small, se_macro)
        bg_macro = cv2.medianBlur(bg_macro, min(k_macro, 51) | 1)
        bg_macro = cv2.GaussianBlur(bg_macro, (0, 0), sigmaX=max(4.0, k_macro * 0.25))

        # 2. Tight stroke-scale closing cho ảnh trung bình/lớn (min_dim >= 900) để khử 100% dải bút dạ quang
        if min_dim >= 900:
            k_tx = max(15, int(round(27 * scale)) | 1)
            k_ty = max(7, int(round(9 * scale)) | 1)
            se_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_tx, k_ty))
            bg_close_raw = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se_tight)
            # Mở rộng nhẹ vùng nền tối của dải dạ quang ra sát mép trên/dưới (3x5) để xóa sạch viền đọng mực dạ quang
            k_rim_y = max(3, int(round(5 * scale)) | 1)
            bg_close_eroded = cv2.erode(
                bg_close_raw,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_rim_y)),
            )
            bg_close_smooth = cv2.GaussianBlur(
                cv2.medianBlur(bg_close_eroded, max(5, int(round(7 * scale)) | 1)),
                (0, 0),
                sigmaX=max(1.2, 2.2 * scale),
            )
            bg_tight = np.minimum(bg_close_eroded, bg_close_smooth)
            bg_work = np.minimum(bg_macro, bg_tight)
        else:
            bg_work = bg_macro

        if scale < 1.0:
            return cv2.resize(bg_work, (w, h), interpolation=cv2.INTER_CUBIC)
        return bg_work

    @staticmethod
    def soft_normalize_luma(gray_or_l: np.ndarray, mode: str = "dilate") -> tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn hóa chiếu sáng mềm đa tỉ lệ + khử bút dạ quang nhiều dòng + lọc thấu mực mặt sau (Hysteresis
        Bleed-Through Suppression) và trả về (soft_norm_float32, background_float32).
        """
        if gray_or_l.ndim == 3:
            gray_in = ImagePreprocessor.neutralize_highlighter_luma(gray_or_l)
        else:
            gray_in = gray_or_l
        h, w = gray_in.shape[:2]
        min_dim = min(h, w)
        bg = ImagePreprocessor.estimate_paper_background(gray_in, mode=mode).astype(np.float32)
        bg_safe = np.maximum(bg, 25.0)
        l_f = gray_in.astype(np.float32)
        soft_norm = np.clip((l_f / bg_safe) * 255.0, 0.0, 255.0)

        # Hysteresis Bleed-Through & Highlighter-Rim Suppression trên ảnh tài liệu trung bình/lớn:
        # Chữ mặt trước luôn có lõi mực đậm (local_core_min <= 125) trong lân cận sát nét chữ (k_core ~ 15-19px),
        # trong khi chữ thấu mực mặt sau (kể cả nằm xen giữa 2 từ mặt trước) hoặc viền bút dạ quang chỉ có
        # độ xám nhạt-trung bình (soft_norm >= 142) và không có lõi mực đen mặt trước.
        if max(h, w) >= 900 and float(np.percentile(soft_norm, 0.5)) < 135.0:
            sn_u8 = soft_norm.astype(np.uint8)
            k_core = max(11, min(19, (min_dim // 110) | 1))
            local_core_min = cv2.erode(
                sn_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core)),
            ).astype(np.float32)
            ghost_w = (
                np.clip((soft_norm - 142.0) / 26.0, 0.0, 1.0)
                * np.clip((local_core_min - 125.0) / 18.0, 0.0, 1.0)
            )
            soft_norm = soft_norm + ghost_w * (255.0 - soft_norm)

        return soft_norm, bg_safe

    @staticmethod
    def compute_stroke_protection(soft_norm: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trả về (stroke_conf, local_drop, blackhat) đo độ tương phản nét chữ tần số cao cục bộ (15x15 - 21x21)
        so với nền sát cạnh để bảo vệ 100% nét chữ mặt trước mà KHÔNG BAO GIỜ giữ lại mảng tô dạ quang
        hay bóng chữ thấu mực mặt sau.
        """
        min_dim = min(h, w)
        scale_loc = 1.0 if min_dim <= 1500 else max(0.4, 1000.0 / float(min_dim))
        norm_u8 = np.clip(soft_norm, 0.0, 255.0).astype(np.uint8)
        work_u8 = (
            cv2.resize(norm_u8, None, fx=scale_loc, fy=scale_loc, interpolation=cv2.INTER_AREA)
            if scale_loc < 1.0
            else norm_u8
        )
        # Chỉ dùng cửa sổ cục bộ hẹp (13x13 - 21x21) cho nét chữ mảnh tần số cao
        k_loc = max(13, min(21, int(round((min_dim // 55) * scale_loc)) | 1))
        loc_bg_w = cv2.morphologyEx(
            work_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_loc, k_loc)),
        )
        k_bh = max(7, min(13, int(round((min_dim // 80) * scale_loc)) | 1))
        bh_w = cv2.morphologyEx(
            work_u8,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bh, k_bh)),
        )
        if scale_loc < 1.0:
            loc_bg = cv2.resize(loc_bg_w, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            bh = cv2.resize(bh_w, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            loc_bg = loc_bg_w.astype(np.float32)
            bh = bh_w.astype(np.float32)

        local_drop = np.maximum(0.0, loc_bg - soft_norm)
        core_ink = np.clip((185.0 - soft_norm) / 34.0, 0.0, 1.0) * np.clip((local_drop - 14.0) / 18.0, 0.0, 1.0)
        sharp_stroke = np.clip((local_drop - 26.0) / 26.0, 0.0, 1.0) * np.clip((228.0 - soft_norm) / 22.0, 0.0, 1.0)
        bh_stroke = np.clip((bh - 22.0) / 24.0, 0.0, 1.0) * np.clip((228.0 - soft_norm) / 22.0, 0.0, 1.0)
        stroke_conf = np.clip(np.maximum(core_ink, np.maximum(sharp_stroke, bh_stroke)), 0.0, 1.0)

        # Trên ảnh lớn có chữ đậm mặt trước, loại bỏ bảo vệ đối với các bóng mờ thấu mực không có lõi đen
        if max(h, w) >= 900 and float(np.percentile(soft_norm, 0.5)) < 135.0:
            k_core = max(11, min(19, (min_dim // 110) | 1))
            local_core_min = cv2.erode(
                norm_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core)),
            ).astype(np.float32)
            has_front_core = np.clip((134.0 - local_core_min) / 18.0, 0.0, 1.0)
            stroke_conf = stroke_conf * np.maximum(has_front_core, np.clip((136.0 - soft_norm) / 20.0, 0.0, 1.0))

        return stroke_conf, local_drop, bh

    def magic_enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Khử bóng đổ, làm trắng nền giấy và bảo vệ nét chữ đậm nét (tương tự CamScanner,
        an toàn tuyệt đối cho ảnh nhỏ / giấy cổ nhờ ước lượng nền thích ứng và Stroke Darkness Protection).
        """
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_f = self.neutralize_highlighter_luma(img).astype(np.float32) if img.ndim == 3 else l_channel.astype(np.float32)
        norm_f, _ = self.soft_normalize_luma(img if img.ndim == 3 else l_channel, mode="dilate")
        stroke_conf, local_drop, _ = self.compute_stroke_protection(norm_f, h, w)

        # Nâng vùng nền giấy / vết ố nhẹ (>= 204 khi không có nét chữ) lên trắng sạch 255,
        # trong khi các pixel nét chữ (stroke_conf > 0) được giữ nguyên độ tối tuyệt đối
        bg_clean = np.where(
            norm_f >= 204.0,
            np.clip(204.0 + ((norm_f - 204.0) / 26.0) * 51.0, 0.0, 255.0),
            norm_f,
        )

        # Stroke Darkness Protection: giữ nguyên và làm đậm nét chữ thuộc vùng tương phản cục bộ rõ
        gamma_dark = 255.0 * np.power(np.clip(norm_f / 248.0, 0.0, 1.0), 1.90)
        contrast_boost = np.clip(norm_f - 0.75 * local_drop, 0.0, 255.0)
        stroke_dark = np.minimum(gamma_dark, contrast_boost)
        stroke_dark = np.where((local_drop >= 26.0) & (norm_f < 195.0), np.minimum(stroke_dark, l_f), stroke_dark)

        out_l = (1.0 - stroke_conf) * bg_clean + stroke_conf * stroke_dark
        stretched = np.clip(out_l, 0.0, 255.0).astype(np.uint8)

        # Giảm sắc độ màu nền (ám vàng/xanh) về trung tính
        a_neutral = cv2.addWeighted(a_channel, 0.20, np.full_like(a_channel, 128), 0.80, 0)
        b_neutral = cv2.addWeighted(b_channel, 0.20, np.full_like(b_channel, 128), 0.80, 0)

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

        # Không chạy bilateralFilter làm nhòe nét mảnh 1px trên ảnh kích thước nhỏ (min(h, w) < 900)
        if self.enable_denoise and min(img.shape[:2]) >= 900:
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
