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

    def clean_handwriting_and_book_gutter(
        self,
        img: np.ndarray,
        remove_colored_pen: bool = False,
        max_gutter_ratio: float = 0.04,
    ) -> np.ndarray:
        """
        1. Mặc định tắt xóa mực màu toàn trang (`remove_colored_pen=False`) để bảo vệ tuyệt đối
           con dấu đỏ, chữ ký xanh và chữ in nằm dưới con dấu. Chỉ khử bút bi xanh/tím/đỏ
           khi truyền tường minh `remove_colored_pen=True`.
        2. Chỉ phát hiện và làm trắng rãnh tối gáy sách / mép trang kế bên ở sát mép ngoài
           trái/phải (< `max_gutter_ratio` = 4% chiều rộng trang), không chạm vùng nội dung.
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

        if remove_colored_pen:
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

        # Bước 2: Phát hiện và làm trắng rãnh tối gáy sách chỉ ở sát mép trái/phải (< 4% chiều rộng trang)
        max_gutter_x = max(1, int(round(w * max(0.01, min(0.04, max_gutter_ratio)))))
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        ink = ((bg_gray.astype(np.int16) - gray.astype(np.int16)) >= 28).astype(np.uint8)
        y0, y1 = int(h * 0.12), int(h * 0.88)
        band = ink[y0:y1, :]
        col_density = band.mean(axis=0)  # tỉ lệ pixel mực trên mỗi cột X

        win = max(15, int(w * 0.015)) | 1
        smooth_col = np.convolve(col_density, np.ones(win) / win, mode="same")

        # Kiểm tra rãnh gáy sách ở sát mép trái (< 4% chiều rộng)
        g_lo_l, g_hi_l = int(w * 0.012), max_gutter_x
        if w >= 650 and g_hi_l > g_lo_l + 6:
            edge_l_end = max(g_lo_l + 2, int(w * 0.025))
            left_edge_ink = float(col_density[int(w * 0.003):edge_l_end].mean())
            left_bg_med = float(np.median(bg_gray[y0:y1, int(w * 0.003):edge_l_end]))
            left_raw_p25 = float(np.percentile(gray[y0:y1, int(w * 0.003):edge_l_end], 25))
            if 0.012 <= left_edge_ink <= 0.35 and left_bg_med >= 120.0 and left_raw_p25 >= 105.0:
                gutter_x = g_lo_l + int(np.argmin(smooth_col[g_lo_l:g_hi_l]))
                if smooth_col[gutter_x] <= 0.012:
                    strip_p25 = float(np.percentile(gray[y0:y1, :max(1, gutter_x - 4)], 25))
                    right_body_ink = float(col_density[gutter_x + 20:min(w, gutter_x + int(w * 0.45))].mean())
                    if strip_p25 >= 105.0 and right_body_ink >= 0.025:
                        out[:, :gutter_x] = bg[:, :gutter_x]
                        LOG.info("  ✓ Đã lọc sạch mép gáy sách bên trái (x=0..%d)", gutter_x)

        # Kiểm tra rãnh gáy sách ở sát mép phải (< 4% chiều rộng)
        g_lo_r, g_hi_r = w - max_gutter_x, w - int(w * 0.012)
        if w >= 650 and g_hi_r > g_lo_r + 6:
            edge_r_start = min(g_hi_r - 2, w - int(w * 0.025))
            right_edge_ink = float(col_density[edge_r_start:w - int(w * 0.003)].mean())
            right_bg_med = float(np.median(bg_gray[y0:y1, edge_r_start:w - int(w * 0.003)]))
            right_raw_p25 = float(np.percentile(gray[y0:y1, edge_r_start:w - int(w * 0.003)], 25))
            if 0.012 <= right_edge_ink <= 0.35 and right_bg_med >= 120.0 and right_raw_p25 >= 105.0:
                gutter_x_r = g_lo_r + int(np.argmin(smooth_col[g_lo_r:g_hi_r]))
                if smooth_col[gutter_x_r] <= 0.012:
                    strip_p25_r = float(np.percentile(gray[y0:y1, min(w - 1, gutter_x_r + 4):], 25))
                    left_body_ink = float(col_density[max(0, gutter_x_r - int(w * 0.45)):max(1, gutter_x_r - 20)].mean())
                    if strip_p25_r >= 105.0 and left_body_ink >= 0.025:
                        out[:, gutter_x_r:] = bg[:, gutter_x_r:]
                        LOG.info("  ✓ Đã lọc sạch mép gáy sách bên phải (x=%d..%d)", gutter_x_r, w)

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
        Khử màu bút dạ quang (vàng, xanh lá, hồng, cam, xanh dương nhạt) về mức kênh sáng nhất
        trước khi chuyển sang Grayscale, nhưng KHÔNG làm nhạt mực con dấu đỏ hay chữ ký xanh đậm.
        """
        if img_or_gray is None:
            return img_or_gray
        if img_or_gray.ndim == 2:
            return img_or_gray
        b, g, r = cv2.split(img_or_gray)
        b_f = b.astype(np.float32)
        g_f = g.astype(np.float32)
        r_f = r.astype(np.float32)
        max_c = np.maximum(np.maximum(b_f, g_f), r_f)
        min_c = np.minimum(np.minimum(b_f, g_f), r_f)
        sat = max_c - min_c
        luma = 0.114 * b_f + 0.587 * g_f + 0.299 * r_f

        # Bút dạ quang & vết ố giấy vàng/nâu có G cao hơn B rõ rệt; ngược lại con dấu đỏ thuần
        # (R vượt trội cả G và B trong khi |G - B| nhỏ) và mực bút bi xanh dương cần được bảo vệ.
        red_stamp_like = (
            np.clip((r_f - g_f - 32.0) / 22.0, 0.0, 1.0)
            * np.clip((r_f - b_f - 32.0) / 22.0, 0.0, 1.0)
            * np.clip((36.0 - np.abs(g_f - b_f)) / 18.0, 0.0, 1.0)
        )
        blue_ink_like = (
            np.clip((b_f - r_f - 20.0) / 18.0, 0.0, 1.0)
            * np.clip((b_f - g_f - 12.0) / 16.0, 0.0, 1.0)
            * np.clip((180.0 - luma) / 45.0, 0.0, 1.0)
        )
        ink_protect = np.maximum(red_stamp_like, blue_ink_like)

        hl_weight = (
            np.clip((sat - 16.0) / 32.0, 0.0, 1.0)
            * np.clip((max_c - 120.0) / 50.0, 0.0, 1.0)
            * (1.0 - ink_protect)
        )
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
    def _compute_edge_contrast_maps(
        norm_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Tính độ tương phản cục bộ trong cửa sổ nhỏ (`15x15` - `21x21`: `local_contrast = local_bg - norm_f`)
        kết hợp độ dốc cạnh cục bộ (`Sobel` và `Laplacian`) trên phiên bản làm mượt nhẹ (`sigma=0.9`)
        để tách bạch nét chữ in xám nhạt sắc nét khỏi thấu mực mờ nhòe / nhiễu hạt giấy.
        """
        h, w = norm_f.shape[:2]
        min_dim = min(h, w)
        scale_loc = 1.0 if min_dim <= 1500 else max(0.55, 1350.0 / float(min_dim))
        norm_u8 = np.clip(norm_f, 0.0, 255.0).astype(np.uint8)
        work_u8 = (
            cv2.resize(norm_u8, None, fx=scale_loc, fy=scale_loc, interpolation=cv2.INTER_AREA)
            if scale_loc < 1.0
            else norm_u8
        )
        k_loc = max(13, min(21, int(round((min_dim // 55) * scale_loc)) | 1))
        loc_bg_w = cv2.morphologyEx(
            work_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_loc, k_loc)),
        )
        loc_bg = (
            cv2.resize(loc_bg_w, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if scale_loc < 1.0
            else loc_bg_w.astype(np.float32)
        )
        local_contrast = np.maximum(0.0, loc_bg - norm_f)

        sm = cv2.GaussianBlur(norm_f, (0, 0), sigmaX=0.9)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        dil_k = np.ones((3, 3), dtype=np.uint8)
        sm_grad = cv2.dilate(np.hypot(gx, gy), dil_k)
        sm_lap = cv2.dilate(np.abs(cv2.Laplacian(sm, cv2.CV_32F, ksize=3)), dil_k)
        return local_contrast, sm_grad, sm_lap, scale_loc

    @staticmethod
    def estimate_page_bleed_severity(raw_norm: np.ndarray) -> float:
        """
        Đo mức độ nhiễu nền / thấu mực mặt sau thực tế của trang trong khoảng [0.0, 1.0].
        - Trang vốn sạch/đều sáng (như IMG20261022100520): trả về ~0.0 (Do No Harm).
        - Trang giấy mỏng thấm mặt sau nặng / có watermark mờ (như real_29.jpg): trả về ~1.0.
        """
        h, w = raw_norm.shape[:2]
        max_side = max(h, w)
        scale = 1.0 if max_side <= 900 else 900.0 / float(max_side)
        rn_w = (
            cv2.resize(raw_norm, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else raw_norm
        )
        rn_u8 = np.clip(rn_w, 0.0, 255.0).astype(np.uint8)
        loc_bg = cv2.morphologyEx(
            rn_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ).astype(np.float32)
        lc = np.maximum(0.0, loc_bg - rn_w)
        core_min = cv2.erode(
            rn_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ).astype(np.float32)
        sm = cv2.GaussianBlur(rn_w, (0, 0), sigmaX=0.9)
        sm_grad = cv2.dilate(
            np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)),
            np.ones((3, 3), dtype=np.uint8),
        )

        orphan_base = (rn_w >= 142.0) & (rn_w <= 232.0) & (core_min >= 128.0) & (lc >= 8.0)
        orphan_mid = float(np.mean(orphan_base))
        orphan_lowgrad = float(np.mean(orphan_base & (sm_grad < 130.0)))
        sev_mid = float(np.clip((orphan_mid - 0.011) / 0.013, 0.0, 1.0))
        sev_lg = float(np.clip((orphan_lowgrad - 0.0065) / 0.0075, 0.0, 1.0))
        return float(max(sev_mid, sev_lg))

    @staticmethod
    def soft_normalize_luma(gray_or_l: np.ndarray, mode: str = "dilate") -> tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn hóa chiếu sáng mềm đa tỉ lệ + khử bút dạ quang nhiều dòng + lọc thấu mực mặt sau
        thích ứng theo độ nhiễu thực tế của trang (Edge-Aware Bleed-Through Suppression).
        Trang sạch giữ nguyên 100% nét xám nhạt sắc nét; trang thấm mặt sau nặng vẫn khử sạch bóng chữ.
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
        raw_norm = np.clip((l_f / bg_safe) * 255.0, 0.0, 255.0)

        # Chỉ kích hoạt lọc thấu mực khi trang thực sự có nhiễu nền/thấu mực (bleed_severity > 0.05)
        # và LUÔN bảo vệ các nét chữ có biên cạnh sắc nét dù màu mực xám nhạt (local_contrast >= 14).
        bleed_severity = ImagePreprocessor.estimate_page_bleed_severity(raw_norm)
        if max(h, w) >= 900 and bleed_severity > 0.05 and float(np.percentile(raw_norm, 0.5)) < 135.0:
            local_contrast, sm_grad, sm_lap, _ = ImagePreprocessor._compute_edge_contrast_maps(raw_norm)
            lc_floor = 14.0 + 14.0 * bleed_severity
            grad_floor = 75.0 + 75.0 * bleed_severity
            edge_keep = (
                np.clip((local_contrast - lc_floor) / 16.0, 0.0, 1.0)
                * np.clip((sm_grad - grad_floor) / 60.0, 0.0, 1.0)
            )
            lap_weight = max(0.0, 1.0 - float(np.clip((bleed_severity - 0.15) / 0.30, 0.0, 1.0)))
            lap_keep = (
                lap_weight
                * np.clip((local_contrast - 18.0) / 14.0, 0.0, 1.0)
                * np.clip((sm_lap - 32.0) / 28.0, 0.0, 1.0)
            )
            strong_contrast_keep = np.clip((local_contrast - (44.0 + 18.0 * bleed_severity)) / 22.0, 0.0, 1.0)
            edge_protect = np.maximum(edge_keep, np.maximum(lap_keep, strong_contrast_keep))

            sn_u8 = raw_norm.astype(np.uint8)
            k_core = max(11, min(19, (min_dim // 110) | 1))
            local_core_min = cv2.erode(
                sn_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core)),
            ).astype(np.float32)
            ghost_w = (
                bleed_severity
                * np.clip((raw_norm - 146.0) / 26.0, 0.0, 1.0)
                * np.clip((local_core_min - 125.0) / 18.0, 0.0, 1.0)
                * (1.0 - edge_protect)
            )
            soft_norm = raw_norm + ghost_w * (255.0 - raw_norm)
        else:
            soft_norm = raw_norm

        return soft_norm, bg_safe

    @staticmethod
    def compute_stroke_protection(
        soft_norm: np.ndarray,
        h: int,
        w: int,
        bleed_severity: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trả về (stroke_conf, local_drop, blackhat) dựa trên độ tương phản cục bộ (`15x15` - `21x21`)
        kết hợp độ dốc cạnh cục bộ (`Sobel / Laplacian`) để bảo vệ 100% nét chữ mặt trước (kể cả chữ
        in xám nhạt, nét mảnh `/`, `:`, `-`) mà vẫn loại bỏ bóng thấu mực mờ nhòe trên giấy mỏng.
        """
        min_dim = min(h, w)
        if bleed_severity is None:
            bleed_severity = ImagePreprocessor.estimate_page_bleed_severity(soft_norm)

        local_drop, sm_grad, sm_lap, scale_loc = ImagePreprocessor._compute_edge_contrast_maps(soft_norm)
        norm_u8 = np.clip(soft_norm, 0.0, 255.0).astype(np.uint8)
        work_u8 = (
            cv2.resize(norm_u8, None, fx=scale_loc, fy=scale_loc, interpolation=cv2.INTER_AREA)
            if scale_loc < 1.0
            else norm_u8
        )
        k_bh = max(7, min(13, int(round((min_dim // 80) * scale_loc)) | 1))
        bh_w = cv2.morphologyEx(
            work_u8,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bh, k_bh)),
        )
        bh = (
            cv2.resize(bh_w, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            if scale_loc < 1.0
            else bh_w.astype(np.float32)
        )

        core_ceil = 195.0 - 10.0 * bleed_severity
        core_drop_min = 12.0 + 2.0 * bleed_severity
        sharp_drop_min = 24.0 + 2.0 * bleed_severity
        bh_min = 20.0 + 2.0 * bleed_severity
        upper_ceil = 232.0 - 4.0 * bleed_severity
        core_ink = np.clip((core_ceil - soft_norm) / 35.0, 0.0, 1.0) * np.clip((local_drop - core_drop_min) / 16.0, 0.0, 1.0)
        sharp_stroke = np.clip((local_drop - sharp_drop_min) / 25.0, 0.0, 1.0) * np.clip((upper_ceil - soft_norm) / 22.0, 0.0, 1.0)
        bh_stroke = np.clip((bh - bh_min) / 23.0, 0.0, 1.0) * np.clip((upper_ceil - soft_norm) / 22.0, 0.0, 1.0)

        # Edge-Aware Stroke Protection: mọi nét mảnh có local_contrast >= 14 kết hợp biên cạnh sắc nét
        # (Sobel / Laplacian) đều được bảo vệ trên trang sạch; trên trang thấm mặt sau nặng hoặc nhiều nhiễu hạt,
        # đòi hỏi độ dốc cạnh thực sự cao để không giữ nhầm nhiễu/vết thấu mực mờ nhòe.
        grad_floor = 48.0 + 102.0 * bleed_severity
        lap_weight = max(0.0, 1.0 - float(np.clip((bleed_severity - 0.15) / 0.30, 0.0, 1.0)))
        edge_signal = np.maximum(
            np.clip((sm_grad - grad_floor) / 60.0, 0.0, 1.0),
            lap_weight
            * np.clip((sm_lap - 20.0) / 26.0, 0.0, 1.0)
            * np.clip((local_drop - 14.0) / 14.0, 0.0, 1.0),
        )
        lc_min_edge = 13.5 + 8.5 * bleed_severity
        upper_edge_soft = 236.0 - 24.0 * bleed_severity
        edge_scale = 1.0 - 0.65 * float(np.clip((bleed_severity - 0.45) / 0.45, 0.0, 1.0))
        edge_stroke = (
            edge_scale
            * np.clip((local_drop - lc_min_edge) / 10.5, 0.0, 1.0)
            * np.clip((upper_edge_soft - soft_norm) / 20.0, 0.0, 1.0)
            * edge_signal
        )
        stroke_conf = np.clip(
            np.maximum(np.maximum(core_ink, edge_stroke), np.maximum(sharp_stroke, bh_stroke)),
            0.0,
            1.0,
        )

        # Chỉ giảm bảo vệ vùng bóng mờ không có lõi đen & không có cạnh sắc khi trang thực sự có thấu mực
        if max(h, w) >= 900 and bleed_severity > 0.05 and float(np.percentile(soft_norm, 0.5)) < 135.0:
            k_core = max(11, min(19, (min_dim // 110) | 1))
            local_core_min = cv2.erode(
                norm_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core)),
            ).astype(np.float32)
            has_front_core = np.clip((134.0 - local_core_min) / 18.0, 0.0, 1.0)
            sharp_keep_w = max(0.0, 1.0 - float(np.clip((bleed_severity - 0.35) / 0.45, 0.0, 1.0)))
            sharp_dark_keep = sharp_keep_w * np.clip((edge_stroke - 0.65) / 0.30, 0.0, 1.0) * np.clip((195.0 - soft_norm) / 25.0, 0.0, 1.0)
            suppress_gate = np.maximum(
                has_front_core,
                np.maximum(
                    np.clip((136.0 - soft_norm) / 20.0, 0.0, 1.0),
                    sharp_dark_keep,
                ),
            )
            factor = (1.0 - bleed_severity) + bleed_severity * suppress_gate
            stroke_conf = stroke_conf * factor

        return stroke_conf, local_drop, bh

    def magic_enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Khử bóng đổ, làm trắng nền giấy và bảo vệ nét chữ đậm nét (tương tự CamScanner,
        an toàn tuyệt đối cho ảnh nhỏ / giấy cổ nhờ ước lượng nền thích ứng và Edge-Aware Stroke Protection).
        """
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_f = self.neutralize_highlighter_luma(img).astype(np.float32) if img.ndim == 3 else l_channel.astype(np.float32)
        norm_f, bg_safe = self.soft_normalize_luma(img if img.ndim == 3 else l_channel, mode="dilate")
        raw_norm = np.clip((l_f / bg_safe) * 255.0, 0.0, 255.0)
        bleed_severity = self.estimate_page_bleed_severity(raw_norm)
        stroke_conf, local_drop, _ = self.compute_stroke_protection(norm_f, h, w, bleed_severity=bleed_severity)

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
