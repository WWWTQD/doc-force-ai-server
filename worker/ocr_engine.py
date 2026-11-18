"""
OCR Engine Module cho Doc Forge AI Worker.

Tích hợp PaddleOCR (hoặc Fallback Engine nếu chưa cài PaddleOCR)
để nhận dạng chữ tiếng Việt/Anh và xây dựng Layout JSON cho Backend.
"""

import os
import re
import logging
import uuid

try:
    from vietnamese_normalizer import normalize_vietnamese_text, is_gibberish_or_handwriting_text
except ImportError:
    def normalize_vietnamese_text(t: str) -> str:
        return t

    def is_gibberish_or_handwriting_text(t: str) -> bool:
        return not bool(t and t.strip())

# Đặt KMP_DUPLICATE_LIB_OK và import torch trước paddle để tránh xung đột OpenMP DLL trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
    torch.set_num_threads(max(4, os.cpu_count() or 4))
except Exception:
    torch = None

import cv2
import numpy as np

try:
    from table_structure import NET_H, NET_W, TableLineNet, extract_tables, predict_line_maps
    TABLE_NET_AVAILABLE = True
except Exception:
    TABLE_NET_AVAILABLE = False

LOG = logging.getLogger("ocr_engine")
LOG.setLevel(logging.INFO)

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    LOG.warning("Không thể nạp YOLO: %s. Bỏ qua YOLO layout detection.", e)

PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    LOG.warning("Không thể nạp PaddleOCR: %s. OCR Engine sẽ chạy ở chế độ Fallback Engine.", e)

# VietOCR: mô hình nhận dạng chữ tiếng Việt (đủ dấu, đ/ư/ơ) — dùng thay bộ nhận dạng Latin của PaddleOCR.
# PaddleOCR khi đó chỉ làm nhiệm vụ phát hiện vị trí dòng chữ.
VIETOCR_AVAILABLE = False
try:
    from PIL import Image
    from vietocr.tool.predictor import Predictor as VietOCRPredictor
    from vietocr.tool.config import Cfg as VietOCRCfg
    VIETOCR_AVAILABLE = True
except Exception as e:
    LOG.warning("Không thể nạp VietOCR: %s. Dùng bộ nhận dạng Latin của PaddleOCR (kém chính xác với tiếng Việt).", e)

# vgg_seq2seq: nhanh (~0.6s/dòng CPU); vgg_transformer: chậm hơn ~5 lần
VIETOCR_MODEL = os.environ.get("VIETOCR_MODEL", "vgg_seq2seq")
USE_VIETOCR = os.environ.get("OCR_RECOGNIZER", "vietocr").lower() == "vietocr"
# Dòng có tỉ lệ rộng/cao lớn hơn ngưỡng này sẽ được tách đôi tại khoảng trắng giữa hai từ,
# vì mô hình seq2seq hay bỏ sót chữ trên chuỗi quá dài.
MAX_LINE_ASPECT = float(os.environ.get("VIETOCR_MAX_ASPECT", "16"))


def _order_poly(pts) -> np.ndarray:
    """Sắp xếp 4 đỉnh tứ giác theo thứ tự chuẩn: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    idx = np.argsort(pts[:, 0])
    left = pts[idx[:2]]
    right = pts[idx[2:]]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _rotate_crop(img: np.ndarray, pts, pad_x: int = 8, pad_y: int | None = None) -> np.ndarray:
    """Cắt dòng chữ theo đúng tứ giác (kể cả dòng nghiêng/cong nhẹ) với đệm đủ cho dấu thanh tiếng Việt."""
    pts = _order_poly(pts)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 1), max(h, 1)
    py = pad_y if pad_y is not None else max(6, min(14, int(round(h * 0.16))))
    px = max(6, pad_x)
    dst = np.array([
        [px, py],
        [w + px, py],
        [w + px, h + py],
        [px, h + py],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        img, M, (w + 2 * px, h + 2 * py),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _resolve_layout_model_path() -> str:
    """
    Model phân tích bố cục (heading/paragraph/table/...).
    Ưu tiên biến môi trường LAYOUT_MODEL_PATH, sau đó layout_best_backup.pt,
    cuối cùng best.pt (tên cũ) để tương thích ngược.
    """
    env_path = os.environ.get("LAYOUT_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    for root in (base_dir, project_root, "/app"):
        for folder in ("models", "Model"):
            models_dir = os.path.join(root, folder)
            for name in ("layout_best_backup.pt", "best.pt"):
                candidate = os.path.join(models_dir, name)
                if os.path.exists(candidate):
                    return candidate
    return os.path.join(base_dir, "models", "layout_best_backup.pt")


def _resolve_model_path(filenames: tuple[str, ...]) -> str | None:
    """Tìm tệp checkpoint model trong các thư mục `models/` tiêu chuẩn."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    for root in (base_dir, project_root, "/app"):
        models_dir = os.path.join(root, "models")
        for name in filenames:
            cand = os.path.join(models_dir, name)
            if os.path.exists(cand):
                return cand
    return None


def _resolve_typography_model_path() -> str | None:
    return _resolve_model_path(("typography_real_best.pth", "typography_best.pth"))


def _resolve_orientation_model_path() -> str | None:
    return _resolve_model_path(("orientation_net.pth",))


def _resolve_table_model_path() -> str | None:
    return _resolve_model_path(("table_line_net.pth",))


def _resolve_vietocr_ft_path() -> str | None:
    return _resolve_model_path(("vietocr_camera_ft.pth",))


if torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F

    class _ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + x)

    class DeepDocumentTypographyResNet(nn.Module):
        """Mạng Deep ResNet 5 khối dự đoán Bold, Italic, Font Family (Serif/Sans) và Font Size (pt)."""

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
            self.res1 = _ResidualBlock(32)
            self.down1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.res2 = _ResidualBlock(64)
            self.down2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.res3 = _ResidualBlock(128)
            self.down3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            self.res4 = _ResidualBlock(256)
            self.res5 = _ResidualBlock(256)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.3)

            self.head_bold = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
            )
            self.head_italic = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
            )
            self.head_serif = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
            )
            self.head_size = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1)
            )

        def forward(self, x):
            x = self.res1(self.stem(x))
            x = self.res2(self.down1(x))
            x = self.res3(self.down2(x))
            x = self.res5(self.res4(self.down3(x)))
            feat = self.dropout(self.global_pool(x).flatten(1))
            return (
                self.head_bold(feat),
                self.head_italic(feat),
                self.head_serif(feat),
                self.head_size(feat).squeeze(-1),
            )

    class _OrientResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            out = F.silu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.silu(x + out)

    class DocumentOrientationResNet(nn.Module):
        """Mạng ResNet 4 lớp phân loại hướng trang tài liệu camera (0°, 90°, 180°, 270°) từ patch 128x128."""

        def __init__(self, num_classes: int = 4):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.SiLU(),
            )
            self.stage1 = nn.Sequential(
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                _OrientResBlock(64),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(),
                _OrientResBlock(96),
            )
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.head = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(96 * 4, 96),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(96, num_classes),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            return self.head(self.pool(x).flatten(1))


class OCREngine:
    """Wrapper quản lý PaddleOCR, VietOCR, YOLO Layout và DeepDocumentTypographyResNet."""

    def __init__(self, lang: str = "vi", use_gpu: bool = False):
        self.lang = lang
        self.ocr_instance = None
        self.yolo_model = None
        self.typo_model = None
        self.orient_model = None
        self.table_model = None
        self.typo_device = torch.device("cuda" if (use_gpu and torch is not None and torch.cuda.is_available()) else "cpu") if torch is not None else None
        root_log = logging.getLogger()
        saved_handlers = list(root_log.handlers)
        root_log.setLevel(logging.INFO)
        LOG.setLevel(logging.INFO)

        def _restore_handlers():
            root_log.setLevel(logging.INFO)
            LOG.setLevel(logging.INFO)
            if saved_handlers:
                for h in list(root_log.handlers):
                    if h not in saved_handlers:
                        root_log.removeHandler(h)
                for h in saved_handlers:
                    if h not in root_log.handlers:
                        root_log.addHandler(h)

        if YOLO_AVAILABLE:
            try:
                model_path = _resolve_layout_model_path()
                if not os.path.exists(model_path):
                    model_path = "models/best.pt"
                if os.path.exists(model_path):
                    LOG.info("Khởi tạo YOLO layout detection (%s)...", model_path)
                    self.yolo_model = YOLO(model_path)
                    _restore_handlers()
                    LOG.info("YOLO model khởi tạo thành công.")
                else:
                    LOG.warning("Không tìm thấy model YOLO tại %s. Bỏ qua layout detection.", model_path)
            except Exception as e:
                _restore_handlers()
                LOG.error("Lỗi khi khởi tạo YOLO model: %s. Bỏ qua layout detection.", e)

        if PADDLE_AVAILABLE:
            try:
                LOG.info("Khởi tạo PaddleOCR (lang=%s, use_gpu=%s)...", lang, use_gpu)
                self.ocr_instance = PaddleOCR(
                    use_angle_cls=False,
                    lang=lang,
                    use_gpu=use_gpu,
                    show_log=False,
                    det_limit_side_len=960,
                    det_db_thresh=0.20,
                    det_db_box_thresh=0.18,
                    det_db_unclip_ratio=1.35,
                )
                _restore_handlers()
                LOG.info("PaddleOCR khởi tạo thành công.")
            except Exception as e:
                _restore_handlers()
                LOG.error("Lỗi khi khởi tạo PaddleOCR: %s. Chuyển sang Fallback Engine.", e)

        if torch is not None:
            try:
                # Giới hạn 4 luồng thực thi vật lý giúp tránh xung đột khóa OpenMP trong Docker trên CPU
                torch.set_num_threads(4)
            except Exception:
                pass

        self.viet_recognizer = None
        if USE_VIETOCR and VIETOCR_AVAILABLE and self.ocr_instance is not None:
            try:
                LOG.info("Khởi tạo VietOCR (%s)...", VIETOCR_MODEL)
                cfg = VietOCRCfg.load_config_from_name(VIETOCR_MODEL)
                cfg["device"] = "cuda" if use_gpu else "cpu"
                cfg["cnn"]["pretrained"] = False
                cfg["predictor"]["beamsearch"] = False
                self.viet_recognizer = VietOCRPredictor(cfg)
                viet_ft_path = _resolve_vietocr_ft_path()
                if viet_ft_path and torch is not None:
                    try:
                        ft_state = torch.load(viet_ft_path, map_location="cpu")
                        self.viet_recognizer.model.load_state_dict(ft_state, strict=False)
                        self.viet_recognizer.model.eval()
                        LOG.info("VietOCR đã nạp camera fine-tune weights (%s).", viet_ft_path)
                    except Exception as ft_exc:
                        LOG.warning("Không nạp được camera weights VietOCR (%s): %s", viet_ft_path, ft_exc)
                LOG.info("VietOCR khởi tạo thành công — dùng làm bộ nhận dạng chữ chính.")
            except Exception as e:
                LOG.error("Lỗi khởi tạo VietOCR: %s. Dùng bộ nhận dạng của PaddleOCR.", e)

        if torch is not None:
            typo_path = _resolve_typography_model_path()
            if typo_path:
                try:
                    ckpt = torch.load(typo_path, map_location=self.typo_device)
                    model = DeepDocumentTypographyResNet().to(self.typo_device)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    self.typo_model = model
                    LOG.info("DeepDocumentTypographyResNet đã nạp thành công (%s).", typo_path)
                except Exception as exc:
                    LOG.warning("Không nạp được DeepDocumentTypographyResNet (%s): %s", typo_path, exc)

            orient_path = _resolve_orientation_model_path()
            table_path = _resolve_table_model_path()
            if table_path and TABLE_NET_AVAILABLE:
                try:
                    t_ckpt = torch.load(table_path, map_location=self.typo_device)
                    t_model = TableLineNet().to(self.typo_device)
                    t_model.load_state_dict(t_ckpt.get("state_dict", t_ckpt) if isinstance(t_ckpt, dict) else t_ckpt)
                    t_model.eval()
                    self.table_model = t_model
                    LOG.info("TableLineNet đã nạp thành công (%s).", table_path)
                except Exception as exc:
                    LOG.warning("Không nạp được TableLineNet (%s): %s", table_path, exc)
            if orient_path:
                try:
                    o_ckpt = torch.load(orient_path, map_location=self.typo_device)
                    o_model = DocumentOrientationResNet(num_classes=4).to(self.typo_device)
                    o_state = o_ckpt.get("state_dict", o_ckpt) if isinstance(o_ckpt, dict) else o_ckpt
                    o_model.load_state_dict(o_state, strict=False)
                    o_model.eval()
                    self.orient_model = o_model
                    LOG.info("DocumentOrientationResNet đã nạp thành công (%s).", orient_path)
                except Exception as exc:
                    LOG.warning("Không nạp được DocumentOrientationResNet (%s): %s", orient_path, exc)

    def process_image(self, img: np.ndarray, page_number: int, table_ref_img: np.ndarray | None = None) -> dict:
        """
        Nhận vào ảnh OpenCV (BGR) và pageNumber.
        Trả về dictionary rawLayoutJson khớp với schema LayoutValidator của Backend.

        Pipeline:
          1. YOLO layout detection (nếu có) → crop từng vùng
          2. PaddleOCR nhận dạng chữ trên từng vùng (hoặc toàn ảnh nếu không có YOLO)
          3. Fallback engine nếu PaddleOCR không sẵn sàng
        """
        h, w = img.shape[:2]

        # Bước 1: YOLO layout detection (chỉ bật khi có GPU để tránh nghẽn 10s trên CPU khi đã có bộ phân tích Bảng & Bố cục trực tiếp)
        use_yolo_layout = bool(self.typo_device is not None and self.typo_device.type == "cuda")
        yolo_regions = self._detect_yolo_regions(img, w, h) if use_yolo_layout else []

        if self.ocr_instance is not None:
            try:
                return self._run_paddle_ocr(img, page_number, w, h, yolo_regions=yolo_regions, table_ref_img=table_ref_img)
            except Exception as e:
                import traceback
                LOG.error("Lỗi trong quá trình chạy PaddleOCR: %s. Dùng fallback engine.", e)
                traceback.print_exc()

        return self._run_fallback_ocr(img, page_number, w, h)

    @property
    def can_score_orientation(self) -> bool:
        return (self.ocr_instance is not None) or (self.orient_model is not None)

    @staticmethod
    def _vi_syllable_validity_score(texts: list[str]) -> float:
        """
        Đánh giá độ hợp lệ ngữ âm tiếng Việt thuần tuý (không dùng danh sách từ cố định):
          - Kiểm tra tỷ lệ âm tiết ghép đúng nguyên âm + vần cuối tiếng Việt
          - Phạt chuỗi ký tự vô nghĩa khi ảnh bị lộn ngược (quá nhiều chữ hoa rời rạc hoặc phụ âm không hợp lệ).
        """
        import unicodedata

        joined = unicodedata.normalize("NFC", " ".join((t or "").strip() for t in texts))
        if len(joined) < 4:
            return 0.0
        tokens = [tok.strip(".,;:!?\"'()[]{}-–—") for tok in joined.split()]
        alpha_tokens = [tok for tok in tokens if len(tok) >= 2 and any(ch.isalpha() for ch in tok)]
        if not alpha_tokens:
            return 0.0

        syllable_re = re.compile(
            r"^(?:[bcdfghklmnpqrstvwxzđ]{0,3})"
            r"[aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ]{1,3}"
            r"(?:ch|ng|nh|c|m|n|p|t|i|o|u|y)?$"
        )
        vn_diacritics = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
        valid_cnt = 0
        dia_tokens = 0
        all_upper_cnt = 0
        for tok in alpha_tokens:
            low = tok.lower()
            if tok.isupper() and len(tok) >= 2:
                all_upper_cnt += 1
            if any(c in vn_diacritics for c in low):
                dia_tokens += 1
            # Tiếng Việt không dùng f, j, w, z trong âm tiết chuẩn và không quá dài một âm tiết
            if len(low) <= 7 and not any(c in "fjwz0123456789" for c in low) and syllable_re.match(low):
                valid_cnt += 1

        valid_ratio = valid_cnt / len(alpha_tokens)
        dia_ratio = min(1.0, dia_tokens / max(1, len(alpha_tokens)) * 1.6)
        upper_ratio = all_upper_cnt / len(alpha_tokens)
        # Văn bản lộn ngược thường cho ra tỷ lệ âm tiết hợp lệ thấp hoặc toàn chữ hoa rời rạc
        score = 0.65 * valid_ratio + 0.35 * (valid_ratio * dia_ratio)
        if upper_ratio > 0.72 and valid_ratio < 0.75:
            score *= 0.45
        return float(max(0.0, min(1.0, score)))

    def score_orientation(self, img: np.ndarray) -> float:
        """
        Chấm điểm "đọc được" của ảnh ở hướng hiện tại (0..1) bằng mạng `DocumentOrientationResNet`
        kết hợp kiểm chứng ngữ âm tiếng Việt tổng quát (tuyệt đối không dùng từ khóa mẫu).
        """
        h, w = img.shape[:2]
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # 1. Điểm AI từ DocumentOrientationResNet trên 6 vùng giàu nét chữ đa tỷ lệ (160x160)
        ai_score = None
        if self.orient_model is not None and torch is not None:
            try:
                scale_ai = min(1.0, 1400.0 / max(h, w))
                g_ai = (
                    cv2.resize(gray_full, None, fx=scale_ai, fy=scale_ai, interpolation=cv2.INTER_AREA)
                    if scale_ai < 1.0
                    else gray_full
                )
                gh, gw = g_ai.shape[:2]
                grad = np.abs(cv2.Sobel(g_ai, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
                    cv2.Sobel(g_ai, cv2.CV_32F, 0, 1, ksize=3)
                )
                cand_patches = []
                for frac in (0.18, 0.26):
                    win = max(120, min(min(gh, gw) - 8, int(min(gh, gw) * frac)))
                    step = max(32, win // 2)
                    y_min, y_max = int(gh * 0.08), max(int(gh * 0.08) + 1, int(gh * 0.88) - win)
                    x_min, x_max = int(gw * 0.08), max(int(gw * 0.08) + 1, int(gw * 0.88) - win)
                    for yy in range(y_min, y_max + 1, step):
                        for xx in range(x_min, x_max + 1, step):
                            g_win = grad[yy : yy + win, xx : xx + win]
                            p_win = g_ai[yy : yy + win, xx : xx + win]
                            if p_win.size > 0 and float(np.std(p_win)) > 16.0:
                                cand_patches.append((float(np.mean(g_win)), p_win))
                cand_patches.sort(key=lambda item: item[0], reverse=True)
                selected = [p[1] for p in cand_patches[:6]]
                if not selected:
                    selected = [cv2.resize(g_ai, (128, 128), interpolation=cv2.INTER_AREA)]
                tensors = []
                for p in selected:
                    p128 = cv2.resize(p, (128, 128), interpolation=cv2.INTER_AREA)
                    t = (p128.astype(np.float32) / 255.0 - 0.5) / 0.5
                    tensors.append(t)
                batch_t = torch.from_numpy(np.stack(tensors, axis=0)).unsqueeze(1).to(self.typo_device)
                with torch.inference_mode():
                    logits = self.orient_model(batch_t)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                p0 = float(probs[:, 0].mean())
                p2 = float(probs[:, 2].mean())
                p_side = float((probs[:, 1] + probs[:, 3]).mean())
                ai_score = p0 / max(1e-4, p0 + p2 + 0.6 * p_side)
            except Exception as exc:
                LOG.debug("DocumentOrientationResNet fallback: %s", exc)
                ai_score = None

        # 2. Điểm kiểm chứng dòng ngang + ngữ âm tiếng Việt từ OCR
        phon_score = None
        if self.ocr_instance is not None:
            scale = min(1.0, 1100.0 / max(h, w))
            small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img
            sh, sw = small.shape[:2]
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

            grad_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(26, sw // 18), 3))
            closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            horiz_crops = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > sw * 0.20 and 10 <= ch <= max(64, sh // 12) and cw > ch * 3.0:
                    if int(sh * 0.06) <= y <= int(sh * 0.90):
                        pad_y = max(2, ch // 5)
                        y1, y2 = max(0, y - pad_y), min(sh, y + ch + pad_y)
                        x1, x2 = max(0, x), min(sw, x + cw)
                        crop = small[y1:y2, x1:x2]
                        if crop.size > 0:
                            horiz_crops.append((cw, crop))

            if horiz_crops:
                horiz_crops.sort(key=lambda item: item[0], reverse=True)
                top_crops = [c[1] for c in horiz_crops[:3]]
                if self.viet_recognizer is not None:
                    pil_samples = []
                    for c in top_crops:
                        ch_c, cw_c = c.shape[:2]
                        new_w = min(440, max(32, int(cw_c * (32.0 / max(1, ch_c)))))
                        rs = cv2.resize(c, (new_w, 32), interpolation=cv2.INTER_AREA)
                        pil_samples.append(Image.fromarray(cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)))
                    if torch is not None:
                        with torch.inference_mode():
                            texts = self.viet_recognizer.predict_batch(pil_samples)
                    else:
                        texts = self.viet_recognizer.predict_batch(pil_samples)
                    phon_score = self._vi_syllable_validity_score(texts)
                else:
                    rec_res, _ = self.ocr_instance.text_recognizer(top_crops)
                    confs = [float(conf) for text, conf in (rec_res or []) if len((text or "").strip()) >= 2]
                    phon_score = float(np.mean(confs)) if confs else 0.0

        if ai_score is not None and phon_score is not None:
            return float(0.72 * ai_score + 0.28 * phon_score)
        if ai_score is not None:
            return float(ai_score)
        return float(phon_score or 0.0)

    def _detect_yolo_regions(self, img: np.ndarray, width: int, height: int) -> list[dict]:
        """
        Chạy YOLO để phát hiện các vùng layout (text block, table, figure, ...).
        Trả về danh sách dict gồm bbox [x1, y1, x2, y2] và class_name.
        Trả về list rỗng nếu YOLO không khả dụng hoặc không detect được gì.
        """
        if self.yolo_model is None:
            return []
        try:
            # imgsz=640 tăng tốc độ phát hiện bố cục trên CPU
            results = self.yolo_model(img, imgsz=640, verbose=False)
            regions = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(x1 + 1, min(x2, width))
                    y2 = max(y1 + 1, min(y2, height))
                    cls_id = int(box.cls[0])
                    cls_name = result.names.get(cls_id, "text")
                    conf = float(box.conf[0])
                    regions.append({
                        "bbox": [x1, y1, x2, y2],
                        "class_name": cls_name,
                        "confidence": conf,
                    })
            if regions:
                LOG.info("YOLO phát hiện %d vùng layout.", len(regions))
            return regions
        except Exception as e:
            LOG.warning("YOLO detection thất bại: %s. Bỏ qua layout detection.", e)
            return []


    def _detect_and_recognize_vietocr(self, img: np.ndarray, width: int) -> list:
        """
        PaddleOCR phát hiện tứ giác từng dòng -> lọc nhiễu lề / gáy sách -> cắt thẳng theo tứ giác
        -> tách đệ quy mọi box bị gộp nhiều dòng dọc -> tách dòng dài tại khoảng trắng giữa các từ
        -> VietOCR đọc trực tiếp từ ảnh thật kèm kiểm định xác suất loại bỏ chữ nháp viết tay.
        Trả về list [poly, (text, confidence)] cùng định dạng PaddleOCR.
        """
        ih, iw = img.shape[:2]
        det_scale = min(1.0, 1280.0 / max(ih, iw))
        det_img = cv2.resize(img, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_AREA) if det_scale < 1.0 else img
        dt_boxes, _ = self.ocr_instance.text_detector(det_img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
        if det_scale < 1.0:
            dt_boxes = [np.asarray(b, dtype=np.float32) / det_scale for b in dt_boxes]

        # Ước lượng cột chữ chính [col_left_est, right(y)] từ các dòng rộng trên trang
        clean_wide = []
        wide_lefts = []
        for b in dt_boxes:
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            bw = max(xs) - min(xs)
            if bw > width * 0.38 and max(xs) < width * 0.97:
                clean_wide.append((float(np.mean(ys)), float(max(xs))))
                if min(xs) > width * 0.05:
                    wide_lefts.append(float(min(xs)))

        col_left_est = float(np.percentile(wide_lefts, 15)) if len(wide_lefts) >= 3 else width * 0.10

        if len(clean_wide) >= 4:
            ys_w = np.array([p[0] for p in clean_wide])
            xs_w = np.array([p[1] for p in clean_wide])
            mid = len(clean_wide) // 2
            order_idx = np.argsort(ys_w)
            top_half = order_idx[:mid]
            bot_half = order_idx[mid:]
            y1_m, x1_m = float(np.median(ys_w[top_half])), float(np.percentile(xs_w[top_half], 96))
            y2_m, x2_m = float(np.median(ys_w[bot_half])), float(np.percentile(xs_w[bot_half], 96))
            slope = (x2_m - x1_m) / max(1.0, (y2_m - y1_m))
            intercept = x1_m - slope * y1_m
        else:
            slope, intercept = 0.0, width * 0.98

        def _right_margin_at(y_val: float) -> float:
            return slope * y_val + intercept

        is_book_page = len(clean_wide) >= 6
        raw_items = []
        single_line_heights = []
        for b in dt_boxes:
            pts = _order_poly(b)
            xs = pts[:, 0]
            ys = pts[:, 1]
            x1, x2 = float(min(xs)), float(max(xs))
            ym = float(np.mean(ys))
            bw = x2 - x1
            rm = _right_margin_at(ym)

            # Loại bỏ toàn bộ mảnh chữ thuộc trang sách kế bên ở gáy sách mép trái
            if x2 < 65 or (is_book_page and x2 < max(110.0, col_left_est - 25.0)):
                continue
            # Lọc chữ nháp viết tay nằm ngoài lề phải của cột chữ in
            if is_book_page and x1 > rm + 12 and bw < 220:
                continue

            h_box = float(0.5 * (np.linalg.norm(pts[0] - pts[3]) + np.linalg.norm(pts[1] - pts[2])))
            raw_items.append((ym, x1, pts, h_box))
            if bw > width * 0.15 and 22.0 <= h_box <= ih * 0.036:
                single_line_heights.append(h_box)

        if not raw_items:
            return []

        max_single_h = max(88.0, float(ih) * 0.036)
        if single_line_heights:
            single_h = float(np.clip(np.median(single_line_heights), 26.0, max_single_h))
        else:
            all_h = [it[3] for it in raw_items]
            single_h = float(np.clip(np.median(all_h), 26.0, max_single_h))

        # _rotate_crop cộng thêm 2 * pad_y (~0.32 * single_h), nên dòng đơn thực tế có ch ~ 1.32-1.48 * single_h
        # và box gộp 2 dòng có ch >= 1.95 * single_h. Đặt split_thresh = max(68.0, single_h * 1.68)
        # đảm bảo tuyệt đối không cắt ngang dòng đơn mà vẫn tách chính xác mọi box gộp 2 dòng.
        split_thresh = max(68.0, single_h * 1.68)

        # Ước lượng độ dốc nghiêng cục bộ (local tilt slope dy/dx) theo vùng Y của trang từ các dòng rộng
        # để chuẩn hóa toạ độ Y chính xác ngay cả khi trang sách hơi cong ở đầu/cuối trang.
        wide_slopes: list[tuple[float, float]] = []
        for ym_s, _, pts_s, _ in raw_items:
            bw_s = float(max(pts_s[:, 0]) - min(pts_s[:, 0]))
            if bw_s > width * 0.30:
                dx_top = float(pts_s[1, 0] - pts_s[0, 0])
                dy_top = float(pts_s[1, 1] - pts_s[0, 1])
                if dx_top > 40.0:
                    wide_slopes.append((float(ym_s), float(np.clip(dy_top / dx_top, -0.12, 0.12))))
        page_tilt_slope = float(np.median([s for _, s in wide_slopes])) if wide_slopes else 0.0

        def _local_tilt_slope(y_q: float) -> float:
            if not wide_slopes:
                return 0.0
            win_y = max(420.0, float(ih) * 0.16)
            near = [s for ym_s, s in wide_slopes if abs(ym_s - y_q) <= win_y]
            if near:
                return float(np.median(near))
            return page_tilt_slope

        def _split_multiline_crop(ym_val: float, pts_arr: np.ndarray, crop_img: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]]:
            """Tách đệ quy một box bị DBNet gộp dọc (2..N dòng) thành từng dòng đơn độc lập mà tuyệt đối không cắt đôi dòng đơn."""
            ch, cw = crop_img.shape[:2]
            if ch <= split_thresh or cw <= 24:
                return [(ym_val, pts_arr, crop_img)]

            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
            _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Dải hẹp [22%..52%] tránh chữ nghiêng nhẹ làm lấp khe trắng giữa 2 dòng sát nhau
            x_lo, x_hi = max(0, int(cw * 0.22)), max(int(cw * 0.22) + 1, int(cw * 0.52))
            if (x_hi - x_lo) < 24:
                x_lo, x_hi = 0, cw
            row_proj = ink[:, x_lo:x_hi].sum(axis=1).astype(np.float32)
            if len(row_proj) >= 5:
                row_proj_smooth = np.convolve(row_proj, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
            else:
                row_proj_smooth = row_proj

            bands: list[tuple[int, int]] = []
            queue = [(0, ch)]
            while queue:
                y0, y1 = queue.pop(0)
                span = y1 - y0
                if span <= split_thresh:
                    bands.append((y0, y1))
                    continue
                # Tìm thung lũng khoảng trắng giữa dòng đầu tiên và phần còn lại
                s_lo = y0 + int(max(18.0, single_h * 0.62))
                s_hi = min(y1 - int(max(18.0, single_h * 0.58)), y0 + int(single_h * 1.45))
                if s_hi <= s_lo + 2:
                    s_lo = y0 + int(span * 0.34)
                    s_hi = y0 + int(span * 0.66)
                if s_hi <= s_lo + 1:
                    bands.append((y0, y1))
                    continue
                rel_min_idx = int(np.argmin(row_proj_smooth[s_lo:s_hi]))
                split_y = s_lo + rel_min_idx
                # Chỉ tách nếu thực sự có khe thưa mực rõ rệt giữa 2 đỉnh chữ (khi chiều cao mới vượt ngưỡng nhẹ)
                if span < single_h * 1.85:
                    top_peak = float(np.max(row_proj_smooth[y0:split_y])) if split_y > y0 else 0.0
                    bot_peak = float(np.max(row_proj_smooth[split_y:y1])) if y1 > split_y else 0.0
                    min_val = float(row_proj_smooth[split_y])
                    if min_val > 0.35 * max(1.0, min(top_peak, bot_peak)):
                        bands.append((y0, y1))
                        continue
                queue.insert(0, (split_y, y1))
                queue.insert(0, (y0, split_y))

            tl, tr, br, bl = pts_arr[0], pts_arr[1], pts_arr[2], pts_arr[3]
            top_y_page = float(min(pts_arr[:, 1]))
            bot_y_page = float(max(pts_arr[:, 1]))
            min_active = max(8, int(single_h * 0.22))
            out_parts = []
            for y0, y1 in bands:
                if (y1 - y0) < 16:
                    continue
                active = int((row_proj[y0:y1] > max(1.0, (x_hi - x_lo) * 0.012)).sum())
                if active < min_active:
                    continue
                r0 = float(y0) / max(1.0, float(ch))
                r1 = float(y1) / max(1.0, float(ch))
                p_tl = tl + r0 * (bl - tl)
                p_tr = tr + r0 * (br - tr)
                p_br = tr + r1 * (br - tr)
                p_bl = tl + r1 * (bl - tl)
                sub_poly = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
                sub_crop = crop_img[max(0, y0 - 2):min(ch, y1 + 2), :]
                sub_ym = top_y_page + 0.5 * (r0 + r1) * (bot_y_page - top_y_page)
                out_parts.append((sub_ym, sub_poly, sub_crop))

            return out_parts if out_parts else [(ym_val, pts_arr, crop_img)]

        # Bước 1: Cắt thẳng và tách mọi box gộp nhiều dòng thành danh sách dòng đơn
        single_items = []
        for ym, _, pts, _ in raw_items:
            crop = _rotate_crop(img, pts)
            ch, cw = crop.shape[:2]
            if ch < 8 or cw < 8 or ch > cw * 2.5:
                continue
            parts = _split_multiline_crop(ym, pts, crop)
            for sub_y, sub_poly, sub_crop in parts:
                sch, scw = sub_crop.shape[:2]
                if sch < 8 or scw < 8 or sch > scw * 2.8:
                    continue
                # Lọc bỏ box bóng chữ thấu mực rất mờ trên ảnh scan lớn (không có lõi nét chữ mặt trước)
                if max(ih, width) >= 900:
                    sc_gray = cv2.cvtColor(sub_crop, cv2.COLOR_BGR2GRAY) if sub_crop.ndim == 3 else sub_crop
                    p5_ink = float(np.percentile(sc_gray, 5))
                    p85_bg = float(np.percentile(sc_gray, 85))
                    if p5_ink > 152.0 and (p85_bg - p5_ink) < 62.0:
                        continue
                sxs = sub_poly[:, 0]
                sx1, sx2 = float(min(sxs)), float(max(sxs))
                if is_book_page and sx2 < max(110.0, col_left_est - 25.0):
                    continue
                sxm = 0.5 * (sx1 + sx2)
                loc_slope = _local_tilt_slope(float(sub_y))
                sub_y_norm = float(sub_y) - loc_slope * (sxm - col_left_est)
                single_items.append((sub_y_norm, float(sub_y), sx1, sx2, sub_poly, sub_crop))

        if not single_items:
            return []

        # Bước 2: Sắp xếp theo toạ độ Y đã bù góc nghiêng trang (sub_y_norm), khử trùng lặp và gom các mảnh cùng 1 dòng ngang
        single_items.sort(key=lambda it: it[0])
        line_clusters: list[list[tuple[float, float, float, float, np.ndarray, np.ndarray]]] = []
        for item in single_items:
            sub_y_norm, sub_y_raw, sx1, sx2, sub_poly, sub_crop = item
            sw = max(1.0, sx2 - sx1)
            placed = False
            for cluster in line_clusters:
                mean_y_norm = float(np.mean([c[0] for c in cluster]))
                max_x_ov = max((min(sx2, c[3]) - max(sx1, c[2])) for c in cluster)
                # Hai mảnh nằm cạnh nhau (không đè X, ví dụ "TIN MIỀN BẮC PHÁT ĐÊM") cho phép sai số Y 0.55 * single_h;
                # hai dòng chồng lấp X chỉ cùng cluster khi chênh lệch Y đã bù nghiêng < 0.42 * single_h.
                same_horiz_row = (abs(sub_y_norm - mean_y_norm) < single_h * 0.42) or (
                    max_x_ov <= 0.15 * sw and abs(sub_y_norm - mean_y_norm) < single_h * 0.55
                )
                if same_horiz_row:
                    # Kiểm tra có trùng lặp ngang với mảnh đã có trong dòng không
                    is_dup = False
                    for idx_c, (cyn, cyr, cx1, cx2, cpoly, ccrop) in enumerate(cluster):
                        ew = max(1.0, cx2 - cx1)
                        x_ov = min(sx2, cx2) - max(sx1, cx1)
                        if x_ov > 0.42 * min(sw, ew) and abs(sub_y_norm - cyn) < single_h * 0.40:
                            # Giữ mảnh rộng hơn
                            if sw > ew * 1.08:
                                cluster[idx_c] = item
                            is_dup = True
                            break
                    if not is_dup:
                        cluster.append(item)
                    placed = True
                    break
            if not placed:
                line_clusters.append([item])

        # Sắp xếp các dòng theo Y đã bù nghiêng và các mảnh trong từng dòng theo X từ trái sang phải
        line_clusters.sort(key=lambda cl: float(np.mean([c[0] for c in cl])))

        # Bước 3: Chuẩn bị danh sách ảnh từng dòng đưa vào VietOCR theo batch (giữ nguyên toàn dòng để mô hình seq2seq có ngữ cảnh đầy đủ)
        line_polys: list[np.ndarray] = []
        line_y_norms: list[float] = []
        line_piece_ranges: list[tuple[int, int]] = []
        all_pil_pieces: list[Image.Image] = []

        for cluster in line_clusters:
            cluster.sort(key=lambda c: c[2])
            # Lọc bỏ mảnh cách quá xa ở rìa phải (chữ nháp lề phải)
            filtered_cl = []
            for item in cluster:
                sub_y_norm, sub_y_raw, sx1, sx2, sub_poly, sub_crop = item
                if filtered_cl and (sx1 - filtered_cl[-1][3]) > width * 0.22 and (sx2 - sx1) < 220:
                    continue
                filtered_cl.append(item)
            if not filtered_cl:
                continue

            # Gộp polygon của các mảnh trên cùng 1 dòng (giữ đúng chiều cao vuông góc thực tế của dòng)
            if len(filtered_cl) == 1:
                merged_poly = filtered_cl[0][4]
            else:
                p_first = filtered_cl[0][4]
                p_last = filtered_cl[-1][4]
                merged_poly = np.array([p_first[0], p_last[1], p_last[2], p_first[3]], dtype=np.float32)

            cluster_y_norm = float(np.mean([c[0] for c in filtered_cl]))
            start_idx = len(all_pil_pieces)
            for _, _, _, _, _, seg_crop in filtered_cl:
                wch, wcw = seg_crop.shape[:2]
                # Giữ nguyên toàn bộ 1 dòng để VietOCR (vgg_seq2seq) đọc trọn vẹn ngữ cảnh từ đầu đến cuối dòng,
                # tuyệt đối không cắt ngang làm gãy đôi từ hoặc rơi chữ ở giữa dòng.
                nw = min(960, max(32, int(round(wcw * (32.0 / max(1, wch))))))
                rs_crop = cv2.resize(seg_crop, (nw, 32), interpolation=cv2.INTER_AREA)
                all_pil_pieces.append(Image.fromarray(cv2.cvtColor(rs_crop, cv2.COLOR_BGR2RGB)))
            end_idx = len(all_pil_pieces)
            if end_idx > start_idx:
                line_polys.append(merged_poly)
                line_y_norms.append(cluster_y_norm)
                line_piece_ranges.append((start_idx, end_idx))

        if not all_pil_pieces:
            return []

        n_pieces = len(all_pil_pieces)
        indexed_pieces = list(enumerate(all_pil_pieces))
        indexed_pieces.sort(key=lambda item: item[1].size[0])
        piece_texts = [""] * n_pieces
        piece_probs = [0.92] * n_pieces
        batch_size = 8
        for b_start in range(0, n_pieces, batch_size):
            chunk = indexed_pieces[b_start:b_start + batch_size]
            chunk_imgs = [it[1] for it in chunk]
            try:
                if torch is not None:
                    with torch.inference_mode():
                        res_batch = self.viet_recognizer.predict_batch(chunk_imgs, return_prob=True)
                else:
                    res_batch = self.viet_recognizer.predict_batch(chunk_imgs, return_prob=True)
                chunk_txts, chunk_prbs = res_batch
            except Exception:
                if torch is not None:
                    with torch.inference_mode():
                        chunk_txts = self.viet_recognizer.predict_batch(chunk_imgs)
                else:
                    chunk_txts = self.viet_recognizer.predict_batch(chunk_imgs)
                chunk_prbs = [0.90] * len(chunk_txts)

            for (orig_idx, _), txt, prb in zip(chunk, chunk_txts, chunk_prbs):
                piece_texts[orig_idx] = (txt or "").strip()
                try:
                    p_val = float(prb)
                    piece_probs[orig_idx] = p_val if not np.isnan(p_val) else 0.85
                except Exception:
                    piece_probs[orig_idx] = 0.85

        try:
            from vietnamese_syllable_validator import invalid_syllables
        except ImportError:
            from worker.vietnamese_syllable_validator import invalid_syllables

        out = []
        for poly, y_norm_val, (s_i, e_i) in zip(line_polys, line_y_norms, line_piece_ranges):
            valid_sub_txts = []
            valid_sub_probs = []
            n_sub = e_i - s_i
            for idx_p in range(s_i, e_i):
                t_sub = piece_texts[idx_p]
                p_sub = piece_probs[idx_p]
                if not t_sub:
                    continue
                if is_gibberish_or_handwriting_text(t_sub):
                    continue
                if p_sub < 0.16 and (n_sub > 1 and idx_p == e_i - 1) and not t_sub.isupper():
                    continue
                valid_sub_txts.append(t_sub)
                valid_sub_probs.append(p_sub)

            if not valid_sub_txts:
                continue
            full_line_text = " ".join(valid_sub_txts).strip()
            if not full_line_text or is_gibberish_or_handwriting_text(full_line_text):
                continue
            avg_prob = float(np.mean(valid_sub_probs)) if valid_sub_probs else 0.90
            if avg_prob < 0.35 or (avg_prob < 0.45 and len(full_line_text) < 12 and not full_line_text.isdigit()):
                continue
            # Lọc bỏ dòng bóng chữ thấu mực có độ tin cậy thấp (conf < 0.58) đi kèm tỷ lệ từ vô nghĩa / số rời rạc cao
            if avg_prob < 0.58 and not full_line_text.isdigit():
                words_line = full_line_text.split()
                alpha_words = [w for w in words_line if any(ch.isalpha() for ch in w)]
                inv_cnt = len(invalid_syllables(full_line_text, allow_loanwords=True))
                vi_score = 1.0 - (inv_cnt / float(max(1, len(alpha_words))))
                num_toks = sum(1 for w in words_line if any(ch.isdigit() for ch in w))
                if vi_score < 0.45 or (len(words_line) >= 4 and num_toks >= 3 and vi_score < 0.65):
                    continue
            out.append([np.asarray(poly).tolist(), (full_line_text, round(avg_prob, 3), float(y_norm_val))])
        return out

    def _run_paddle_ocr(
        self,
        img: np.ndarray,
        page_number: int,
        width: int,
        height: int,
        yolo_regions: list[dict] | None = None,
        table_ref_img: np.ndarray | None = None,
    ) -> dict:
        """Thực thi PaddleOCR và nhóm các dòng chữ thành khối layout trực tiếp từ ảnh quét."""
        # PaddleOCR hỗ trợ nhận ndarray trực tiếp
        if self.viet_recognizer is not None:
            results = [self._detect_and_recognize_vietocr(img, width)]
        else:
            results = self.ocr_instance.ocr(img, cls=False)

        raw_boxes = []
        if results and len(results) > 0 and results[0] is not None:
            for line in results[0]:
                poly, rec_info = line
                if len(rec_info) == 3:
                    text, confidence, y_norm_line = rec_info
                else:
                    text, confidence = rec_info
                    y_norm_line = None
                pts_poly = _order_poly(poly)
                xs = pts_poly[:, 0]
                ys = pts_poly[:, 1]
                # Dùng chiều cao vuông góc thực tế của dòng quanh tâm Y (tránh bị phình y2-y1 khi dòng hơi nghiêng)
                lh = 0.5 * (float(np.linalg.norm(pts_poly[0] - pts_poly[3])) + float(np.linalg.norm(pts_poly[1] - pts_poly[2])))
                lh = max(12.0, lh)
                ym = float(np.mean(ys))
                x1, x2 = int(round(float(min(xs)))), int(round(float(max(xs))))
                y1, y2 = int(round(ym - lh * 0.5)), int(round(ym + lh * 0.5))

                # Clamp coordinates
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))

                raw_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "height_px": y2 - y1,
                    "y_norm": float(y_norm_line) if y_norm_line is not None else float(ym),
                })

        # Sắp xếp các dòng theo vị trí Y đã bù góc nghiêng tăng dần
        raw_boxes.sort(key=lambda b: (b.get("y_norm", b["bbox"][1]), b["bbox"][0]))

        # Ánh xạ class_name của YOLO → block_type của Doc Forge
        _YOLO_CLASS_MAP = {
            "title": "heading",
            "heading": "heading",
            "table": "table",
            "figure": "paragraph",
            "list": "list",
            "text": "paragraph",
            "paragraph": "paragraph",
        }

        def _block_type_from_yolo(bbox: list[int]) -> str | None:
            """Trả về block_type từ YOLO region có overlap lớn nhất với bbox."""
            if not yolo_regions:
                return None
            bx1, by1, bx2, by2 = bbox
            best_type, best_overlap = None, 0.0
            for region in yolo_regions:
                rx1, ry1, rx2, ry2 = region["bbox"]
                ix1, iy1 = max(bx1, rx1), max(by1, ry1)
                ix2, iy2 = min(bx2, rx2), min(by2, ry2)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = (ix2 - ix1) * (iy2 - iy1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        raw_cls = region["class_name"].lower()
                        best_type = _YOLO_CLASS_MAP.get(raw_cls, "paragraph")
            return best_type

        # Phân tích để gán loại block (dùng YOLO nếu có, fallback heuristics chiều cao)
        blocks = []
        avg_height = np.mean([b["height_px"] for b in raw_boxes]) if raw_boxes else 20.0

        # Tính toán lề trang chuẩn hóa từ các dòng rộng thực sự thuộc cột chữ chính
        substantial_boxes = [b for b in raw_boxes if (b["bbox"][2] - b["bbox"][0]) > width * 0.32 and b["bbox"][0] > width * 0.05]
        if not substantial_boxes:
            substantial_boxes = [b for b in raw_boxes if (b["bbox"][2] - b["bbox"][0]) > width * 0.25]
        if substantial_boxes:
            left_coords = sorted([b["bbox"][0] for b in substantial_boxes])
            dominant_left = left_coords[int(len(left_coords) * 0.15)]
            right_coords = sorted([b["bbox"][2] for b in substantial_boxes])
            dominant_right = right_coords[int(len(right_coords) * 0.88)]
        else:
            dominant_left = width * 0.15
            dominant_right = width * 0.85

        # Lọc sạch các mảnh chữ rác ở gáy sách mép trái (trang kế bên) và chữ nháp lề phải
        valid_boxes = []
        for b in raw_boxes:
            bx1, by1, bx2, by2 = b["bbox"]
            bw = bx2 - bx1
            if bx2 < max(120, dominant_left - 25):
                continue
            if bx1 > dominant_right - 15 and bw < 220:
                continue
            if is_gibberish_or_handwriting_text(b["text"]):
                continue
            valid_boxes.append(b)
        raw_boxes = valid_boxes

        col_width = max(100, dominant_right - dominant_left)
        col_center = (dominant_left + dominant_right) / 2

        left_m = min([b["bbox"][0] for b in raw_boxes]) if raw_boxes else 0
        top_m = min([b["bbox"][1] for b in raw_boxes]) if raw_boxes else 0
        right_m = max([b["bbox"][2] for b in raw_boxes]) if raw_boxes else width
        bottom_m = max([b["bbox"][3] for b in raw_boxes]) if raw_boxes else height

        page_margins = {
            "top": round((top_m / height) * 100, 2),
            "bottom": round(((height - bottom_m) / height) * 100, 2),
            "left": round((left_m / width) * 100, 2),
            "right": round(((width - right_m) / width) * 100, 2),
        }

        # Chạy batch inference DeepDocumentTypographyResNet + phát hiện nét gạch chân (underline)
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        def _has_underline_stroke(bx1: int, by1: int, bx2: int, by2: int) -> bool:
            ch = by2 - by1
            cw = bx2 - bx1
            if ch < 12 or cw < 75:
                return False
            y_start = max(0, by1 + int(ch * 0.65))
            y_end = min(height, by2 + max(3, int(ch * 0.14)))
            strip = gray_full[y_start:y_end, max(0, bx1):min(width, bx2)]
            if strip.size == 0 or strip.shape[0] < 3 or strip.shape[1] < 50:
                return False
            _, bw_inv = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            k_len = max(45, int(cw * 0.42))
            horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
            horiz_lines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, horiz_k)
            row_max = int(horiz_lines.max(axis=0).astype(bool).sum()) if horiz_lines.size > 0 else 0
            return row_max >= int(cw * 0.48)

        def _measure_crop_stroke_and_slant(crop_g: np.ndarray) -> tuple[float, float, float]:
            """
            Trả về (stroke_width_ratio, slant_deg, serif_ratio) bằng hình học pixel trực tiếp:
            - stroke_width_ratio: tỉ lệ diện tích mực / biên chữ (đo độ đậm Bold)
            - slant_deg: góc nghiêng nét dọc tối ưu (đo độ nghiêng Italic, dương khi nghiêng phải)
            - serif_ratio: mật độ chân chữ ngang ở đáy/đỉnh thân chữ (phân biệt Serif vs Sans-serif)
            """
            if crop_g.size == 0 or crop_g.shape[0] < 8 or crop_g.shape[1] < 12:
                return 1.0, 0.0, 0.95
            _, bw_i = cv2.threshold(crop_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ink_mask = (bw_i > 0).astype(np.uint8)
            ink_sum = float(ink_mask.sum())
            if ink_sum < 15.0:
                return 1.0, 0.0, 0.95
            edges = cv2.Canny(bw_i, 50, 150)
            edge_sum = max(1.0, float((edges > 0).sum()))
            sw_ratio = ink_sum / edge_sum

            ch_c, cw_c = ink_mask.shape[:2]
            if cw_c > 220:
                ink_sub = ink_mask[:, int(cw_c * 0.05):int(cw_c * 0.05) + 200]
            else:
                ink_sub = ink_mask
            sh_h, sh_w = ink_sub.shape[:2]
            best_deg = 0.0
            best_var = -1.0
            ys_idx = np.arange(sh_h, dtype=np.float32)[:, None]
            xs_idx = np.arange(sh_w, dtype=np.float32)[None, :]
            for deg in (-2.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0):
                tan_a = float(np.tan(np.deg2rad(deg)))
                shifted_x = np.clip(np.round(xs_idx + (ys_idx - sh_h * 0.5) * tan_a).astype(np.int32), 0, sh_w - 1)
                col_proj = ink_sub[np.arange(sh_h)[:, None], shifted_x].sum(axis=0).astype(np.float32)
                v = float(np.var(col_proj))
                if v > best_var:
                    best_var = v
                    best_deg = deg

            dist_map = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 3)
            dist_vals = dist_map[ink_mask > 0]
            if dist_vals.size >= 15:
                # Chữ có chân (Times New Roman) có độ biến thiên độ dày nét (nét thanh - nét đậm + chân chữ) cao hơn chữ không chân (Arial)
                cv_dist = float(np.std(dist_vals) / max(1e-4, float(np.mean(dist_vals))))
                serif_r = cv_dist * 2.5
            else:
                serif_r = 0.90
            return sw_ratio, best_deg, serif_r

        # Đo thông số nét chữ trực tiếp cho toàn bộ các dòng trên trang
        visual_metrics = []
        for b in raw_boxes:
            bx1, by1, bx2, by2 = b["bbox"]
            crop_g = gray_full[max(0, by1):min(height, by2), max(0, bx1):min(width, bx2)]
            visual_metrics.append(_measure_crop_stroke_and_slant(crop_g))

        body_metrics = [
            vm for vm, b in zip(visual_metrics, raw_boxes)
            if (b["bbox"][2] - b["bbox"][0]) > width * 0.40 and b["height_px"] < height * 0.045
        ]
        if body_metrics:
            median_page_sw = float(np.median([vm[0] for vm in body_metrics]))
            median_page_slant = float(np.median([vm[1] for vm in body_metrics]))
            median_page_serif = float(np.median([vm[2] for vm in body_metrics]))
        else:
            median_page_sw = float(np.median([vm[0] for vm in visual_metrics])) if visual_metrics else 1.4
            median_page_slant = float(np.median([vm[1] for vm in visual_metrics])) if visual_metrics else 0.0
            median_page_serif = float(np.median([vm[2] for vm in visual_metrics])) if visual_metrics else 0.95

        page_font_family = "Times New Roman" if median_page_serif >= 0.82 else "Arial"

        body_h_list = [
            b["height_px"] for b in raw_boxes
            if (b["bbox"][2] - b["bbox"][0]) > width * 0.40 and b["height_px"] < height * 0.045
        ]
        median_body_h = float(np.median(body_h_list)) if body_h_list else float(avg_height)

        typo_predictions = []
        if self.typo_model is not None and torch is not None and raw_boxes:
            try:
                batch_tensors = []
                for b in raw_boxes:
                    bx1, by1, bx2, by2 = b["bbox"]
                    crop = gray_full[max(0, by1):min(height, by2), max(0, bx1):min(width, bx2)]
                    if crop.size == 0:
                        canvas = np.full((32, 128), 255, dtype=np.uint8)
                    else:
                        ch, cw = crop.shape[:2]
                        scale = min(28.0 / max(1, ch), 120.0 / max(1, cw))
                        nw, nh = max(4, int(cw * scale)), max(4, int(ch * scale))
                        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                        canvas = np.full((32, 128), 255, dtype=np.uint8)
                        oy, ox = (32 - nh) // 2, (128 - nw) // 2
                        canvas[oy:oy + nh, ox:ox + nw] = resized
                    batch_tensors.append(canvas.astype(np.float32) / 255.0)
                t_in = torch.from_numpy(np.stack(batch_tensors, axis=0)).unsqueeze(1).to(self.typo_device)
                with torch.inference_mode():
                    out_tuple = self.typo_model(t_in)
                    p_bold, p_italic, p_serif, p_size = out_tuple[0], out_tuple[1], out_tuple[2], out_tuple[3]
                    bold_probs = torch.softmax(p_bold, dim=1)[:, 1].cpu().tolist()
                    italic_probs = torch.softmax(p_italic, dim=1)[:, 1].cpu().tolist()
                    serif_probs = torch.softmax(p_serif, dim=1)[:, 1].cpu().tolist()
                    sizes = p_size.cpu().tolist()
                ai_body_serifs = [
                    sp for idx_t, sp in enumerate(serif_probs)
                    if idx_t < len(raw_boxes) and (raw_boxes[idx_t]["bbox"][2] - raw_boxes[idx_t]["bbox"][0]) > width * 0.40
                ]
                median_ai_serif = float(np.median(ai_body_serifs)) if ai_body_serifs else (float(np.median(serif_probs)) if serif_probs else 0.5)
                doc_serif_family = (
                    "Times New Roman"
                    if (median_ai_serif >= 0.48 or (median_ai_serif >= 0.35 and median_page_serif >= 0.85))
                    else "Arial"
                )
                for idx_t, (b_prob, i_prob, s_prob, sz) in enumerate(zip(bold_probs, italic_probs, serif_probs, sizes)):
                    sw_r, slant_deg, serif_r = visual_metrics[idx_t] if idx_t < len(visual_metrics) else (median_page_sw, median_page_slant, median_page_serif)
                    rel_slant = slant_deg - median_page_slant
                    is_bold_fused = (b_prob >= 0.85) or (b_prob >= 0.65 and sw_r >= median_page_sw * 1.03) or (sw_r >= median_page_sw * 1.18)
                    is_italic_fused = (i_prob >= 0.85) or (i_prob >= 0.65 and rel_slant >= 2.5) or (rel_slant >= 6.0)
                    line_font = (
                        "Times New Roman"
                        if (0.55 * s_prob + 0.45 * float(doc_serif_family == "Times New Roman")) >= 0.50
                        else "Arial"
                    )
                    typo_predictions.append({
                        "bold": bool(is_bold_fused),
                        "italic": bool(is_italic_fused),
                        "fontFamily": line_font,
                        "fontSizePt": round(max(11.5, min(18.0, float(sz))), 1),
                        "sw_ratio": sw_r,
                        "slant_deg": slant_deg,
                        "rel_slant": rel_slant,
                    })
            except Exception as exc:
                LOG.debug("Typography batch inference fallback: %s", exc)
                typo_predictions = []

        # ── Phát hiện & tái tạo cấu trúc Bảng biểu 2D tự động (OpenCV Grid + Cell Clustering) ──
        detected_tables = []
        consumed_box_indices = set()

        # (a) Bảng bằng mạng nơ-ron TableLineNet tự huấn luyện:
        # Xóa đường viền bảng bằng vạch ngăn nền (white barrier) theo đúng lưới h_lines/v_lines
        # rồi cho DBNet + VietOCR đọc từng ô độc lập — không dính cột và không tạo chữ ảo ở khoảng trống.
        if getattr(self, "table_model", None) is not None and raw_boxes:
            try:
                if table_ref_img is not None and table_ref_img.shape[:2] == (height, width):
                    tbl_src_img = table_ref_img
                else:
                    tbl_src_img = img
                gray_tbl = cv2.cvtColor(tbl_src_img, cv2.COLOR_BGR2GRAY) if tbl_src_img.ndim == 3 else tbl_src_img
                h_prob, v_prob = predict_line_maps(self.table_model, gray_tbl, self.typo_device)
                sx_n, sy_n = NET_W / float(width), NET_H / float(height)
                for grid in extract_tables(h_prob, v_prob):
                    if grid.n_rows < 2 or grid.n_cols < 2 or grid.n_cols > 12:
                        continue
                    gx0, gy0, gx1, gy1 = grid.bbox
                    tx0, ty0, tx1, ty1 = int(gx0 / sx_n), int(gy0 / sy_n), int(gx1 / sx_n), int(gy1 / sy_n)
                    matched_g = []
                    for idx_b, b_item in enumerate(raw_boxes):
                        if idx_b in consumed_box_indices:
                            continue
                        bx1, by1, bx2, by2 = b_item["bbox"]
                        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        if (tx0 - 16) <= bcx <= (tx1 + 16) and (ty0 - 10) <= bcy <= (ty1 + 10):
                            matched_g.append(idx_b)

                    # Tạo ảnh vùng bảng đã tách ranh giới cột/hàng bằng màu nền giấy
                    clean_tbl = img.copy()
                    bg_vals = np.percentile(clean_tbl[max(0, ty0):min(height, ty1), max(0, tx0):min(width, tx1)], 88, axis=(0, 1))
                    bg_color = tuple(int(v) for v in bg_vals) if clean_tbl.ndim == 3 else int(bg_vals)
                    x_pad0, y_pad0 = max(0, tx0 - 12), max(0, ty0 - 12)
                    x_pad1, y_pad1 = min(width, tx1 + 12), min(height, ty1 + 12)
                    for a_l, b_l in grid.h_lines:
                        p1 = (x_pad0, int(round((a_l * (x_pad0 * sx_n) + b_l) / sy_n)))
                        p2 = (x_pad1, int(round((a_l * (x_pad1 * sx_n) + b_l) / sy_n)))
                        cv2.line(clean_tbl, p1, p2, bg_color, thickness=14)
                    for a_l, b_l in grid.v_lines:
                        p1 = (int(round((a_l * (y_pad0 * sy_n) + b_l) / sx_n)), y_pad0)
                        p2 = (int(round((a_l * (y_pad1 * sy_n) + b_l) / sx_n)), y_pad1)
                        cv2.line(clean_tbl, p1, p2, bg_color, thickness=16)

                    crop_tbl = clean_tbl[y_pad0:y_pad1, x_pad0:x_pad1]
                    cell_dt_boxes, _ = self.ocr_instance.text_detector(crop_tbl)
                    crops_for_rec = []
                    coords_for_rec = []
                    if cell_dt_boxes is not None:
                        for c_box in cell_dt_boxes:
                            pts = _order_poly(np.asarray(c_box, dtype=np.float32))
                            pts[:, 0] += x_pad0
                            pts[:, 1] += y_pad0
                            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                            if not grid.contains(cx * sx_n, cy * sy_n, margin=6.0):
                                continue
                            r_i, c_i = grid.locate(cx * sx_n, cy * sy_n)
                            c_img = _rotate_crop(clean_tbl, pts, pad_x=6, pad_y=None)
                            ch_c, cw_c = c_img.shape[:2]
                            if ch_c < 8 or cw_c < 8:
                                continue
                            nw_c = min(512, max(32, int(round(cw_c * (32.0 / max(1, ch_c))))))
                            rs_c = cv2.resize(c_img, (nw_c, 32), interpolation=cv2.INTER_AREA)
                            crops_for_rec.append(Image.fromarray(cv2.cvtColor(rs_c, cv2.COLOR_BGR2RGB)))
                            coords_for_rec.append((r_i, c_i, float(pts[:, 1].min()), float(pts[:, 0].min())))

                    cell_buf = [[[] for _ in range(grid.n_cols)] for _ in range(grid.n_rows)]
                    if crops_for_rec and self.viet_recognizer is not None:
                        with torch.inference_mode():
                            rec_texts = self.viet_recognizer.predict_batch(crops_for_rec)
                        for (r_i, c_i, y_v, x_v), txt_r in zip(coords_for_rec, rec_texts):
                            t_clean = normalize_vietnamese_text(txt_r or "").strip()
                            if t_clean:
                                cell_buf[r_i][c_i].append((y_v, x_v, t_clean))

                    rows_g = [
                        [" ".join(it[2] for it in sorted(cell_buf[r_i][c_i], key=lambda k: (k[0], k[1]))).strip() for c_i in range(grid.n_cols)]
                        for r_i in range(grid.n_rows)
                    ]
                    # Bù an toàn từ raw_boxes nếu ô có box DBNet nằm gọn bên trong nhưng bị sót
                    for idx_b in matched_g:
                        bx1, by1, bx2, by2 = raw_boxes[idx_b]["bbox"]
                        r_i, c_i = grid.locate((bx1 + bx2) / 2.0 * sx_n, (by1 + by2) / 2.0 * sy_n)
                        q_rc = grid.cell_quad(r_i, c_i)
                        col_w_img = max(1.0, (q_rc[1][0] - q_rc[0][0]) / sx_n)
                        if not rows_g[r_i][c_i] and (bx2 - bx1) <= col_w_img * 1.15:
                            fallback_t = normalize_vietnamese_text(raw_boxes[idx_b]["text"]).strip()
                            if fallback_t:
                                rows_g[r_i][c_i] = fallback_t

                    non_empty_cells = sum(1 for row in rows_g for val in row if val)
                    if non_empty_cells < 3:
                        continue
                    for idx_b in matched_g:
                        consumed_box_indices.add(idx_b)
                    detected_tables.append({
                        "type": "table",
                        "bbox": [tx0, ty0, tx1, ty1],
                        "text": "\n".join(" | ".join(r) for r in rows_g),
                        "confidence": 0.98,
                        "style": {"fontFamily": page_font_family, "fontSizePt": 12.0, "textAlign": "left", "bold": False},
                        "tableData": {"headers": rows_g[0], "rows": rows_g[1:]},
                        "is_indented": False,
                        "standalone_line": True,
                    })
                    LOG.info("  ✓ TableLineNet (Barrier DBNet+VietOCR): bảng %dx%d ô tại y=%d..%d.", grid.n_rows, grid.n_cols, ty0, ty1)
            except Exception as net_tbl_exc:
                LOG.warning("TableLineNet bỏ qua: %s", net_tbl_exc)

        def _detect_tables_opencv_fallback():
            _, bw_table = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(75, width // 9), 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(42, height // 22)))
            h_lines_img = cv2.morphologyEx(bw_table, cv2.MORPH_OPEN, h_kernel)
            v_lines_img = cv2.morphologyEx(bw_table, cv2.MORPH_OPEN, v_kernel)
            grid_img = cv2.bitwise_or(h_lines_img, v_lines_img)
            grid_dilated = cv2.dilate(grid_img, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
            tbl_cnts, _ = cv2.findContours(grid_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            def _cluster_1d(coords: list[int], min_gap: int = 18) -> list[int]:
                if not coords:
                    return []
                coords = sorted(coords)
                clusters = [[coords[0]]]
                for c_val in coords[1:]:
                    if c_val - clusters[-1][-1] <= min_gap:
                        clusters[-1].append(c_val)
                    else:
                        clusters.append([c_val])
                return [int(np.mean(cl)) for cl in clusters]

            for cnt in tbl_cnts:
                tx, ty, tw, th = cv2.boundingRect(cnt)
                if tw < width * 0.38 or th < max(65, height * 0.05):
                    continue
                sub_h = h_lines_img[ty:ty + th, tx:tx + tw]
                sub_v = v_lines_img[ty:ty + th, tx:tx + tw]
                h_ys = np.where(sub_h.sum(axis=1) > tw * 0.25 * 255)[0].tolist()
                v_xs = np.where(sub_v.sum(axis=0) > th * 0.22 * 255)[0].tolist()
                row_lines = _cluster_1d([ty + y_rel for y_rel in h_ys], min_gap=max(16, int(median_body_h * 0.55)))
                col_lines = _cluster_1d([tx + x_rel for x_rel in v_xs], min_gap=max(32, width // 25))
                if len(row_lines) < 2 or len(col_lines) < 2:
                    continue
                if row_lines[0] > ty + 20:
                    row_lines.insert(0, ty)
                if row_lines[-1] < ty + th - 20:
                    row_lines.append(ty + th)
                if col_lines[0] > tx + 25:
                    col_lines.insert(0, tx)
                if col_lines[-1] < tx + tw - 25:
                    col_lines.append(tx + tw)

                n_rows = len(row_lines) - 1
                n_cols = len(col_lines) - 1
                if n_rows < 2 or n_cols < 2 or n_cols > 12:
                    continue

                cells_grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
                matched_in_tbl = []
                for idx_b, b_item in enumerate(raw_boxes):
                    if idx_b in consumed_box_indices:
                        continue
                    bx1, by1, bx2, by2 = b_item["bbox"]
                    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if tx - 12 <= cx <= tx + tw + 12 and ty - 8 <= cy <= ty + th + 8:
                        r_idx = max(0, min(n_rows - 1, int(np.searchsorted(row_lines[1:], cy))))
                        c_idx = max(0, min(n_cols - 1, int(np.searchsorted(col_lines[1:], cx))))
                        clean_cell_txt = normalize_vietnamese_text(b_item["text"]).strip()
                        if clean_cell_txt:
                            cells_grid[r_idx][c_idx].append((by1, bx1, clean_cell_txt))
                        matched_in_tbl.append(idx_b)

                if len(matched_in_tbl) < 3:
                    continue

                rendered_rows = []
                for r_i in range(n_rows):
                    row_vals = []
                    for c_i in range(n_cols):
                        items_rc = sorted(cells_grid[r_i][c_i], key=lambda it: (it[0], it[1]))
                        row_vals.append(" ".join(it[2] for it in items_rc).strip())
                    if any(row_vals):
                        rendered_rows.append(row_vals)

                if len(rendered_rows) >= 2:
                    for idx_b in matched_in_tbl:
                        consumed_box_indices.add(idx_b)
                    headers = rendered_rows[0]
                    data_rows = rendered_rows[1:]
                    flat_lines = [" | ".join(headers)] + [" | ".join(r) for r in data_rows]
                    detected_tables.append({
                        "type": "table",
                        "bbox": [tx, ty, tx + tw, ty + th],
                        "text": "\n".join(flat_lines),
                        "confidence": 0.96,
                        "style": {
                            "fontFamily": page_font_family,
                            "fontSizePt": 12.0,
                            "textAlign": "left",
                            "bold": False,
                        },
                        "tableData": {
                            "headers": headers,
                            "rows": data_rows,
                        },
                        "is_indented": False,
                        "standalone_line": True,
                    })
                    LOG.info("  ✓ Tự động phát hiện & tái tạo Bảng biểu (%dx%d ô) tại y=%d..%d.", len(rendered_rows), n_cols, ty, ty + th)

        if not detected_tables:
            try:
                _detect_tables_opencv_fallback()
            except Exception as tbl_exc:
                LOG.debug("Phát hiện bảng tự động bỏ qua: %s", tbl_exc)

        import re
        line_items = []
        for idx_b, b in enumerate(raw_boxes):
            if idx_b in consumed_box_indices:
                continue
            bx1, by1, bx2, by2 = b["bbox"]
            b_width = bx2 - bx1
            b_center = (bx1 + bx2) / 2

            normalized_text = normalize_vietnamese_text(b["text"]).strip()
            if not normalized_text or is_gibberish_or_handwriting_text(normalized_text):
                continue
            if len(normalized_text) >= 6 and (normalized_text.count("0") / len(normalized_text)) > 0.55:
                continue

            # Bỏ qua mảnh chữ nháp rất ngắn nằm lơ lửng không phải số trang hay tiêu đề
            is_page_num = len(normalized_text) <= 4 and normalized_text.isdigit() and by1 > height * 0.80
            is_chapter_prefix = bool(re.match(r"^(Chương|CHƯƠNG|Phần|PHẦN)\s+\d+", normalized_text))
            is_top_header_piece = normalized_text.isupper() and by1 < height * 0.25 and len(normalized_text) >= 2
            if (
                b_width < col_width * 0.16
                and not is_page_num
                and not is_chapter_prefix
                and not is_top_header_piece
                and len(normalized_text) < 12
                and b["confidence"] < 0.82
            ):
                continue

            cur_y_norm = float(b.get("y_norm", 0.5 * (by1 + by2)))
            # Lọc bỏ dòng trùng lặp thực sự (cùng toạ độ Y đã bù nghiêng và chồng lấp ngang > 45%)
            if line_items:
                prev_bx1, prev_by1, prev_bx2, prev_by2 = line_items[-1]["bbox"]
                prev_y_norm = float(line_items[-1].get("y_norm", 0.5 * (prev_by1 + prev_by2)))
                x_ov_line = min(bx2, prev_bx2) - max(bx1, prev_bx1)
                min_w_line = max(1.0, float(min(bx2 - bx1, prev_bx2 - prev_bx1)))
                min_h_line = max(1.0, float(min(by2 - by1, prev_by2 - prev_by1)))
                if abs(cur_y_norm - prev_y_norm) < 0.42 * min_h_line and x_ov_line > 0.45 * min_w_line:
                    continue

            sw_r, slant_deg, serif_r = visual_metrics[idx_b] if idx_b < len(visual_metrics) else (median_page_sw, median_page_slant, median_page_serif)
            rel_slant = slant_deg - median_page_slant
            typo = typo_predictions[idx_b] if idx_b < len(typo_predictions) else {
                "bold": bool(sw_r >= median_page_sw * 1.18),
                "italic": bool(rel_slant >= 6.0),
                "fontFamily": page_font_family,
                "fontSizePt": 13.0,
                "sw_ratio": sw_r,
                "slant_deg": slant_deg,
                "rel_slant": rel_slant,
            }
            has_ul = _has_underline_stroke(bx1, by1, bx2, by2)

            upper_txt = normalized_text.upper()
            lower_txt = normalized_text.lower()
            left_gap = bx1 - dominant_left
            right_gap = dominant_right - bx2
            is_uppercase = normalized_text.isupper() and len(normalized_text) > 3
            is_centered = (
                left_gap >= col_width * 0.08
                and right_gap >= col_width * 0.08
                and abs(left_gap - right_gap) < col_width * 0.14
                and b_width < col_width * 0.78
                and (is_uppercase or is_chapter_prefix or is_page_num or "độc lập" in lower_txt or "cộng hòa" in lower_txt)
            )
            diff_from_left = bx1 - dominant_left
            is_indented = (diff_from_left >= max(34, col_width * 0.032)) and (diff_from_left <= max(240, col_width * 0.22))

            # Phân loại chính xác cấp độ Tiêu đề (Heading Levels 0, 1, 2, 3) & Đề mục nghiêng
            is_level3_heading = bool(re.match(r"^\d+\.\d+\.\d+\.?\s*\S+", normalized_text))
            is_level2_heading = (not is_level3_heading) and bool(re.match(r"^\d+\.\d+\.?\s*\S+", normalized_text))
            is_level1_heading = (not is_level2_heading and not is_level3_heading) and bool(
                re.match(r"^(\d+|[IVX]+)\.\s*[A-ZĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]{2,}", normalized_text)
            )
            is_sec_heading = is_level1_heading or is_level2_heading or is_level3_heading

            is_main_title = (
                is_chapter_prefix
                or any(
                    kw in upper_txt
                    for kw in ["GIẤY XÁC NHẬN", "TỜ KHAI", "QUYẾT ĐỊNH", "THÔNG BÁO", "TỜ TRÌNH", "HỢP ĐỒNG", "CÁC HOẠT ĐỘNG CHÍNH"]
                )
                or (is_uppercase and not is_sec_heading and by1 < height * 0.35)
            )

            is_admin_header = (
                any(kw in upper_txt for kw in ["CỘNG HÒA XÃ HỘI", "ĐỘC LẬP - TỰ DO"])
                or bool(re.match(r"^PHẦN\s+(?:THỨ\s+)?(?:\d+|[IVX]+)\b", upper_txt))
                or is_main_title
            )

            # Đề mục con in nghiêng đứng riêng một dòng (ví dụ: "Về kinh tế:", "Về chính trị - xã hội:", "Thứ nhất, về điểm tương đồng")
            is_sub_italic_header = (
                bool(
                    re.match(
                        r"^(?:[-+•]\s*)?(?:Về\s+[^:]{2,48}:?|Thứ\s+(?:nhất|hai|ba|tư)[,.]\s+[^.:]{2,65}:?)$",
                        normalized_text,
                        re.IGNORECASE,
                    )
                )
                and b_width < col_width * 0.82
            )

            if is_admin_header or (is_centered and is_uppercase) or is_sec_heading:
                block_type = "heading"
            else:
                block_type = _block_type_from_yolo(b["bbox"])
                if block_type not in ("heading", "table", "list"):
                    block_type = "paragraph"

            # Ước lượng cỡ chữ tỷ lệ thực: đoạn văn chuẩn giữ đồng nhất 13.0pt (tránh nhảy cỡ do dao động khung DBNet),
            # chỉ phóng đại theo pixel thực đối với tiêu đề/khối lớn (ratio_h >= 1.38 hoặc heading) và thu nhỏ cho chú thích/số trang.
            line_h = float(by2 - by1)
            ratio_h = line_h / max(12.0, median_body_h)
            body_pt = 13.0
            pixel_pt = round(min(24.0, max(10.0, body_pt * (ratio_h ** 0.88))) * 2.0) / 2.0
            ai_pt = float(typo.get("fontSizePt", pixel_pt))

            if is_page_num:
                estimated_pt = 11.5
            elif block_type == "heading" or ratio_h >= 1.38:
                estimated_pt = round(min(24.0, max(13.5, 0.72 * pixel_pt + 0.28 * ai_pt)) * 2.0) / 2.0
            elif ratio_h <= 0.68 and b_width < col_width * 0.55:
                estimated_pt = round(max(10.0, min(12.0, pixel_pt)) * 2.0) / 2.0
            else:
                estimated_pt = body_pt

            # Giữ phân cấp tối thiểu cho tiêu đề mục rõ ràng
            if is_main_title:
                estimated_pt = max(estimated_pt, 16.0 if not is_chapter_prefix else 15.5)
            elif is_level1_heading:
                estimated_pt = max(estimated_pt, 14.5)
            elif is_level2_heading:
                estimated_pt = max(estimated_pt, 14.0)
            elif is_level3_heading:
                estimated_pt = max(estimated_pt, 13.5)

            line_font_family = typo.get("fontFamily", page_font_family)
            if b_width < col_width * 0.28:
                line_font_family = page_font_family

            style_meta = {
                "fontFamily": line_font_family,
                "fontSizePt": estimated_pt,
            }

            if is_page_num:
                style_meta["textAlign"] = "center"
                style_meta["bold"] = False
            elif block_type == "heading" or is_centered or is_admin_header:
                style_meta["textAlign"] = "center" if (is_centered or is_admin_header or is_chapter_prefix or is_main_title) else "left"
                style_meta["bold"] = True
                if is_level3_heading or (typo.get("italic", False) and not is_main_title and not is_level1_heading):
                    style_meta["italic"] = True
                if is_indented and style_meta["textAlign"] != "center":
                    style_meta["isIndented"] = True
                if ("độc lập" in lower_txt and "hạnh phúc" in lower_txt) or has_ul:
                    style_meta["underline"] = True
            elif is_sub_italic_header:
                style_meta["textAlign"] = "left"
                style_meta["italic"] = True
                style_meta["isIndented"] = True
            elif "nhà xuất bản" in lower_txt or (any(k in lower_txt for k in ["ngày ", "tháng ", "năm "]) and bx1 > width * 0.36):
                style_meta["textAlign"] = "right" if bx2 > dominant_right - 60 or bx1 > width * 0.36 else "center"
                if "nhà xuất bản" in lower_txt:
                    style_meta["bold"] = True
                else:
                    style_meta["italic"] = True
            else:
                style_meta["textAlign"] = "justify" if b_width > col_width * 0.65 else "left"
                if is_indented:
                    style_meta["isIndented"] = True
                if is_uppercase or normalized_text.startswith("Điều ") or typo.get("bold", False):
                    style_meta["bold"] = True
                if typo.get("italic", False):
                    style_meta["italic"] = True
                if has_ul:
                    style_meta["underline"] = True
                    normalized_text = f"<u>{normalized_text}</u>"

                # Phát hiện cụm mở đầu đoạn văn in nghiêng hoặc in đậm (Inline Lead-in Italic/Bold)
                if not style_meta.get("bold") and not style_meta.get("italic") and "*" not in normalized_text:
                    lead_match = re.match(
                        r"^((?:Phương pháp|Các phương pháp|Về mặt|Một là|Hai là|Ba là|Bốn là|Năm là|Thứ nhất|Thứ hai|-\s*[^:]{2,35})[^.:]{0,45}[.:])(\s+.*)?$",
                        normalized_text,
                    )
                    if lead_match:
                        prefix_str = lead_match.group(1)
                        rest_str = lead_match.group(2) or ""
                        prefix_w_px = int(b_width * min(0.85, len(prefix_str) / max(1.0, float(len(normalized_text)))))
                        sub_crop_g = gray_full[max(0, by1):min(height, by2), max(0, bx1):min(width, bx1 + max(30, prefix_w_px))]
                        p_sw, p_slant, _ = _measure_crop_stroke_and_slant(sub_crop_g)
                        if prefix_str.startswith("-"):
                            normalized_text = f"**{prefix_str}**{rest_str}" if p_sw >= median_page_sw * 1.12 else normalized_text
                        elif (p_slant - median_page_slant) >= 4.5 or prefix_str.lower().startswith(("phương pháp", "các phương pháp", "về mặt")):
                            normalized_text = f"*{prefix_str}*{rest_str}"

            line_items.append({
                "type": block_type,
                "bbox": [bx1, by1, bx2, by2],
                "text": normalized_text,
                "confidence": float(b["confidence"]),
                "style": style_meta,
                "is_indented": is_indented,
                "standalone_line": bool(is_sub_italic_header or is_page_num or is_chapter_prefix),
                "y_norm": cur_y_norm,
            })

        if detected_tables:
            line_items.extend(detected_tables)
            line_items.sort(key=lambda it: (it.get("y_norm", it["bbox"][1]), it["bbox"][0]))

        # Gộp các dòng liên tiếp thuộc cùng một đoạn văn hoặc tiêu đề dài bị tràn dòng (Paragraph & Wrapped-Heading Merging)
        blocks = []
        new_para_prefix = re.compile(
            r"^([-+*•]|\d+[\.\)]|[a-d][\.\)]|Chương\s+\d+|Một là|Hai là|Ba là|Bốn là|Năm là|Thứ nhất|Thứ hai|Về kinh tế|Về chính trị|Về văn hóa|Về xã hội|Như vậy,|Từ thực tiễn)\b",
            re.IGNORECASE,
        )
        for item in line_items:
            if not blocks:
                blk = {
                    "id": f"b-1-{uuid.uuid4().hex[:6]}",
                    "type": item["type"],
                    "order": 1,
                    "bbox": list(item["bbox"]),
                    "text": item["text"],
                    "confidence": round(item["confidence"], 3),
                    "style": dict(item["style"]),
                    "_standalone": item.get("standalone_line", False),
                    "_last_y_norm": float(item.get("y_norm", 0.5 * (item["bbox"][1] + item["bbox"][3]))),
                    "_last_bbox": list(item["bbox"]),
                }
                if "tableData" in item:
                    blk["tableData"] = item["tableData"]
                blocks.append(blk)
                continue

            prev = blocks[-1]
            px1, py1, px2, py2 = prev["bbox"]
            lpx1, lpy1, lpx2, lpy2 = prev.get("_last_bbox", prev["bbox"])
            cx1, cy1, cx2, cy2 = item["bbox"]
            cur_yn = float(item.get("y_norm", 0.5 * (cy1 + cy2)))
            prev_yn = float(prev.get("_last_y_norm", 0.5 * (lpy1 + lpy2)))
            v_gap = min(cy1 - py2, (cur_yn - prev_yn) - median_body_h)
            prev_reached_right = px2 >= (dominant_right - col_width * 0.14)

            # 1. Gộp các mảnh rời rạc nằm trên cùng 1 hàng ngang (vertical overlap >= 50%, ví dụ: "TIN MIỀN BẮC PHÁT ĐÊM")
            v_ov_row = min(lpy2, cy2) - max(lpy1, cy1)
            min_h_row = max(1.0, float(min(lpy2 - lpy1, cy2 - cy1)))
            same_row_cont = (
                (v_ov_row >= 0.50 * min_h_row or abs(cur_yn - prev_yn) < median_body_h * 0.42)
                and cx1 >= lpx2 - 35
                and (cx1 - lpx2) <= width * 0.22
                and "tableData" not in prev
                and "tableData" not in item
            )
            if same_row_cont:
                prev["text"] = f"{prev['text']} {item['text']}"
                prev["bbox"] = [min(px1, cx1), min(py1, cy1), max(px2, cx2), max(py2, cy2)]
                prev["confidence"] = round((prev["confidence"] + item["confidence"]) / 2.0, 3)
                prev["_last_y_norm"] = 0.5 * (prev_yn + cur_yn)
                prev["_last_bbox"] = [min(lpx1, cx1), min(lpy1, cy1), max(lpx2, cx2), max(lpy2, cy2)]
                continue

            # 2. Kiểm tra trường hợp tiêu đề dài (ví dụ 1.2.3..., 3.3..., hoặc tiêu đề IN HOA) tràn sang dòng thứ 2
            is_wrapped_heading_cont = False
            if (
                prev["type"] == "heading"
                and not prev.get("_standalone", False)
                and v_gap <= median_body_h * 0.90
                and not bool(new_para_prefix.match(item["text"]))
            ):
                if prev["text"].isupper() and item["text"].isupper() and prev["style"].get("textAlign") == "center":
                    is_wrapped_heading_cont = True
                elif (
                    prev_reached_right
                    and not item["is_indented"]
                    and (cx2 - cx1) < col_width * 0.55
                    and not prev["text"].endswith((".", ":"))
                ):
                    is_wrapped_heading_cont = True

            if is_wrapped_heading_cont:
                prev["text"] = f"{prev['text']} {item['text']}"
                prev["bbox"] = [min(px1, cx1), min(py1, cy1), max(px2, cx2), max(py2, cy2)]
                prev["confidence"] = round((prev["confidence"] + item["confidence"]) / 2.0, 3)
                prev["_last_y_norm"] = cur_yn
                prev["_last_bbox"] = list(item["bbox"])
                continue

            starts_new_para = (
                item["type"] != "paragraph"
                or prev["type"] != "paragraph"
                or prev.get("_standalone", False)
                or item.get("standalone_line", False)
                or item["is_indented"]
                or bool(new_para_prefix.match(item["text"]))
                or prev["style"].get("textAlign") in ("center", "right")
                or item["style"].get("textAlign") in ("center", "right")
                or not prev_reached_right
                or v_gap > median_body_h * 0.92
            )
            if not starts_new_para:
                prev["text"] = f"{prev['text']} {item['text']}"
                prev["bbox"] = [min(px1, cx1), min(py1, cy1), max(px2, cx2), max(py2, cy2)]
                prev["confidence"] = round((prev["confidence"] + item["confidence"]) / 2.0, 3)
                prev["style"]["textAlign"] = "justify"
                prev["_last_y_norm"] = cur_yn
                prev["_last_bbox"] = list(item["bbox"])
                prev.setdefault("_b_votes", []).append(1 if item["style"].get("bold") else 0)
                prev.setdefault("_i_votes", []).append(1 if item["style"].get("italic") else 0)
                prev.setdefault("_sizes", []).append(float(item["style"].get("fontSizePt", 13.0)))
            else:
                blk = {
                    "id": f"b-{len(blocks) + 1}-{uuid.uuid4().hex[:6]}",
                    "type": item["type"],
                    "order": len(blocks) + 1,
                    "bbox": list(item["bbox"]),
                    "text": item["text"],
                    "confidence": round(item["confidence"], 3),
                    "style": dict(item["style"]),
                    "_standalone": item.get("standalone_line", False),
                    "_last_y_norm": cur_yn,
                    "_last_bbox": list(item["bbox"]),
                    "_b_votes": [1 if item["style"].get("bold") else 0],
                    "_i_votes": [1 if item["style"].get("italic") else 0],
                    "_sizes": [float(item["style"].get("fontSizePt", 13.0))],
                }
                if "tableData" in item:
                    blk["tableData"] = item["tableData"]
                blocks.append(blk)

        for blk in blocks:
            b_votes = blk.pop("_b_votes", [])
            i_votes = blk.pop("_i_votes", [])
            sizes_blk = blk.pop("_sizes", [])
            blk.pop("_last_y_norm", None)
            blk.pop("_last_bbox", None)
            if len(b_votes) >= 2 and blk.get("type") == "paragraph":
                if sum(b_votes) / len(b_votes) >= 0.55:
                    blk["style"]["bold"] = True
                else:
                    blk["style"].pop("bold", None)
                if sum(i_votes) / len(i_votes) >= 0.55:
                    blk["style"]["italic"] = True
                else:
                    blk["style"].pop("italic", None)
            if len(sizes_blk) >= 2 and blk.get("type") == "paragraph":
                med_pt = round(float(np.median(sizes_blk)) * 2.0) / 2.0
                blk["style"]["fontSizePt"] = med_pt
            blk.pop("_standalone", None)

        # Đồng nhất fontFamily chủ đạo trên toàn trang cho các khối văn bản (paragraph)
        page_fonts = [
            b.get("style", {}).get("fontFamily")
            for b in blocks
            if b.get("style", {}).get("fontFamily")
        ]
        if page_fonts:
            from collections import Counter
            dom_font, dom_cnt = Counter(page_fonts).most_common(1)[0]
            if dom_cnt / float(len(page_fonts)) >= 0.55:
                for b in blocks:
                    if b.get("type") == "paragraph" and "style" in b:
                        b["style"]["fontFamily"] = dom_font

        if not blocks:
            # Nếu không tìm thấy chữ nào
            blocks.append({
                "id": f"b-1-{uuid.uuid4().hex[:6]}",
                "type": "paragraph",
                "order": 1,
                "bbox": [10, 10, width - 10, 50],
                "text": "[Trang không có nội dung chữ nhận dạng được]",
                "confidence": 1.0,
            })

        return {
            "schemaVersion": 1,
            "pageNumber": page_number,
            "width": width,
            "height": height,
            "pageMargins": page_margins,
            "blocks": blocks,
        }

    def _run_fallback_ocr(self, img: np.ndarray, page_number: int, width: int, height: int) -> dict:
        """
        Engine dự phòng thực tế (Tesseract OCR tiếng Việt) khi PaddleOCR gặp sự cố.
        Tuyệt đối không sinh văn bản mẫu giả lập.
        """
        blocks = []
        try:
            import pytesseract
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
            data = pytesseract.image_to_data(rgb, lang="vie", output_type=pytesseract.Output.DICT)
            lines: dict[tuple[int, int, int], list[tuple[int, int, int, int, str, float]]] = {}
            n_boxes = len(data.get("text", []))
            for i in range(n_boxes):
                txt = (data["text"][i] or "").strip()
                conf = float(data["conf"][i]) if str(data["conf"][i]).replace(".", "", 1).lstrip("-").isdigit() else -1.0
                if not txt or conf < 20:
                    continue
                key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
                x, y, bw, bh = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
                lines.setdefault(key, []).append((x, y, x + bw, y + bh, txt, conf / 100.0))

            for order, items in enumerate(lines.values(), start=1):
                x1 = max(0, min(it[0] for it in items))
                y1 = max(0, min(it[1] for it in items))
                x2 = min(width, max(it[2] for it in items))
                y2 = min(height, max(it[3] for it in items))
                line_text = normalize_vietnamese_text(" ".join(it[4] for it in items)).strip()
                if not line_text:
                    continue
                avg_conf = float(np.mean([it[5] for it in items]))
                blocks.append({
                    "id": f"b-{order}-{uuid.uuid4().hex[:6]}",
                    "type": "paragraph",
                    "order": len(blocks) + 1,
                    "bbox": [x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)],
                    "text": line_text,
                    "confidence": round(avg_conf, 3),
                })
        except Exception as exc:
            LOG.warning("Local Tesseract không khả dụng (%s), thử chạy qua Docker Tesseract engine...", exc)
            try:
                import subprocess
                _, enc = cv2.imencode(".png", img)
                tsv_out = None
                for cname in ["docforge-tesseract-engine", "pbl4-doc_forge-backend-ai-worker-1"]:
                    res = subprocess.run(
                        ["docker", "exec", "-i", cname, "tesseract", "stdin", "stdout", "-l", "vie", "tsv"],
                        input=enc.tobytes(),
                        capture_output=True,
                        timeout=30,
                    )
                    if res.returncode == 0 and res.stdout:
                        tsv_out = res.stdout.decode("utf-8", errors="ignore")
                        break
                if tsv_out:
                    tsv_lines: dict[tuple[int, int, int], list[tuple[int, int, int, int, str, float]]] = {}
                    rows = tsv_out.strip().splitlines()
                    if len(rows) > 1:
                        header = rows[0].split("\t")
                        for r in rows[1:]:
                            cols = r.split("\t")
                            if len(cols) >= 12:
                                txt = cols[11].strip()
                                try:
                                    conf = float(cols[10])
                                except ValueError:
                                    conf = -1.0
                                if not txt or conf < 20:
                                    continue
                                b_num, p_num, l_num = int(cols[2]), int(cols[3]), int(cols[4])
                                x, y, bw, bh = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
                                tsv_lines.setdefault((b_num, p_num, l_num), []).append((x, y, x + bw, y + bh, txt, conf / 100.0))

                    for order, items in enumerate(tsv_lines.values(), start=1):
                        x1 = max(0, min(it[0] for it in items))
                        y1 = max(0, min(it[1] for it in items))
                        x2 = min(width, max(it[2] for it in items))
                        y2 = min(height, max(it[3] for it in items))
                        line_text = normalize_vietnamese_text(" ".join(it[4] for it in items)).strip()
                        if not line_text:
                            continue
                        avg_conf = float(np.mean([it[5] for it in items]))
                        blocks.append({
                            "id": f"b-{order}-{uuid.uuid4().hex[:6]}",
                            "type": "paragraph",
                            "order": len(blocks) + 1,
                            "bbox": [x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)],
                            "text": line_text,
                            "confidence": round(avg_conf, 3),
                        })
            except Exception as dock_exc:
                LOG.error("Fallback Docker Tesseract cũng gặp lỗi: %s", dock_exc)

        if not blocks:
            blocks.append({
                "id": f"b-1-{uuid.uuid4().hex[:6]}",
                "type": "paragraph",
                "order": 1,
                "bbox": [10, 10, max(11, width - 10), min(height, 50)],
                "text": "[Trang không có nội dung chữ nhận dạng được]",
                "confidence": 0.0,
            })

        return {
            "schemaVersion": 1,
            "pageNumber": page_number,
            "width": width,
            "height": height,
            "blocks": blocks,
        }
