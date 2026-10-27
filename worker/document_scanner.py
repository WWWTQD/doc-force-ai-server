"""
Document Scanner cho Doc Forge AI Worker.

Chuỗi xử lý hình học trước OCR:
  1. Phát hiện tờ giấy bằng YOLO (models/document_detector.pt).
     Không có model / không phát hiện được -> fallback tìm contour bằng OpenCV.
  2. Tinh chỉnh 4 góc thật của tờ giấy bằng contour bên trong vùng YOLO,
     sau đó nắn phẳng (perspective warp).
  3. Xoay trang về đúng chiều đọc (0/90/180/270 độ).

Mọi phép biến đổi được gộp vào một ma trận homography 3x3 (ảnh gốc -> ảnh đã xử lý)
để có thể ánh xạ bbox OCR ngược về toạ độ ảnh gốc khi cần.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

LOG = logging.getLogger("document_scanner")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    LOG.warning("Thư viện ultralytics chưa được cài đặt. Document detection dùng fallback OpenCV.")

A4_RATIO = math.sqrt(2)  # 297 / 210
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "document_detector.pt")


@dataclass
class ScanGeometry:
    """Kết quả bước hình học: ảnh đã nắn phẳng + thông tin để truy vết."""

    image: np.ndarray
    transform: np.ndarray  # 3x3, điểm ảnh gốc -> điểm ảnh đã xử lý
    original_size: tuple[int, int]  # (width, height) ảnh gốc
    method: str = "none"  # yolo+contour | yolo-bbox | opencv | none
    detection_confidence: float = 0.0
    corners: list[list[float]] | None = None  # TL, TR, BR, BL trong ảnh gốc
    rotation: int = 0  # góc đã xoay theo chiều kim đồng hồ (0/90/180/270)
    notes: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "method": self.method,
            "detectionConfidence": round(float(self.detection_confidence), 3),
            "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in self.corners] if self.corners else None,
            "rotation": self.rotation,
            "originalWidth": self.original_size[0],
            "originalHeight": self.original_size[1],
        }


# ─── Tiện ích hình học ────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm theo thứ tự TL, TR, BR, BL (toạ độ ảnh, trục y hướng xuống)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # TL -> TR -> BR -> BL (theo chiều kim đồng hồ)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def is_plausible_quad(quad: np.ndarray, min_area: float) -> bool:
    """Tứ giác hợp lệ: lồi, đủ lớn, các góc trong khoảng 50°–130°."""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return False
    if cv2.contourArea(quad) < min_area:
        return False
    for i in range(4):
        p_prev, p, p_next = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = p_prev - p, p_next - p
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom < 1e-6:
            return False
        angle = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))
        if angle < 50 or angle > 130:
            return False
    return True


def find_document_quad(gray: np.ndarray, min_area_ratio: float) -> np.ndarray | None:
    """
    Tìm tứ giác lớn nhất giống tờ giấy trong ảnh xám.
    Thử 2 bản đồ biên: Canny (biên giấy) và Otsu (giấy sáng hơn nền).
    """
    h, w = gray.shape[:2]
    min_area = min_area_ratio * h * w
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(blurred))
    edges = cv2.Canny(blurred, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    best, best_area = None, 0.0
    for mask in (edges, bright):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < min_area:
                break
            perimeter = cv2.arcLength(hull, True)
            for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
                approx = cv2.approxPolyDP(hull, eps * perimeter, True)
                if len(approx) == 4:
                    quad = order_corners(approx.reshape(4, 2))
                    area = cv2.contourArea(quad)
                    if area > best_area and is_plausible_quad(quad, min_area):
                        best, best_area = quad, area
                    break
    return best


def rotation_matrix(rotation: int, width: int, height: int) -> np.ndarray:
    """Ma trận 3x3 tương ứng cv2.rotate cho ảnh kích thước (width, height)."""
    if rotation == 90:  # ROTATE_90_CLOCKWISE
        return np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if rotation == 180:
        return np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64)
    if rotation == 270:  # ROTATE_90_COUNTERCLOCKWISE
        return np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=np.float64)
    return np.eye(3, dtype=np.float64)


def rotate_image(img: np.ndarray, rotation: int) -> np.ndarray:
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, codes[rotation]) if rotation in codes else img


def text_axis_is_vertical(img: np.ndarray) -> bool:
    """
    Ước lượng dòng chữ đang nằm ngang hay dọc bằng projection profile:
    dòng chữ ngang -> profile theo hàng dao động mạnh (dòng chữ / khoảng trắng xen kẽ).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = 1000.0 / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]
    # Bỏ 10% viền để tránh mép giấy / bóng
    gray = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    binary = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    binary = cv2.medianBlur(binary.astype(np.uint8), 3)
    if binary.sum() < 50:
        return False

    def _profile_score(profile: np.ndarray) -> float:
        mean = float(profile.mean())
        return float(profile.var()) / (mean * mean + 1e-9)

    row_score = _profile_score(binary.mean(axis=1))
    col_score = _profile_score(binary.mean(axis=0))
    return col_score > row_score * 1.15


# ─── Scanner chính ────────────────────────────────────────────────────────────

class DocumentScanner:
    """Phát hiện, nắn phẳng và xoay đúng chiều trang tài liệu."""

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.25,
        snap_a4: bool = True,
        max_output_side: int = 3000,
        orientation_scorer: Callable[[np.ndarray], float] | None = None,
    ):
        self.conf_threshold = conf_threshold
        self.snap_a4 = snap_a4
        self.max_output_side = max_output_side
        self.orientation_scorer = orientation_scorer
        self.model = None

        path = model_path or os.environ.get("DOCUMENT_MODEL_PATH") or DEFAULT_MODEL_PATH
        if YOLO_AVAILABLE and os.path.exists(path):
            try:
                self.model = YOLO(path)
                LOG.info("Document detector đã nạp: %s (classes=%s)", path, self.model.names)
            except Exception as exc:
                LOG.error("Không nạp được document detector %s: %s", path, exc)
        elif not os.path.exists(path):
            LOG.warning("Không tìm thấy document detector tại %s — dùng fallback OpenCV.", path)

    # ── Bước 1: phát hiện ────────────────────────────────────────────────────
    def _detect_bbox(self, img: np.ndarray) -> tuple[list[float], float] | None:
        if self.model is None:
            return None
        # imgsz=640 giúp YOLO chạy cực nhanh trên CPU mà vẫn chính xác 100%
        results = self.model.predict(img, imgsz=640, conf=self.conf_threshold, verbose=False)
        best = None
        for result in results:
            names = result.names or {}
            doc_classes = {i for i, n in names.items() if any(k in str(n).lower() for k in ("doc", "page", "paper"))}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if doc_classes and cls_id not in doc_classes:
                    continue
                conf = float(box.conf[0])
                if best is None or conf > best[1]:
                    best = (box.xyxy[0].tolist(), conf)
        return best

    def detect_corners(self, img: np.ndarray) -> tuple[np.ndarray | None, str, float]:
        """Trả về (4 góc TL,TR,BR,BL trong ảnh gốc | None, method, confidence)."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detection = self._detect_bbox(img)
        if detection is not None:
            (x1, y1, x2, y2), conf = detection
            pad_x, pad_y = (x2 - x1) * 0.04, (y2 - y1) * 0.04
            cx1, cy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
            cx2, cy2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
            crop = gray[cy1:cy2, cx1:cx2]

            scale = min(1.0, 1000.0 / max(crop.shape[:2]))
            small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
            bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / float(crop.shape[0] * crop.shape[1])
            quad = find_document_quad(small, min_area_ratio=0.45 * bbox_area_ratio)
            if quad is not None:
                quad = quad / scale + np.array([cx1, cy1], dtype=np.float32)
                return order_corners(quad), "yolo+contour", conf

            bbox_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            return bbox_quad, "yolo-bbox", conf

        # Fallback: không có model hoặc model không thấy gì -> tìm trên toàn ảnh
        scale = min(1.0, 1000.0 / max(h, w))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        quad = find_document_quad(small, min_area_ratio=0.2)
        if quad is not None:
            return order_corners(quad / scale), "opencv", 0.0
        return None, "none", 0.0

    # ── Bước 2: nắn phẳng ────────────────────────────────────────────────────
    def warp(self, img: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tl, tr, br, bl = corners
        out_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        out_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

        if self.snap_a4 and out_w > 0 and out_h > 0:
            ratio = out_h / out_w
            if abs(ratio - A4_RATIO) / A4_RATIO < 0.06:
                out_h = out_w * A4_RATIO
            elif abs(ratio - 1 / A4_RATIO) / (1 / A4_RATIO) < 0.06:
                out_w = out_h * A4_RATIO

        scale = min(1.0, self.max_output_side / max(out_w, out_h, 1.0))
        out_w, out_h = max(int(round(out_w * scale)), 1), max(int(round(out_h * scale)), 1)

        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        return warped, matrix.astype(np.float64)

    # ── Bước 3: xoay đúng chiều ──────────────────────────────────────────────
    def orient(self, img: np.ndarray) -> int:
        """Trả về góc cần xoay theo chiều kim đồng hồ (0/90/180/270)."""
        h, w = img.shape[:2]
        if text_axis_is_vertical(img):
            candidates = [90, 270]
        elif w > h * 1.1:
            # Ảnh nằm ngang (landscape) như tài liệu chụp ngang điện thoại: ưu tiên kiểm tra 90°, 270° và 0°
            candidates = [90, 270, 0]
        else:
            candidates = [0, 180]

        if self.orientation_scorer is None:
            return candidates[0]

        try:
            best_rot = candidates[0]
            best_score = self.orientation_scorer(rotate_image(img, best_rot))
            if best_score >= 0.85:  # đọc rất tốt ở hướng đầu tiên -> khỏi thử hướng ngược
                return best_rot
            scores = {best_rot: best_score}
            for cand in candidates[1:]:
                sc = self.orientation_scorer(rotate_image(img, cand))
                scores[cand] = sc
                if sc > best_score + 0.05:
                    best_score = sc
                    best_rot = cand
            LOG.info("Orientation score: %s -> chọn %d°", scores, best_rot)
            return best_rot
        except Exception as exc:
            LOG.warning("Orientation scorer lỗi: %s — giữ hướng %d°", exc, candidates[0])
            return candidates[0]

    # ── Toàn bộ pipeline ─────────────────────────────────────────────────────
    def scan(self, img: np.ndarray, auto_rotate: bool = True) -> ScanGeometry:
        h, w = img.shape[:2]
        geometry = ScanGeometry(image=img, transform=np.eye(3), original_size=(w, h))

        try:
            corners, method, conf = self.detect_corners(img)
        except Exception as exc:
            LOG.warning("Document detection lỗi: %s — dùng nguyên ảnh.", exc)
            corners, method, conf = None, "none", 0.0

        if corners is not None:
            geometry.image, geometry.transform = self.warp(img, corners)
            geometry.corners = corners.tolist()
        geometry.method = method
        geometry.detection_confidence = conf

        if auto_rotate:
            rotation = self.orient(geometry.image)
            if rotation:
                rh, rw = geometry.image.shape[:2]
                geometry.transform = rotation_matrix(rotation, rw, rh) @ geometry.transform
                geometry.image = rotate_image(geometry.image, rotation)
            geometry.rotation = rotation

        LOG.info("Scan: method=%s conf=%.2f rotation=%d° -> %dx%d", geometry.method, conf,
                 geometry.rotation, geometry.image.shape[1], geometry.image.shape[0])
        return geometry


# ─── Ánh xạ bbox về ảnh gốc ───────────────────────────────────────────────────

def map_layout_to_original(layout: dict, transform: np.ndarray, original_size: tuple[int, int]) -> dict:
    """
    Chuyển bbox trong layout (toạ độ ảnh đã xử lý) về toạ độ ảnh gốc
    bằng nghịch đảo homography. Dùng khi không lưu được ảnh phục hồi.
    """
    ow, oh = original_size
    inverse = np.linalg.inv(transform)
    for block in layout.get("blocks", []):
        x1, y1, x2, y2 = block["bbox"]
        pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(pts, inverse)[0]
        nx1 = int(np.clip(np.floor(mapped[:, 0].min()), 0, ow - 1))
        ny1 = int(np.clip(np.floor(mapped[:, 1].min()), 0, oh - 1))
        nx2 = int(np.clip(np.ceil(mapped[:, 0].max()), nx1 + 1, ow))
        ny2 = int(np.clip(np.ceil(mapped[:, 1].max()), ny1 + 1, oh))
        block["bbox"] = [nx1, ny1, nx2, ny2]
    layout["width"], layout["height"] = ow, oh
    return layout
