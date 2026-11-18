"""
3-Panel Visual Comparison & OCR Benchmark for U-Net v5.1 (Stroke-Preserving Hybrid).

Compares on `real_11.jpg`, `real_24.jpg`, and `real_36.jpg`:
  - Column 1: Original image
  - Column 2: Mode A (OpenCV `magic_enhance`)
  - Column 3: Mode B (U-Net v5.1 Stroke-Preserving Hybrid)

Saves:
  - `eval_results/bench_real_11_compare_3panel.jpg`
  - `eval_results/bench_real_24_compare_3panel.jpg`
  - `eval_results/bench_real_36_compare_3panel.jpg`
  - `eval_results/bench_unet_v5_1_ocr_report.json`
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKER_DIR = os.path.join(PROJECT_ROOT, "worker")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

from app.services.restoration_service import RestorationService
from worker.ocr_engine import OCREngine
from worker.preprocessor import ImagePreprocessor

TARGET_FILES = ("real_11.jpg", "real_24.jpg", "real_36.jpg")


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        os.path.join(PROJECT_ROOT, "models", "fonts", "timesbd.ttf"),
        os.path.join(r"C:\Windows\Fonts", "arialbd.ttf"),
        os.path.join(r"C:\Windows\Fonts", "timesbd.ttf"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _render_3panel(
    orig_bgr: np.ndarray,
    mode_a_bgr: np.ndarray,
    mode_b_bgr: np.ndarray,
    titles: List[str],
    subtitles: List[str],
    display_height: int = 1050,
) -> np.ndarray:
    colors = [(175, 30, 30), (20, 110, 175), (15, 135, 55)]
    title_font = _load_font(24)
    sub_font = _load_font(18)
    header_h = 72

    panels: List[np.ndarray] = []
    for img_bgr, title, sub, col_rgb in zip([orig_bgr, mode_a_bgr, mode_b_bgr], titles, subtitles, colors):
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        h, w = img_bgr.shape[:2]
        scale = float(display_height) / float(max(1, h))
        nw = max(360, int(round(w * scale)))
        resized = cv2.resize(img_bgr, (nw, display_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

        header_pil = Image.new("RGB", (nw, header_h), (246, 248, 250))
        draw = ImageDraw.Draw(header_pil)
        draw.text((14, 10), title, fill=col_rgb, font=title_font)
        draw.text((14, 42), sub, fill=(55, 65, 81), font=sub_font)
        draw.line([(0, header_h - 1), (nw, header_h - 1)], fill=(205, 210, 218), width=2)

        header_bgr = cv2.cvtColor(np.array(header_pil), cv2.COLOR_RGB2BGR)
        panels.append(np.vstack([header_bgr, resized]))

    divider = np.full((panels[0].shape[0], 6, 3), 195, dtype=np.uint8)
    return np.hstack([panels[0], divider, panels[1], divider, panels[2]])


def _eval_ocr(ocr: OCREngine, img_bgr: np.ndarray) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    raw_lines = ocr._detect_and_recognize_vietocr(img_bgr, w) if ocr.viet_recognizer else []
    line_confs = [float(item[1][1]) for item in raw_lines]
    mid_lines = []
    for poly, (txt, conf) in raw_lines:
        ys = [pt[1] for pt in poly]
        yc = float(np.mean(ys))
        if 0.30 * h <= yc <= 0.75 * h:
            mid_lines.append({"text": txt, "conf": float(conf), "y_pct": round(yc / h * 100.0, 1)})

    layout = ocr.process_image(img_bgr, page_number=1)
    blocks = [b for b in layout.get("blocks", []) if not b.get("text", "").startswith("[Trang không")]
    block_confs = [float(b.get("confidence", 0.0)) for b in blocks]
    total_chars = sum(len(b.get("text", "")) for b in blocks)

    return {
        "detected_lines": len(raw_lines),
        "line_avg_conf": round(float(np.mean(line_confs)) if line_confs else 0.0, 4),
        "middle_band_lines": len(mid_lines),
        "middle_band_samples": mid_lines[:5],
        "blocks": len(blocks),
        "block_avg_conf": round(float(np.mean(block_confs)) if block_confs else 0.0, 4),
        "total_chars": total_chars,
    }


def run_benchmark() -> dict:
    eval_dir = os.path.join(PROJECT_ROOT, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    prep = ImagePreprocessor()
    restoration = RestorationService.get_instance()
    restoration.initialize(
        model_path=os.path.join(PROJECT_ROOT, "models", "restoration_best.pth"),
        device_name="cpu",
    )
    # Warm up PyTorch CPU thread pool once so per-image timing reflects steady-state worker latency
    dummy = np.full((256, 256, 3), 240, dtype=np.uint8)
    _ = restoration.restore_image_bgr(dummy, max_side=640, preserve_color=True, force=True)

    ocr = OCREngine(lang="vi", use_gpu=False)

    report: Dict[str, Any] = {"images": {}}

    for fname in TARGET_FILES:
        img_path = os.path.join(PROJECT_ROOT, "datasets", "real", fname)
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        stem = os.path.splitext(fname)[0]
        h, w = orig_bgr.shape[:2]

        # Mode A: OpenCV clean + magic_enhance
        t_a0 = time.perf_counter()
        pre_clean = prep.clean_handwriting_and_book_gutter(orig_bgr)
        mode_a_bgr = prep.magic_enhance(pre_clean)
        mode_a_ms = (time.perf_counter() - t_a0) * 1000.0

        # Mode B: U-Net v5.1 Stroke-Preserving Hybrid
        t_b0 = time.perf_counter()
        mode_b_bgr = restoration.restore_image_bgr(
            orig_bgr,
            preserve_color=True,
            max_side=640,
            force=True,
        )
        mode_b_ms = (time.perf_counter() - t_b0) * 1000.0

        ocr_orig = _eval_ocr(ocr, orig_bgr)
        ocr_a = _eval_ocr(ocr, mode_a_bgr)
        ocr_b = _eval_ocr(ocr, mode_b_bgr)

        titles = [
            "1. Ảnh gốc (Original)",
            "2. Mode A (OpenCV magic_enhance)",
            "3. Mode B (U-Net v5.1 Stroke-Preserving)",
        ]
        subtitles = [
            f"{w}x{h} | Lines: {ocr_orig['detected_lines']} | avg_conf: {ocr_orig['line_avg_conf']*100:.2f}%",
            f"Time: {mode_a_ms:.0f} ms | Lines: {ocr_a['detected_lines']} | avg_conf: {ocr_a['line_avg_conf']*100:.2f}%",
            f"Time: {mode_b_ms:.0f} ms | Lines: {ocr_b['detected_lines']} | avg_conf: {ocr_b['line_avg_conf']*100:.2f}%",
        ]

        panel_img = _render_3panel(orig_bgr, mode_a_bgr, mode_b_bgr, titles, subtitles, display_height=1050)
        out_panel = os.path.join(eval_dir, f"bench_{stem}_compare_3panel.jpg")
        cv2.imwrite(out_panel, panel_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        report["images"][fname] = {
            "resolution": f"{w}x{h}",
            "panel_path": out_panel,
            "mode_a_ms": round(mode_a_ms, 1),
            "unet_v5_1_ms": round(mode_b_ms, 1),
            "orig": ocr_orig,
            "mode_a": ocr_a,
            "unet_v5_1": ocr_b,
        }
        print(
            f"[{fname}] U-Net v5.1={mode_b_ms:.1f}ms (Mode A={mode_a_ms:.1f}ms) | "
            f"Lines: Orig={ocr_orig['detected_lines']}, ModeA={ocr_a['detected_lines']}, UNet5.1={ocr_b['detected_lines']} | "
            f"line_avg_conf: Orig={ocr_orig['line_avg_conf']:.4f}, ModeA={ocr_a['line_avg_conf']:.4f}, UNet5.1={ocr_b['line_avg_conf']:.4f} | "
            f"block_avg_conf: Orig={ocr_orig['block_avg_conf']:.4f}, ModeA={ocr_a['block_avg_conf']:.4f}, UNet5.1={ocr_b['block_avg_conf']:.4f}",
            flush=True,
        )

    out_json = os.path.join(eval_dir, "bench_unet_v5_1_ocr_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved report -> {out_json}", flush=True)
    return report


if __name__ == "__main__":
    run_benchmark()
