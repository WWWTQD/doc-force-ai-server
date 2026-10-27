import os
import sys
import time
import cv2
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.restoration_service import RestorationService
from worker.ocr_engine import OCREngine


def run_experiment(image_path: str):
    print(f"=== ĐANG CHẠY THỬ NGHIỆM TRÊN FILE: {image_path} ===")
    if not os.path.exists(image_path):
        print(f"File {image_path} không tồn tại!")
        return

    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        print("Không đọc được ảnh!")
        return

    h, w = orig_bgr.shape[:2]
    print(f"Kích thước ảnh gốc: {w}x{h}")

    # Khởi tạo OCR Engine và Restoration Service
    ocr_engine = OCREngine(lang="vi", use_gpu=False)
    restoration_service = RestorationService.get_instance()
    restoration_service.initialize()

    # ─────────────────────────────────────────────────────────────
    # Cách A: Không có U-Net (Ảnh gốc đưa trực tiếp vào PaddleOCR)
    # ─────────────────────────────────────────────────────────────
    print("\n--- [Cách A] Đang nhận dạng OCR trên ảnh gốc (Không qua U-Net)... ---")
    t0 = time.perf_counter()
    res_a = ocr_engine.process_image(orig_bgr, page_number=1)
    time_a = (time.perf_counter() - t0) * 1000.0

    blocks_a = res_a.get("blocks", [])
    valid_blocks_a = [b for b in blocks_a if not b["text"].startswith("[Trang không")]
    conf_a = np.mean([b["confidence"] for b in valid_blocks_a]) if valid_blocks_a else 0.0

    # ─────────────────────────────────────────────────────────────
    # Cách B: Có U-Net (Ảnh qua RestorationService rồi đưa vào PaddleOCR)
    # ─────────────────────────────────────────────────────────────
    print("\n--- [Cách B] Đang khôi phục ảnh bằng U-Net v4 và nhận dạng OCR... ---")
    t_unet_0 = time.perf_counter()
    restored_bgr = restoration_service.restore_image_bgr(
        orig_bgr,
        tile_size=512,
        overlap=64,
        preserve_color=True,
    )
    time_unet = (time.perf_counter() - t_unet_0) * 1000.0

    # Lưu ảnh phục hồi để kiểm tra trực quan
    os.makedirs("eval_results", exist_ok=True)
    out_restored_path = os.path.join("eval_results", "experiment_restored.jpg")
    cv2.imwrite(out_restored_path, restored_bgr)
    print(f"  ✓ Đã lưu ảnh khôi phục tại: {out_restored_path}")

    t_ocr_b = time.perf_counter()
    res_b = ocr_engine.process_image(restored_bgr, page_number=1)
    time_ocr_b = (time.perf_counter() - t_ocr_b) * 1000.0

    blocks_b = res_b.get("blocks", [])
    valid_blocks_b = [b for b in blocks_b if not b["text"].startswith("[Trang không")]
    conf_b = np.mean([b["confidence"] for b in valid_blocks_b]) if valid_blocks_b else 0.0

    # ─────────────────────────────────────────────────────────────
    # In bảng so sánh
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(" BẢNG SO SÁNH KẾT QUẢ ĐỌC DÒNG CHỮ (OCR) TRƯỚC VÀ SAU KHI CÓ U-NET")
    print("=" * 75)
    print(f"{'Tiêu chí':<32} | {'Cách A (Ảnh gốc)':<18} | {'Cách B (Có U-Net)':<18}")
    print("-" * 75)
    print(f"{'Số dòng chữ (Bounding Boxes)':<32} | {len(valid_blocks_a):<18} | {len(valid_blocks_b):<18}")
    print(f"{'Độ tự tin OCR trung bình':<32} | {conf_a * 100:.2f}%{'':<11} | {conf_b * 100:.2f}%{'':<11}")
    print(f"{'Thời gian suy luận OCR':<32} | {time_a:.1f} ms{'':<10} | {time_ocr_b:.1f} ms{'':<10}")
    print(f"{'Thời gian khôi phục U-Net':<32} | {'---':<18} | {time_unet:.1f} ms{'':<10}")
    print("-" * 75)

    print("\n--- MẪU DÒNG TEXT TRÍCH XUẤT CỦA CÁCH A (Ảnh gốc) ---")
    for b in valid_blocks_a[:6]:
        print(f"  [{b['confidence']:.2f}] {b['text']}")

    print("\n--- MẪU DÒNG TEXT TRÍCH XUẤT CỦA CÁCH B (Có U-Net) ---")
    for b in valid_blocks_b[:6]:
        print(f"  [{b['confidence']:.2f}] {b['text']}")
    print("=" * 75)


if __name__ == "__main__":
    test_path = "datasets/real/real_11.jpg"
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    run_experiment(test_path)
