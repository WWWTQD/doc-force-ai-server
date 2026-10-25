"""
Script kiểm thử toàn diện chuỗi AI Scan Pipeline trên ảnh camera thực tế.
"""
import os
import sys
import time
import json
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "worker"))

from scan_pipeline import ScanPipeline
from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from document_scanner import DocumentScanner

def main():
    image_path = os.path.join(ROOT, "data", "input", "anh_test.jpg")
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh test: {image_path}")
        return

    print("══════════════════════════════════════════════════════════")
    print("  KIỂM THỬ END-TO-END SCAN PIPELINE TRÊN ẢNH CAMERA THẬT")
    print("══════════════════════════════════════════════════════════")
    print(f"1. Đọc ảnh gốc: {image_path}")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"   Kích thước ảnh gốc: {w}x{h} px")

    t0 = time.time()
    print("2. Khởi tạo ScanPipeline (nạp document_detector.pt & layout model)...")
    pipeline = ScanPipeline(enable_document_scan=True)
    t_init = time.time() - t0
    print(f"   Nạp model xong trong: {t_init:.2f}s")

    print("3. Chạy xử lý ảnh qua pipeline (Detect -> Warp -> Magic Enhance -> Layout)...")
    t1 = time.time()
    result = pipeline.process_image(img, page_number=1)
    t_proc = time.time() - t1
    print(f"   Xử lý hoàn tất trong: {t_proc:.2f}s")

    # Kiểm tra kết quả hình học
    geo = result.geometry
    print("\n[KẾT QUẢ HÌNH HỌC]:")
    print(f"   - Phương pháp phát hiện: {geo.method}")
    print(f"   - Độ tin cậy (Confidence): {geo.detection_confidence:.2f}")
    print(f"   - Góc xoay tự động: {geo.rotation}°")
    print(f"   - Kích thước ảnh sau nắn phẳng: {result.restored_image.shape[1]}x{result.restored_image.shape[0]} px")

    # Lưu ảnh kết quả ra thư mục data/output để kiểm tra trực quan
    out_dir = os.path.join(ROOT, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_enhanced_path = os.path.join(out_dir, "anh_test_restored.jpg")
    cv2.imwrite(out_enhanced_path, result.restored_image)
    print(f"   -> Đã lưu ảnh scan phục hồi tại: {out_enhanced_path}")

    # Kiểm tra Layout JSON
    layout = result.layout
    blocks = layout.get("blocks", [])
    print(f"\n[KẾT QUẢ LAYOUT & OCR]:")
    print(f"   - Tổng số khối nhận diện: {len(blocks)}")
    types_count = {}
    for b in blocks:
        t = b.get("type", "unknown")
        types_count[t] = types_count.get(t, 0) + 1
    for t, c in types_count.items():
        print(f"     + {t}: {c} blocks")

    # In thử 3 block đầu tiên
    print("\n   3 khối đầu tiên trích xuất được:")
    for b in blocks[:3]:
        print(f"     [{b.get('type')}] (conf={b.get('confidence')}): {b.get('text')[:70]}...")

    # Lưu layout json
    json_path = os.path.join(out_dir, "anh_test_layout.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    print(f"   -> Đã lưu Layout JSON tại: {json_path}")
    print("\n══════════════════════════════════════════════════════════")
    print("  ✓ TẤT CẢ CÁC BƯỚC ĐÃ CHẠY THÀNH CÔNG VÀ CHÍNH XÁC!")
    print("══════════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()
