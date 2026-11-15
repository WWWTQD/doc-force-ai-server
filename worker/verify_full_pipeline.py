"""
Kiểm chứng toàn diện (End-to-End Verification) cho Hệ thống AI Scan & Export (DOCX + PDF).
"""
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

os.environ["ENABLE_IMAGE_RESTORATION"] = "false"

from scan_pipeline import ScanPipeline
from real_worker import _generate_docx_from_snapshot, _generate_pdf_from_snapshot


def main():
    t0 = time.time()
    pipe = ScanPipeline(enable_document_scan=True, enable_restoration=False)
    img_path = os.path.join(ROOT_DIR, "real", "doc_2026-10-22T15-59-33-053Z.jpg")
    with open(img_path, "rb") as f:
        res = pipe.process_bytes(f.read(), page_number=1)

    snapshot = {
        "schemaVersion": 1,
        "documentId": "2b7ae878-19fd-452d-bad9-f1867a13bbc4",
        "pages": [{"pageNumber": 1, "layout": res.layout}],
    }
    docx_bytes = _generate_docx_from_snapshot(snapshot)
    pdf_bytes = _generate_pdf_from_snapshot(snapshot)

    out_dir = os.path.join(ROOT_DIR, "data", "output")
    os.makedirs(out_dir, exist_ok=True)
    docx_out = os.path.join(out_dir, "DocForge_Scan_HoanChinh.docx")
    pdf_out = os.path.join(out_dir, "DocForge_Scan_HoanChinh.pdf")
    with open(docx_out, "wb") as f:
        f.write(docx_bytes)
    with open(pdf_out, "wb") as f:
        f.write(pdf_bytes)

    blocks = res.layout.get("blocks", [])
    print("==================================================================")
    print(f"HOÀN TẤT TRONG {time.time() - t0:.2f}s")
    print(f"  - Phương pháp Scan: {res.geometry.method} (conf={res.geometry.detection_confidence:.2f})")
    print(f"  - Tự động xoay:     {res.geometry.rotation}° -> Khổ ảnh sạch: {res.restored_image.shape[1]}x{res.restored_image.shape[0]}")
    print(f"  - Tổng số khối chữ: {len(blocks)} khối")
    print(f"  - File Word (.docx): {docx_out} ({len(docx_bytes):,} bytes)")
    print(f"  - File PDF  (.pdf):  {pdf_out} ({len(pdf_bytes):,} bytes)")
    print("==================================================================")
    for b in blocks[:6]:
        print(f"  [{b['order']:02d}] ({b['type']}) style={b.get('style')} -> {b['text']}")


if __name__ == "__main__":
    main()
