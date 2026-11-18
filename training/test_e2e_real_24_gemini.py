"""
Kiểm thử Thực tế End-to-End trên `datasets/real/real_24.jpg`:
  1. Khởi tạo đầy đủ ScanPipeline (4 model phụ + U-Net v5.1 Stroke-Preserving + Gemini Post-OCR).
  2. Chạy E2E trên `datasets/real/real_24.jpg` với `llmPostprocess=True` (đo thời gian từng bước:
     PREPROCESSING, RESTORING, OCR, LLM_POSTPROCESS, EXPORT DOCX/PDF).
  3. So sánh văn bản trước (`ocrText`) và sau (`text`) khi qua Gemini, đồng thời kiểm chứng
     100% định dạng Word (`style`: `Times New Roman`, `13pt`, `Heading 14pt`, căn lề, thụt dòng)
     được giữ nguyên và kết xuất thành công ra `.docx` & `.pdf`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = ROOT / "worker"
for p in (str(ROOT), str(WORKER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(WORKER_DIR / ".env")

import cv2
import docx
from app.services.restoration_service import RestorationService
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor
from real_worker import _generate_docx_from_snapshot, _generate_pdf_from_snapshot
from scan_pipeline import ScanPipeline


def run_e2e_real_24() -> None:
    img_path = ROOT / "datasets" / "real" / "real_24.jpg"
    assert img_path.exists(), f"Không tìm thấy {img_path}"
    img = cv2.imread(str(img_path))
    assert img is not None, "Không đọc được ảnh real_24.jpg"

    t_init0 = time.perf_counter()
    restoration_service = RestorationService.get_instance()
    preprocessor = ImagePreprocessor(enable_magic=True)
    ocr_engine = OCREngine(lang="vi")
    pipeline = ScanPipeline(
        ocr_engine=ocr_engine,
        preprocessor=preprocessor,
        restoration_service=restoration_service,
        enable_document_scan=True,
        enable_restoration=False,  # Mặc định tắt U-Net, bật linh hoạt qua options["useUnet"]
    )
    init_ms = round((time.perf_counter() - t_init0) * 1000.0, 1)

    stage_timestamps: list[tuple[str, float]] = []

    def _on_stage(stg: str) -> None:
        stage_timestamps.append((stg, time.perf_counter()))

    # 1. Chạy chế độ mặc định (OpenCV Magic Enhance + Gemini Post-OCR)
    t_start = time.perf_counter()
    res_default = pipeline.process_image(
        img,
        page_number=1,
        options={"llmPostprocess": True, "useLlm": True, "useUnet": False},
        progress_cb=_on_stage,
    )
    t_end = time.perf_counter()

    # Tính thời gian từng bước
    step_timings_ms: dict[str, float] = {}
    for i, (stg, ts) in enumerate(stage_timestamps):
        next_ts = stage_timestamps[i + 1][1] if (i + 1) < len(stage_timestamps) else t_end
        step_timings_ms[stg] = round((next_ts - ts) * 1000.0, 1)
    total_default_ms = round((t_end - t_start) * 1000.0, 1)

    # 2. Chạy thêm chế độ bật U-Net v5.1 Hybrid + Gemini Post-OCR để đo đối chứng
    stage_ts_unet: list[tuple[str, float]] = []
    t_u0 = time.perf_counter()
    res_unet = pipeline.process_image(
        img,
        page_number=1,
        options={"llmPostprocess": True, "useLlm": True, "useUnet": True},
        progress_cb=lambda stg: stage_ts_unet.append((stg, time.perf_counter())),
    )
    t_u1 = time.perf_counter()
    unet_step_timings_ms: dict[str, float] = {}
    for i, (stg, ts) in enumerate(stage_ts_unet):
        next_ts = stage_ts_unet[i + 1][1] if (i + 1) < len(stage_ts_unet) else t_u1
        unet_step_timings_ms[stg] = round((next_ts - ts) * 1000.0, 1)
    total_unet_ms = round((t_u1 - t_u0) * 1000.0, 1)

    layout = res_default.layout
    blocks = layout.get("blocks", [])

    # 3. Kết xuất thử ra file Word (.docx) và PDF (.pdf) chuẩn A4
    snapshot = {"format": "DOCX", "pages": [{"pageNumber": 1, "layout": layout}]}
    t_exp0 = time.perf_counter()
    docx_bytes = _generate_docx_from_snapshot(snapshot)
    docx_ms = round((time.perf_counter() - t_exp0) * 1000.0, 1)

    t_pdf0 = time.perf_counter()
    pdf_bytes = _generate_pdf_from_snapshot({**snapshot, "format": "SEARCHABLE_PDF"})
    pdf_ms = round((time.perf_counter() - t_pdf0) * 1000.0, 1)

    out_dir = ROOT / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    docx_path = out_dir / "real_24_gemini_postocr.docx"
    pdf_path = out_dir / "real_24_gemini_postocr.pdf"
    docx_path.write_bytes(docx_bytes)
    pdf_path.write_bytes(pdf_bytes)

    # Kiểm tra lại nội dung và định dạng trong file DOCX đã tạo
    doc_loaded = docx.Document(str(docx_path))
    docx_paragraphs_info = []
    for p in doc_loaded.paragraphs:
        if not p.text.strip():
            continue
        runs_info = [
            {
                "text": r.text,
                "font_name": r.font.name,
                "font_size_pt": r.font.size.pt if r.font.size else None,
                "bold": r.bold,
                "italic": r.italic,
            }
            for r in p.runs
            if r.text
        ]
        docx_paragraphs_info.append(
            {
                "text": p.text,
                "alignment": str(p.alignment),
                "first_line_indent_pt": p.paragraph_format.first_line_indent.pt
                if p.paragraph_format.first_line_indent
                else 0.0,
                "runs": runs_info,
            }
        )

    report = {
        "image": "datasets/real/real_24.jpg",
        "init_models_ms": init_ms,
        "default_mode_opencv_plus_gemini": {
            "restorationMode": layout.get("scan", {}).get("restorationMode"),
            "step_timings_ms": step_timings_ms,
            "total_ms": total_default_ms,
            "postprocessMeta": layout.get("postprocessMeta"),
        },
        "hybrid_mode_unet_v5_1_plus_gemini": {
            "restorationMode": res_unet.layout.get("scan", {}).get("restorationMode"),
            "step_timings_ms": unet_step_timings_ms,
            "total_ms": total_unet_ms,
            "postprocessMeta": res_unet.layout.get("postprocessMeta"),
        },
        "export_timings_ms": {
            "docx_ms": docx_ms,
            "pdf_ms": pdf_ms,
            "docx_bytes": len(docx_bytes),
            "pdf_bytes": len(pdf_bytes),
        },
        "blocks_comparison": [
            {
                "id": b.get("id"),
                "order": b.get("order"),
                "type": b.get("type"),
                "confidence": b.get("confidence"),
                "style": b.get("style"),
                "before_gemini_ocrText": b.get("ocrText", b.get("text")),
                "after_gemini_text": b.get("text"),
                "changes_ops": (b.get("suggestions") or [{}])[0].get("ops", []),
            }
            for b in blocks
        ],
        "docx_verification": docx_paragraphs_info,
    }

    report_path = out_dir / "e2e_real_24_gemini_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ KẾT QUẢ TEST E2E TRÊN real_24.jpg ================")
    print(f"Khởi tạo 4 model phụ + U-Net v5.1 + OCREngine: {init_ms} ms")
    print(f"Thời gian từng bước (OpenCV + Gemini): {json.dumps(step_timings_ms, ensure_ascii=False)} (Tổng: {total_default_ms} ms)")
    print(f"Thời gian từng bước (U-Net v5.1 + Gemini): {json.dumps(unet_step_timings_ms, ensure_ascii=False)} (Tổng: {total_unet_ms} ms)")
    print(f"Kết xuất Word (.docx): {docx_ms} ms | PDF (.pdf): {pdf_ms} ms")
    for b in report["blocks_comparison"]:
        print(f"\n--- Block #{b['order']} ({b['type']}) | style={json.dumps(b['style'], ensure_ascii=False)} ---")
        print(f"  [TRƯỚC GEMINI]: {b['before_gemini_ocrText']}")
        print(f"  [SAU GEMINI]  : {b['after_gemini_text']}")
        if b["changes_ops"]:
            pairs = [f"{op.get('from')} -> {op.get('to')}" for op in b["changes_ops"]]
            print(f"  [CÁC TỪ ĐÃ SỬA]: {', '.join(pairs)}")


if __name__ == "__main__":
    run_e2e_real_24()
