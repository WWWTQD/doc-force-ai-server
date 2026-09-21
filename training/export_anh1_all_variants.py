import os
import sys
import json

# PRE-IMPORT TORCH AND APPLY PADDLE CPU STATIC RUNNER PATCH FOR WINDOWS
try:
    import torch
    import paddle
    import paddle.inference as paddle_infer
    if hasattr(paddle_infer, "Config") and hasattr(paddle_infer.Config, "enable_mkldnn"):
        paddle_infer.Config.enable_mkldnn = lambda self, *args, **kwargs: None

    import paddlex.inference.models.runners.paddle_static.runner as paddle_runner
    if hasattr(paddle_runner, "resolve_paddle_static_engine_config"):
        orig_resolve = paddle_runner.resolve_paddle_static_engine_config

        def safe_resolve(model_name, config):
            cfg = orig_resolve(model_name, config)
            if cfg.get("device_type") == "cpu":
                cfg["run_mode"] = "paddle"
            return cfg

        paddle_runner.resolve_paddle_static_engine_config = safe_resolve
except Exception as e:
    print(f"Patch warning: {e}")

from paddleocr import PaddleOCR


def process_image(ocr, img_path):
    if not os.path.exists(img_path):
        return {"error": "file_not_found", "count": 0, "lines": []}

    try:
        results = list(ocr.predict(img_path))
        extracted = []
        for r in results:
            raw_dict = r.json if hasattr(r, "json") and r.json else {}
            res_inner = raw_dict.get("res", {})
            
            # Check rec_texts or rec_text (plural/singular fallback)
            rec_texts = res_inner.get("rec_texts", res_inner.get("rec_text", []))
            rec_scores = res_inner.get("rec_scores", res_inner.get("rec_score", []))
            
            for line_idx, (t, s) in enumerate(zip(rec_texts, rec_scores), 1):
                extracted.append({
                    "line_no": line_idx,
                    "text": t,
                    "confidence": round(float(s), 4)
                })

        return {
            "image": img_path,
            "count": len(extracted),
            "lines": extracted
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return {"error": str(e), "count": 0, "lines": []}


def main():
    # Set sys.stdout encoding to utf-8 for Windows console support
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("RUNNING PP-STRUCTUREV3 / PP-OCR ON ALL ANH1 RESTORED VARIANTS")
    print("=" * 70)

    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang="vi")

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)

    master_summary = {}

    for var in variants:
        restored_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))
        restored_data = process_image(ocr, restored_path)

        master_summary[var] = restored_data

        print(f"\nVariant: {var} ({restored_data['count']} text lines extracted)")
        print("-" * 50)
        for line_item in restored_data.get("lines", []):
            print(f"  Line {line_item['line_no']:2d} [{line_item['confidence']:.2f}]: {line_item['text']}")

    # Save summary JSON
    out_file = os.path.join(out_dir, "anh1_all_restored_ocr_output.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"ALL ANH1 VARIANTS PROCESSED SUCCESSFULLY!")
    print(f"Master JSON output saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
