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


def process_variant(ocr, var_name, base_dir="outputs/restoration/test"):
    img_path = os.path.normpath(os.path.join(base_dir, var_name, "restored.png"))
    if not os.path.exists(img_path):
        return {"error": "file_not_found", "count": 0, "lines": []}

    print(f"\nProcessing {var_name} ({img_path})...", flush=True)
    results = list(ocr.predict(img_path))
    lines = []
    for r in results:
        raw_dict = r.json if hasattr(r, "json") and r.json else {}
        res_inner = raw_dict.get("res", {})
        
        rec_texts = res_inner.get("rec_texts", res_inner.get("rec_text", []))
        rec_scores = res_inner.get("rec_scores", res_inner.get("rec_score", []))
        
        for idx, (t, s) in enumerate(zip(rec_texts, rec_scores), 1):
            lines.append({
                "line_no": idx,
                "text": t,
                "confidence": round(float(s), 4)
            })

    return {
        "image": img_path,
        "count": len(lines),
        "lines": lines
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70, flush=True)
    print("RUNNING PP-STRUCTUREV3 / PP-OCR ON ALL ANH1 RESTORED VARIANTS", flush=True)
    print("=" * 70, flush=True)

    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang="vi")

    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for var in variants:
        data = process_variant(ocr, var)
        summary[var] = data

        print(f"\nVariant: {var} -> Extracted {data['count']} text lines:", flush=True)
        print("-" * 60, flush=True)
        for line in data["lines"]:
            print(f"  Line {line['line_no']:2d} [{line['confidence']:.2f}]: {line['text']}", flush=True)

    # Save summary JSON
    out_file = os.path.join(out_dir, "anh1_all_restored_ocr_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70, flush=True)
    print(f"ALL ANH1 VARIANTS PROCESSED SUCCESSFULLY!", flush=True)
    print(f"Saved master JSON to: {out_file}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
