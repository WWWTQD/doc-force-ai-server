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


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_single_variant.py <variant_name>")
        sys.exit(1)

    var_name = sys.argv[1]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang="vi")

    base_dir = "outputs/restoration/test"
    img_path = os.path.normpath(os.path.join(base_dir, var_name, "restored.png"))

    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        sys.exit(1)

    print(f"Running PP-StructureV3 / PP-OCR on {var_name}: {img_path}...", flush=True)
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

    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{var_name}_ocr.json")

    output_data = {
        "variant": var_name,
        "image_path": img_path,
        "total_lines": len(lines),
        "lines": lines
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{var_name} Extracted {len(lines)} lines (Saved to {out_path}):", flush=True)
    print("-" * 60, flush=True)
    for l in lines:
        print(f"  Line {l['line_no']:2d} [{l['confidence']:.2f}]: {l['text']}", flush=True)


if __name__ == "__main__":
    main()
