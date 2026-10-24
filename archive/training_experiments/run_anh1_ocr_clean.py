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
        return {"error": "file_not_found"}

    try:
        results = list(ocr.predict(img_path))
        lines = []
        for r in results:
            raw_dict = r.json if hasattr(r, "json") and r.json else {}
            res_inner = raw_dict.get("res", {})
            rec_texts = res_inner.get("rec_text", [])
            rec_scores = res_inner.get("rec_score", [])
            for t, s in zip(rec_texts, rec_scores):
                lines.append({
                    "text": t,
                    "confidence": round(float(s), 4)
                })

        return {
            "image": img_path,
            "count": len(lines),
            "lines": lines
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("PP-STRUCTUREV3 / PADDLEOCR READING FOR ANH1 RESTORED IMAGES")
    print("=" * 60)

    # Disable UVDoc unwarping to prevent distorting zero-padded document images
    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang='vi')

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for var in variants:
        restored_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))

        print(f"\n[+] Processing {var} RESTORED: {restored_path}")
        out_data = process_image(ocr, restored_path)
        summary[var] = out_data

        print(f"    -> Extracted {out_data.get('count', 0)} text lines.")
        if out_data.get("lines"):
            for line_item in out_data["lines"]:
                # Print all recognized text
                print(f"       * {line_item['text']} (conf: {line_item['confidence']:.2f})")

        # Save individual variant JSON immediately
        var_file = os.path.join(out_dir, f"{var}_ocr_result.json")
        with open(var_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

    # Save summary JSON
    out_file = os.path.join(out_dir, "anh1_restored_ocr_output.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"ALL ANH1 RESTORED PREDICTIONS COMPLETED!")
    print(f"Master JSON saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
