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


def process_image_lines(ocr, img_path):
    if not os.path.exists(img_path):
        return {"error": "file_not_found", "count": 0, "lines": []}

    try:
        # Run PaddleOCR prediction
        results = ocr.ocr(img_path, cls=False)
        extracted = []
        if results and results[0]:
            for line_idx, line in enumerate(results[0], 1):
                box, (text, score) = line
                extracted.append({
                    "line_no": line_idx,
                    "text": text.strip(),
                    "confidence": round(float(score), 4),
                    "box": box
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
    print("=" * 70)
    print("RUNNING PP-STRUCTUREV3 / PP-OCR ON ALL ANH1 RESTORED VARIANTS")
    print("=" * 70)

    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang="vi")

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)

    master_results = {}

    for var in variants:
        img_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))
        data = process_image_lines(ocr, img_path)
        master_results[var] = data

        var_out_file = os.path.join(out_dir, f"{var}_ocr_parsed.json")
        with open(var_out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nVariant: {var} ({data['count']} text lines extracted)")
        print("-" * 50)
        for line in data["lines"]:
            print(f"  Line {line['line_no']:2d} [{line['confidence']:.2f}]: {line['text']}")

    # Save summary master output
    master_file = os.path.join(out_dir, "anh1_all_restored_ocr_summary.json")
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"ALL ANH1 VARIANTS PROCESSED SUCCESSFULLY!")
    print(f"Master summary saved to: {master_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
