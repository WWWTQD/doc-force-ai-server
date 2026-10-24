import os
import sys
import json

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
except Exception:
    pass

from paddleocr import PaddleOCR

def process_file(ocr, img_path):
    if not os.path.exists(img_path):
        return {"error": "file_not_found"}
    try:
        results = list(ocr.predict(img_path))
        extracted = []
        for r in results:
            raw_data = r.json if hasattr(r, "json") and r.json else {}
            res_inner = raw_data.get("res", {})
            rec_texts = res_inner.get("rec_text", [])
            rec_scores = res_inner.get("rec_score", [])
            for text, score in zip(rec_texts, rec_scores):
                extracted.append({
                    "text": text,
                    "confidence": round(float(score), 4)
                })
        return {
            "image": img_path,
            "count": len(extracted),
            "lines": extracted
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("Extracting text from Anh1 restored images...")
    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang="vi")

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]

    master_results = {}

    for var in variants:
        restored_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))
        blur_path = os.path.normpath(os.path.join(base_dir, var, "input_blur.png"))

        restored_out = process_file(ocr, restored_path)
        blur_out = process_file(ocr, blur_path)

        master_results[var] = {
            "restored": restored_out,
            "blur": blur_out
        }

        print(f"\n--- {var} ---")
        print(f"  RESTORED: {restored_out.get('count', 0)} lines extracted")
        if restored_out.get("lines"):
            for l in restored_out["lines"][:8]:
                # Safe print ascii representation for console
                ascii_text = l['text'].encode('ascii', 'ignore').decode('ascii')
                print(f"    - {ascii_text if ascii_text.strip() else l['text'].encode('utf-8')}")
        print(f"  BLUR:     {blur_out.get('count', 0)} lines extracted")

    # Save to JSON file with UTF-8
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "anh1_restored_ocr_output.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Extracted results saved to: {out_file}")

if __name__ == "__main__":
    main()
