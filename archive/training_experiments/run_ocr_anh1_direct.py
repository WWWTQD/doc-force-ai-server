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
        extracted = []
        if results:
            for res in results:
                if hasattr(res, "json") and res.json:
                    ocr_res = res.json.get("ocr_res", {})
                    rec_texts = ocr_res.get("rec_text", [])
                    rec_scores = ocr_res.get("rec_score", [])
                    for text, score in zip(rec_texts, rec_scores):
                        extracted.append({
                            "text": text,
                            "confidence": round(float(score), 4)
                        })
                elif hasattr(res, "to_dict"):
                    d = res.to_dict()
                    rec_texts = d.get("rec_text", [])
                    rec_scores = d.get("rec_score", [])
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
    print("=" * 60)
    print("PADDLEOCR READING (PATCHED CPU) FOR ANH1 RESTORED IMAGES")
    print("=" * 60)

    # Disable UVDoc unwarping to prevent distorting zero-padded document images
    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang='vi')

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]

    output_summary = {}

    for var in variants:
        restored_img = os.path.join(base_dir, var, "restored.png")
        blur_img = os.path.join(base_dir, var, "input_blur.png")

        print(f"\n--- {var} ---")
        restored_res = process_image(ocr, restored_img)
        blur_res = process_image(ocr, blur_img)

        output_summary[var] = {
            "restored": restored_res,
            "blur": blur_res
        }

        print(f"  RESTORED: Extracted {restored_res.get('count', 0)} text lines.")
        if restored_res.get("lines"):
            for l in restored_res["lines"][:8]:
                # Safe print ascii representation for console log
                clean_text = l['text'].encode('ascii', 'ignore').decode('ascii')
                print(f"    - {clean_text if clean_text else l['text'].encode('utf-8')} (conf: {l['confidence']:.2f})")

        print(f"  BLUR:     Extracted {blur_res.get('count', 0)} text lines.")

    # Save output JSON with utf-8 encoding
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "anh1_restored_ocr_output.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"OCR Output saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
