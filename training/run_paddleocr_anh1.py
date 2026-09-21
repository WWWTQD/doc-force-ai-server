import os
import sys
import json

try:
    import torch
except Exception:
    pass

from paddleocr import PaddleOCR

def main():
    print("=" * 60)
    print("RUNNING PADDLEOCR ON ANH1 RESTORED & BLUR IMAGES")
    print("=" * 60)

    # Initialize PaddleOCR engine (Vietnamese / English)
    ocr = PaddleOCR(use_angle_cls=False, lang='vi')

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]

    output_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(output_dir, exist_ok=True)

    summary = {}

    for var in variants:
        restored_path = os.path.join(base_dir, var, "restored.png")
        blur_path = os.path.join(base_dir, var, "input_blur.png")

        var_data = {}

        if os.path.exists(restored_path):
            print(f"\n---> Analyzing {var} RESTORED: {restored_path}")
            res_restored = ocr.ocr(restored_path, cls=False)
            lines_restored = []
            if res_restored and res_restored[0]:
                for line in res_restored[0]:
                    box, (text, score) = line
                    lines_restored.append({
                        "text": text,
                        "confidence": round(float(score), 4),
                        "box": [[int(p[0]), int(p[1])] for p in box]
                    })
            print(f"     Recognized {len(lines_restored)} text lines.")
            var_data["restored"] = {
                "line_count": len(lines_restored),
                "lines": lines_restored
            }

        if os.path.exists(blur_path):
            print(f"---> Analyzing {var} BLUR: {blur_path}")
            res_blur = ocr.ocr(blur_path, cls=False)
            lines_blur = []
            if res_blur and res_blur[0]:
                for line in res_blur[0]:
                    box, (text, score) = line
                    lines_blur.append({
                        "text": text,
                        "confidence": round(float(score), 4),
                        "box": [[int(p[0]), int(p[1])] for p in box]
                    })
            print(f"     Recognized {len(lines_blur)} text lines.")
            var_data["blur"] = {
                "line_count": len(lines_blur),
                "lines": lines_blur
            }

        summary[var] = var_data

        # Save result JSON for this variant
        with open(os.path.join(output_dir, f"{var}_ocr_result.json"), "w", encoding="utf-8") as f:
            json.dump(var_data, f, indent=2, ensure_ascii=False)

    # Save summary report
    summary_path = os.path.join(output_dir, "anh1_ocr_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"PADDLEOCR SUMMARY SAVED TO: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
