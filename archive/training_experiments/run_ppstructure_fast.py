import os
import sys
import json
import logging

try:
    import torch
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def format_ppstructure_output(res_list):
    """Format PP-StructureV3 output into clean human-readable text and layout elements."""
    extracted_texts = []
    layout_elements = []

    for item in res_list:
        if isinstance(item, dict):
            # Extract layout or OCR results
            res_dict = item.get("parsing_res", item)
            
            # Check for layout blocks or text elements
            if "layout_det_res" in item:
                boxes = item["layout_det_res"].get("boxes", [])
                for b in boxes:
                    layout_elements.append({
                        "label": b.get("label", "text"),
                        "score": round(float(b.get("score", 0)), 4),
                        "box": b.get("coordinate", [])
                    })

            if "ocr_res" in item:
                rec_texts = item["ocr_res"].get("rec_text", [])
                rec_scores = item["ocr_res"].get("rec_score", [])
                for text, score in zip(rec_texts, rec_scores):
                    extracted_texts.append(f"{text} (conf: {score:.2f})")

    return {
        "text_count": len(extracted_texts),
        "texts": extracted_texts,
        "layout_count": len(layout_elements),
        "layouts": layout_elements,
        "raw": res_list
    }


def main():
    print("=" * 60)
    print("RUNNING PP-STRUCTUREV3 (FAST MODE) ON ANH1 RESTORED IMAGES")
    print("=" * 60)

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]

    from app.services.structure_engine import StructureEngine
    engine = StructureEngine.get_instance()

    print("Initializing PP-StructureV3 (Layout + OCR)...")
    engine.initialize(device="cpu", use_table_recognition=False)

    output_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(output_dir, exist_ok=True)

    summary_report = {}

    for var in variants:
        restored_path = os.path.join(base_dir, var, "restored.png")
        blur_path = os.path.join(base_dir, var, "input_blur.png")

        if not os.path.exists(restored_path):
            continue

        print(f"\nAnalyzing {var} restored image: {restored_path}")
        res_restored = engine.predict(restored_path)
        parsed_restored = format_ppstructure_output(res_restored)

        parsed_blur = None
        if os.path.exists(blur_path):
            print(f"Analyzing {var} blur image: {blur_path}")
            res_blur = engine.predict(blur_path)
            parsed_blur = format_ppstructure_output(res_blur)

        summary_report[var] = {
            "restored": parsed_restored,
            "blur": parsed_blur,
        }

        # Save result JSON for this variant
        with open(os.path.join(output_dir, f"{var}_fast.json"), "w", encoding="utf-8") as f:
            json.dump({
                "restored": parsed_restored,
                "blur": parsed_blur
            }, f, indent=2, ensure_ascii=False)

    # Save master summary
    with open(os.path.join(output_dir, "anh1_fast_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FINISHED PP-STRUCTUREV3 FAST ANALYSIS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
