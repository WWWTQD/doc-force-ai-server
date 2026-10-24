import os
import sys
import json

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.structure_engine import StructureEngine

def main():
    print("=" * 70)
    print("RUNNING PP-STRUCTURE V3 ON ANH1 RESTORED VARIANTS (LIGHT, MEDIUM, STRONG)")
    print("=" * 70)

    # Initialize PP-StructureV3 singleton engine
    engine = StructureEngine.get_instance()
    engine.initialize(
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_table_recognition=True,
        use_formula_recognition=False
    )

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]
    out_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(out_dir, exist_ok=True)

    summary_report = {}

    for var in variants:
        img_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))
        if not os.path.exists(img_path):
            print(f"Error: {img_path} not found.")
            continue

        print(f"\nProcessing {var}: {img_path}...")
        try:
            results = engine.predict(img_path)
            
            # Save raw structure JSON output for variant
            raw_out_path = os.path.join(out_dir, f"{var}_structure_raw.json")
            with open(raw_out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Extract clean layout and text items
            extracted_items = []
            ocr_text_lines = []

            for r_idx, res in enumerate(results):
                res_inner = res.get("res", {})
                
                # Check for parsing objects / layout elements
                parsing_res = res_inner.get("parsing_res_list", [])
                if parsing_res:
                    for item in parsing_res:
                        extracted_items.append({
                            "type": item.get("type"),
                            "bbox": item.get("bbox"),
                            "text": item.get("text", "").strip() if item.get("text") else None
                        })

                # Check rec_text / rec_score
                rec_texts = res_inner.get("rec_text", [])
                rec_scores = res_inner.get("rec_score", [])
                if rec_texts:
                    for text, score in zip(rec_texts, rec_scores):
                        ocr_text_lines.append({
                            "text": text,
                            "confidence": round(float(score), 4)
                        })

                # Check if result format is list of boxes/texts directly
                elif "overall_ocr_res" in res_inner:
                    ocr_res = res_inner["overall_ocr_res"]
                    texts = ocr_res.get("rec_text", [])
                    scores = ocr_res.get("rec_score", [])
                    for text, score in zip(texts, scores):
                        ocr_text_lines.append({
                            "text": text,
                            "confidence": round(float(score), 4)
                        })

            summary_report[var] = {
                "image_path": img_path,
                "total_text_lines": len(ocr_text_lines),
                "total_layout_blocks": len(extracted_items),
                "text_lines": ocr_text_lines,
                "layout_blocks": extracted_items
            }

            print(f"  [SUCCESS] Extracted {len(ocr_text_lines)} text lines and {len(extracted_items)} layout blocks.")

        except Exception as e:
            print(f"  [ERROR] Failed to process {var}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary report JSON
    summary_path = os.path.join(out_dir, "anh1_ppstructure_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"COMPLETE! PP-StructureV3 output saved to {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
