import os
import sys
import json

# Pre-import torch to avoid Windows DLL errors
try:
    import torch
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.structure_engine import StructureEngine


def main():
    print("=" * 60)
    print("RUNNING PP-STRUCTUREV3 ON ANH1 RESTORED & BLUR IMAGES")
    print("=" * 60)

    # Initialize Singleton Engine
    engine = StructureEngine.get_instance()
    engine.initialize(device="cpu", use_table_recognition=False)

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]

    output_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(output_dir, exist_ok=True)

    summary_results = {}

    for var in variants:
        restored_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))
        blur_path = os.path.normpath(os.path.join(base_dir, var, "input_blur.png"))
        orig_path = os.path.normpath(os.path.join(base_dir, var, "original.png"))

        var_data = {}

        if os.path.exists(restored_path):
            print(f"\n[+] Processing {var} RESTORED: {restored_path}")
            res_restored = engine.predict(restored_path)
            var_data["restored"] = res_restored
            print(f"    -> Restored output generated. Items: {len(res_restored)}")

        if os.path.exists(blur_path):
            print(f"[+] Processing {var} BLUR: {blur_path}")
            res_blur = engine.predict(blur_path)
            var_data["blur"] = res_blur
            print(f"    -> Blur output generated. Items: {len(res_blur)}")

        summary_results[var] = var_data

        # Save result JSON for this variant
        var_file = os.path.join(output_dir, f"{var}_ppstructure_result.json")
        with open(var_file, "w", encoding="utf-8") as f:
            json.dump(var_data, f, indent=2, ensure_ascii=False)
        print(f"    -> Saved variant JSON to: {var_file}")

    # Save summary report
    summary_path = os.path.join(output_dir, "anh1_ppstructure_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"PP-STRUCTUREV3 SUMMARY SAVED TO: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
