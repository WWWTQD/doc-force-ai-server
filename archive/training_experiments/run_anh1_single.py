import os
import sys
import json

try:
    import torch
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.structure_engine import StructureEngine


def main():
    print("=" * 60)
    print("RUNNING PP-STRUCTUREV3 ON ANH1 RESTORED IMAGES ONLY")
    print("=" * 60)

    # Initialize Singleton Engine
    engine = StructureEngine.get_instance()
    print("Initializing PP-StructureV3 engine...")
    engine.initialize(device="cpu", use_table_recognition=False)
    print("Engine initialized!")

    base_dir = "outputs/restoration/test"
    variants = ["Anh1_light", "Anh1_medium", "Anh1_strong"]

    output_dir = "outputs/ppstructure_anh1_results"
    os.makedirs(output_dir, exist_ok=True)

    summary_results = {}

    for var in variants:
        restored_path = os.path.normpath(os.path.join(base_dir, var, "restored.png"))

        if not os.path.exists(restored_path):
            print(f"Skipping {var}: file not found at {restored_path}")
            continue

        print(f"\n[+] Predicting RESTORED image: {var} ({restored_path})")
        res = engine.predict(restored_path)
        
        summary_results[var] = res
        print(f"    -> Complete! Items detected: {len(res)}")

        # Save individual JSON
        var_file = os.path.join(output_dir, f"{var}_restored_structure.json")
        with open(var_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    # Save summary JSON
    summary_path = os.path.join(output_dir, "anh1_restored_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"ALL ANH1 RESTORED PREDICTIONS COMPLETED!")
    print(f"Saved master JSON to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
