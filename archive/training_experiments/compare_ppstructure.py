import os
import sys
import argparse
import csv
import cv2
import json
import numpy as np

# Try to import PP-StructureV3
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.services.structure_engine import StructureEngine
except ImportError:
    print("Warning: PP-StructureV3 engine not available (import failed). Exiting gracefully.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Compare PP-StructureV3 on blur vs restored images")
    parser.add_argument('--test-csv', type=str, default='datasets/metadata/test.csv')
    args = parser.parse_args()

    # Initialize PP-StructureV3
    try:
        engine = StructureEngine.get_instance()
        if not engine.initialized:
            print("Initializing PP-StructureV3 engine...")
            engine.initialize(device='cpu')
    except Exception as e:
        print(f"Warning: Failed to initialize PP-StructureV3: {e}. Exiting gracefully.")
        sys.exit(0)

    out_dir = 'outputs/restoration/ppstructure_comparison'
    os.makedirs(out_dir, exist_ok=True)

    # Read test CSV
    test_pairs = []
    with open(args.test_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_pairs.append(row)

    if not test_pairs:
        print("No test pairs found. Exiting.")
        return

    processed = 0
    errors = 0

    for row in test_pairs:
        source_id = row['source_id']
        blur_level = row.get('blur_level', 'unknown')
        blur_path = os.path.normpath(row['input'])
        sample_name = f"{source_id}_{blur_level}"
        restored_path = os.path.join('outputs', 'restoration', 'test', sample_name, 'restored.png')

        if not os.path.exists(blur_path):
            print(f"  SKIP {sample_name}: blur image not found at {blur_path}")
            continue
        if not os.path.exists(restored_path):
            print(f"  SKIP {sample_name}: restored image not found at {restored_path}")
            continue

        # Run PP-StructureV3 on blur image
        try:
            blur_results = engine.predict(blur_path)
            blur_json_path = os.path.join(out_dir, f"{sample_name}_blur.json")
            with open(blur_json_path, 'w', encoding='utf-8') as f:
                json.dump(blur_results, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"  ERROR on blur image {sample_name}: {e}")
            errors += 1

        # Run PP-StructureV3 on restored image
        try:
            restored_results = engine.predict(restored_path)
            restored_json_path = os.path.join(out_dir, f"{sample_name}_restored.json")
            with open(restored_json_path, 'w', encoding='utf-8') as f:
                json.dump(restored_results, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"  ERROR on restored image {sample_name}: {e}")
            errors += 1

        processed += 1
        print(f"  Processed PP-StructureV3 for {sample_name}")

    print(f"\nSummary: Processed {processed} pairs. Errors: {errors}.")


if __name__ == '__main__':
    main()
