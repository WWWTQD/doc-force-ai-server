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
    print("Initializing engine...")
    engine = StructureEngine.get_instance()
    engine.initialize(device="cpu", use_table_recognition=False)
    print("Engine initialized.")

    img_path = "outputs/restoration/test/Anh1_medium/restored.png"
    print(f"Predicting on: {img_path}")

    # Directly iterate prediction pipeline
    for i, item in enumerate(engine.pipeline.predict(img_path)):
        print(f"Prediction item {i} type: {type(item)}")
        print(f"  has json: {hasattr(item, 'json')}")
        if hasattr(item, "json"):
            print("  json keys:", list(item.json.keys()) if isinstance(item.json, dict) else "not dict")
            print("  json sample:", str(item.json)[:500])

if __name__ == "__main__":
    main()
