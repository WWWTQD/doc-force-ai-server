"""
Generate restoration dataset metadata CSVs directly from datasets/processed/
Supports 223 samples across Anh01-Anh22:
  - train.csv: 183 samples (Anh01-Anh09, Anh12-Anh20)
  - val.csv: 24 samples (Anh10, Anh22)
  - test.csv: 16 samples (Anh11, Anh21)
  - dataset_manifest.csv: 223 samples total
"""
import os
import re
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("datasets")
PROCESSED = ROOT / "processed"
METADATA = ROOT / "metadata"

SPECIAL_MAP = {
    # Train
    "ColorJitter_Anh09.png": ("Anh09", "ColorJitter", "01"),
    "LowInkRandomLines+LightingGradient.png": ("Anh09", "LowInkRandomLines+LightingGradient", "01"),
    "LightingGradient+BookBinding+Perspective+ISONoise_Anh16.png": ("Anh16", "LightingGradient+BookBinding+Perspective+ISONoise", "05"),
    "ShadowCast+ColorJitter+Perspective+OpticalDistortion_Anh16.png": ("Anh16", "ShadowCast+ColorJitter+Perspective+OpticalDistortion", "06"),
    "LightingGradient+ShadowCast+OpticalDistortion.png": ("Anh16", "LightingGradient+ShadowCast+OpticalDistortion", "07"),
    "MotionBlur+Defocus.png": ("Anh16", "MotionBlur+Defocus", "08"),
    "Defocus+Perspective.png": ("Anh16", "Defocus+Perspective", "09"),
    "Perspective+Defocus+MotionBlur.png": ("Anh16", "Perspective+Defocus+MotionBlur", "10"),
    "Perspective+ISONoise.png": ("Anh16", "Perspective+ISONoise", "11"),
    "LowInkRandomLines.png": ("Anh20", "LowInkRandomLines", "12"),
    "LowLightNoise.png": ("Anh20", "LowLightNoise", "13"),
    # Val
    "LowInkRandomLines+OpticalDistortion_Anh10_07_Anh22_04.png": ("Anh10", "LowInkRandomLines+OpticalDistortion", "07"),
}

STD_RE = re.compile(r"^(?P<deg>.+?)_(?P<id>Anh\d+)_(?P<var>\d+)$", re.IGNORECASE)


def parse_filename(fname: str):
    if fname in SPECIAL_MAP:
        return SPECIAL_MAP[fname]
    stem, _ = os.path.splitext(fname)
    m = STD_RE.match(stem)
    if m:
        deg = m.group("deg")
        oid = m.group("id")
        oid = f"Anh{int(oid[3:]):02d}"
        var = f"{int(m.group('var')):02d}"
        return oid, deg, var
    raise ValueError(f"Could not parse filename: {fname}")


def generate_metadata():
    METADATA.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    manifest_rows = []
    split_records = {"train": [], "val": [], "test": []}

    for split in splits:
        input_dir = PROCESSED / split / "input"
        target_dir = PROCESSED / split / "target"

        # Map original_id -> target filename (e.g. 'Anh01': 'Anh01.png')
        target_map = {}
        for tf in sorted(os.listdir(target_dir)):
            oid = os.path.splitext(tf)[0]
            # Normalize Anh01
            m = re.match(r"^anh0*(\d+)$", oid, re.IGNORECASE)
            if m:
                oid_norm = f"Anh{int(m.group(1)):02d}"
                target_map[oid_norm] = tf

        files = sorted(os.listdir(input_dir))
        for fname in files:
            oid, deg, var = parse_filename(fname)
            if oid not in target_map:
                raise FileNotFoundError(f"Missing target for {oid} (file: {fname}) in {target_dir}")

            target_filename = target_map[oid]
            in_rel_repo = f"datasets/processed/{split}/input/{fname}"
            tgt_rel_repo = f"datasets/processed/{split}/target/{target_filename}"
            in_rel_manifest = f"processed/{split}/input/{fname}"
            tgt_rel_manifest = f"processed/{split}/target/{target_filename}"

            blur_level = f"{deg}_{var}"

            split_records[split].append({
                "input": in_rel_repo,
                "target": tgt_rel_repo,
                "source_id": oid,
                "degradation": deg,
                "blur_level": blur_level,
            })

            manifest_rows.append({
                "split": split,
                "original_id": oid,
                "degradation_type": deg,
                "variant_id": var,
                "input_path": in_rel_manifest,
                "target_path": tgt_rel_manifest,
            })

    # Sort manifest rows by split, original_id, degradation_type, variant_id
    split_order = {"train": 0, "val": 1, "test": 2}
    manifest_rows.sort(key=lambda r: (split_order.get(r["split"], 99), r["original_id"], r["degradation_type"], r["variant_id"]))

    # Write train.csv, val.csv, test.csv
    fieldnames_split = ["input", "target", "source_id", "degradation", "blur_level"]
    for split in splits:
        out_csv = METADATA / f"{split}.csv"
        recs = split_records[split]
        # Sort records for determinism
        recs.sort(key=lambda r: (r["source_id"], r["degradation"], r["blur_level"], r["input"]))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_split)
            w.writeheader()
            w.writerows(recs)
        print(f"Generated {out_csv}: {len(recs)} rows")

    # Write dataset_manifest.csv
    manifest_csv = METADATA / "dataset_manifest.csv"
    fieldnames_manifest = ["split", "original_id", "degradation_type", "variant_id", "input_path", "target_path"]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_manifest)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Generated {manifest_csv}: {len(manifest_rows)} rows")

    # Write updated dataset_report.json
    report_json = METADATA / "dataset_report.json"
    per_split = {}
    for split in splits:
        oids = sorted(list({r["source_id"] for r in split_records[split]}))
        per_split[split] = {
            "inputs": len(split_records[split]),
            "unique_targets": len(oids),
            "original_ids": oids,
        }

    summary = {
        "total_inputs": len(manifest_rows),
        "unique_targets": sum(len(v["original_ids"]) for v in per_split.values()),
        "per_split": per_split,
        "inputs_per_degradation": dict(Counter(r["degradation_type"] for r in manifest_rows)),
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Generated {report_json}")

    return len(split_records["train"]), len(split_records["val"]), len(split_records["test"]), len(manifest_rows)


if __name__ == "__main__":
    n_train, n_val, n_test, n_tot = generate_metadata()
    print(f"\nSummary: train={n_train}, val={n_val}, test={n_test}, total={n_tot}")
    assert n_train == 183, f"Expected 183 train rows, got {n_train}"
    assert n_val == 24, f"Expected 24 val rows, got {n_val}"
    assert n_test == 16, f"Expected 16 test rows, got {n_test}"
    assert n_tot == 223, f"Expected 223 total manifest rows, got {n_tot}"
    print("All assertions passed successfully!")
