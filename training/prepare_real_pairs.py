import os
import sys
import csv
import glob
import shutil
import argparse
import logging
from typing import List, Dict, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.restoration_dataset import (
    load_image,
    save_image,
    unsharp_mask,
    _sauvola_threshold_integral,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_real_pairs")


def refine_real_clean_target(raw_bgr: np.ndarray) -> np.ndarray:
    """
    Create a Master Clean Target (`clean_real_target`) from a real camera document photo:
    1. Multi-Scale Morphological Closing (`21x21`, `45x45`, `75x75`) to flatten uneven background
       illumination and book spine shadows to pure white `255`.
    2. Bleed-through & Watermark Suppression: Regions with normalized intensity `> 210` and low
       edge gradient are pushed cleanly to `255`, while 100% protecting front-side printed text
       and thin handwritten ballpoint pen strokes.
    3. Stroke Contrast & Sharpness Enhancement: Maps foreground stroke cores to deep crisp ink
       (`0 -> 30`) with smooth anti-aliased edges.
    """
    if raw_bgr.ndim == 2:
        raw_gray = raw_bgr.copy()
        blue_ink_boost = raw_gray
    else:
        b_ch, g_ch, r_ch = cv2.split(raw_bgr)
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        # Protect blue ballpoint pen strokes and blue links (where B > R and G is darker than B)
        blue_excess = b_ch.astype(np.float32) - np.maximum(r_ch.astype(np.float32), g_ch.astype(np.float32))
        rg_mean = 0.5 * (r_ch.astype(np.float32) + g_ch.astype(np.float32))
        blue_ink_boost = np.where(
            blue_excess > 12.0,
            np.minimum(raw_gray.astype(np.float32), rg_mean),
            raw_gray.astype(np.float32),
        ).astype(np.uint8)

    # Step 1: Edge-Preserving Pre-Denoising
    denoised = cv2.bilateralFilter(blue_ink_boost, d=5, sigmaColor=20, sigmaSpace=20)

    # Step 2: Multi-Scale Morphological Closing for Background Illumination Flattening
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))

    bg_small = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, k_small)
    bg_med = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, k_med)
    bg_large = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, k_large)
    bg = np.maximum(np.maximum(bg_small, bg_med), bg_large)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=15, sigmaY=15)

    norm = np.clip(
        (denoised.astype(np.float32) / np.maximum(bg.astype(np.float32), 15.0)) * 255.0,
        0.0,
        255.0,
    )

    #Also compute standard luminance normalized reflectance to detect pastel/orange watermarks
    if raw_bgr.ndim == 3:
        denoised_luma = cv2.bilateralFilter(raw_gray, d=5, sigmaColor=20, sigmaSpace=20)
        bg_luma = cv2.GaussianBlur(
            np.maximum(
                cv2.morphologyEx(denoised_luma, cv2.MORPH_CLOSE, k_small),
                cv2.morphologyEx(denoised_luma, cv2.MORPH_CLOSE, k_med),
            ),
            (0, 0),
            sigmaX=15,
            sigmaY=15,
        )
        norm_luma = np.clip(
            (denoised_luma.astype(np.float32) / np.maximum(bg_luma.astype(np.float32), 15.0)) * 255.0,
            0.0,
            255.0,
        )
    else:
        norm_luma = norm

    # Step 3: Edge Gradient & Multi-Scale Sauvola Thresholding
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    local_grad = cv2.dilate(grad_mag, k_grad)

    t_small = _sauvola_threshold_integral(norm, window_size=15, k=0.14, r=128.0)
    t_med = _sauvola_threshold_integral(norm, window_size=31, k=0.18, r=128.0)
    t_sauvola = np.maximum(t_small, t_med)

    # Bleed-through and watermark suppression:
    # - Regions with normalized intensity > 210 and low edge gradient (< 180) are pushed to 255
    # - Regions with norm > 220 or above Sauvola threshold are background (255)
    # - Thin ballpoint pen strokes have sharp local gradients (>= 180) or darker cores (norm <= 210)
    is_bleed_or_watermark = (
        (norm_luma > 210.0) & (local_grad < 180.0)
    ) | (norm > 220.0) | ((norm - t_sauvola) > 4.0)

    fg_mask = ~is_bleed_or_watermark

    # Connected-component filtering to remove isolated faint bleed-through / dust islands
    fg_u8 = fg_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    lap_g = (
        np.abs(cv2.Laplacian(raw_bgr[:, :, 1].astype(np.float32), cv2.CV_32F, ksize=3))
        if raw_bgr.ndim == 3
        else np.abs(cv2.Laplacian(raw_gray.astype(np.float32), cv2.CV_32F, ksize=3))
    )
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        x0 = stats[lbl, cv2.CC_STAT_LEFT]
        y0 = stats[lbl, cv2.CC_STAT_TOP]
        w0 = stats[lbl, cv2.CC_STAT_WIDTH]
        h0 = stats[lbl, cv2.CC_STAT_HEIGHT]
        roi_mask = labels[y0:y0 + h0, x0:x0 + w0] == lbl
        roi_norm = norm_luma[y0:y0 + h0, x0:x0 + w0][roi_mask]
        roi_grad = grad_mag[y0:y0 + h0, x0:x0 + w0][roi_mask]

        min_val = float(np.min(roi_norm))
        max_g = float(np.max(roi_grad))
        # Remove micro-speckles or faint diffuse bleed-through blobs
        if (area <= 3 and min_val > 145.0) or (min_val > 206.0 and max_g < 210.0):
            fg_mask[y0:y0 + h0, x0:x0 + w0][roi_mask] = False
            continue

        # Suppress diffuse back-side red stamp bleed-through (high Red channel, low Laplacian sharpness)
        if raw_bgr.ndim == 3:
            roi_r = raw_bgr[y0:y0 + h0, x0:x0 + w0, 2][roi_mask].astype(np.float32)
            roi_g = raw_bgr[y0:y0 + h0, x0:x0 + w0, 1][roi_mask].astype(np.float32)
            roi_b = raw_bgr[y0:y0 + h0, x0:x0 + w0, 0][roi_mask].astype(np.float32)
            red_exc = float(np.mean(roi_r - 0.5 * (roi_g + roi_b)))
            if red_exc > 25.0 and float(np.min(roi_r)) > 88.0:
                roi_lap = float(np.percentile(lap_g[y0:y0 + h0, x0:x0 + w0][roi_mask], 95))
                if roi_lap < 185.0 and float(np.percentile(roi_g, 10)) > 28.0:
                    fg_mask[y0:y0 + h0, x0:x0 + w0][roi_mask] = False

    # Step 4: Smooth Anti-Aliased Contrast Mapping (Stroke Cores -> [0, 30])
    matted_f = np.full_like(norm, 255.0, dtype=np.float32)
    if np.any(fg_mask):
        fg_vals = norm[fg_mask]
        black_floor = float(np.clip(np.percentile(fg_vals, 25), 35.0, 155.0))
        upper_bound = np.minimum(t_sauvola + 4.0, 218.0)
        span = np.maximum(upper_bound - black_floor, 18.0)

        alpha = np.clip((upper_bound - norm) / span, 0.0, 1.0)
        alpha_boosted = np.power(alpha, 0.62)
        # Map stroke cores cleanly into [0, 30] and anti-aliased edges smoothly to 255
        stroke_val = 255.0 * (1.0 - alpha_boosted) * 0.88
        matted_f = np.where(fg_mask, stroke_val, 255.0)

    matted = np.clip(matted_f, 0.0, 255.0).astype(np.uint8)

    # Step 5: Foreground Unsharp Masking for crisp stroke edges
    sharpened = unsharp_mask(matted, amount=0.75, radius=1.0)
    clean_target = np.where(matted < 245, sharpened, 255).astype(np.uint8)
    return clean_target


BLIND_EVAL_BASENAMES = {"real_11", "real_12"}


def _is_blind_eval_name(name_or_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(name_or_path))[0].lower()
    return stem in BLIND_EVAL_BASENAMES


def purge_blind_eval_from_real_pairs(dataset_root: str = "datasets") -> List[str]:
    """
    Strictly remove `real_11.*` and `real_12.*` from `datasets/processed/real_pairs/{input,target}/`
    if present, preventing any data leakage into the training set.
    """
    removed = []
    for sub in ("input", "target"):
        folder = os.path.join(dataset_root, "processed", "real_pairs", sub)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if _is_blind_eval_name(fname):
                fpath = os.path.join(folder, fname)
                try:
                    os.remove(fpath)
                    removed.append(fpath)
                    logger.warning(f"Removed leaked blind-eval file from real_pairs: {fpath}")
                except OSError:
                    pass
    return removed


def update_metadata_with_real_pairs(
    real_records: List[Dict[str, str]],
    dataset_root: str = "datasets",
    train_ratio: float = 1.0,
    train_oversample: int = 3,
) -> Tuple[int, int]:
    """
    Put valid `real_pairs` into `datasets/metadata/train.csv` (with `oversample x3`),
    strictly excluding `real_11` and `real_12`.
    Also registers `real_12.jpg` in `val.csv` and `real_11.jpg` in `test.csv` as
    unpaired blind evaluation entries (`target=""`, `degradation="unpaired_real"`).
    Idempotent: removes previous `real_pairs` and `unpaired_real` entries before appending.
    """
    meta_dir = os.path.join(dataset_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # Filter out any accidental blind eval records
    safe_records = [
        r for r in real_records
        if not _is_blind_eval_name(r.get("source_id", ""))
        and not _is_blind_eval_name(r.get("csv_input", ""))
    ]

    n_total = len(safe_records)
    if train_ratio >= 1.0:
        n_train = n_total
    else:
        n_train = max(1, int(round(n_total * train_ratio))) if n_total > 1 else n_total
    train_records = safe_records[:n_train]
    val_records = safe_records[n_train:]

    def _is_real_pair_or_blind_row(row: Dict[str, str]) -> bool:
        row_str = " ".join(str(v) for v in row.values()).lower().replace("\\", "/")
        deg = (row.get("degradation") or row.get("degradation_type") or "").lower()
        return (
            "real_pairs" in row_str
            or deg in ("real_pairs", "unpaired_real")
            or "real_11" in row_str
            or "real_12" in row_str
        )

    # 1. Update train.csv (strictly no real_11 or real_12)
    train_csv_path = os.path.join(meta_dir, "train.csv")
    train_fields = ["input", "target", "source_id", "degradation", "blur_level"]
    existing_train = []
    if os.path.exists(train_csv_path):
        with open(train_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                train_fields = list(reader.fieldnames)
            existing_train = [r for r in reader if not _is_real_pair_or_blind_row(r)]

    added_train = 0
    for rec in train_records:
        for rep in range(1, train_oversample + 1):
            row_data = {
                "input": rec["csv_input"],
                "target": rec["csv_target"],
                "source_id": rec["source_id"],
                "degradation": "real_pairs",
                "blur_level": f"real_pairs_rep{rep:02d}",
            }
            existing_train.append({k: row_data.get(k, "") for k in train_fields})
            added_train += 1

    with open(train_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_fields)
        writer.writeheader()
        writer.writerows(existing_train)

    # 2. Update val.csv (include unpaired real_12.jpg if present in datasets/processed/val/input/)
    val_csv_path = os.path.join(meta_dir, "val.csv")
    val_fields = ["input", "target", "source_id", "degradation", "blur_level"]
    existing_val = []
    if os.path.exists(val_csv_path):
        with open(val_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                val_fields = list(reader.fieldnames)
            existing_val = [r for r in reader if not _is_real_pair_or_blind_row(r)]

    added_val = 0
    for rec in val_records:
        row_data = {
            "input": rec["csv_input"],
            "target": rec["csv_target"],
            "source_id": rec["source_id"],
            "degradation": "real_pairs",
            "blur_level": "real_pairs_val",
        }
        existing_val.append({k: row_data.get(k, "") for k in val_fields})
        added_val += 1

    val_real12_abs = os.path.join(dataset_root, "processed", "val", "input", "real_12.jpg")
    if os.path.exists(val_real12_abs):
        row_real12 = {
            "input": "datasets/processed/val/input/real_12.jpg",
            "target": "",
            "source_id": "real_12",
            "degradation": "unpaired_real",
            "blur_level": "real_blind_val",
        }
        existing_val.append({k: row_real12.get(k, "") for k in val_fields})

    with open(val_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=val_fields)
        writer.writeheader()
        writer.writerows(existing_val)

    # 3. Update test.csv (include unpaired real_11.jpg if present in datasets/processed/test/input/)
    test_csv_path = os.path.join(meta_dir, "test.csv")
    test_fields = ["input", "target", "source_id", "degradation", "blur_level"]
    existing_test = []
    if os.path.exists(test_csv_path):
        with open(test_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                test_fields = list(reader.fieldnames)
            existing_test = [r for r in reader if not _is_real_pair_or_blind_row(r)]

        test_real11_abs = os.path.join(dataset_root, "processed", "test", "input", "real_11.jpg")
        if os.path.exists(test_real11_abs):
            row_real11 = {
                "input": "datasets/processed/test/input/real_11.jpg",
                "target": "",
                "source_id": "real_11",
                "degradation": "unpaired_real",
                "blur_level": "real_blind_test",
            }
            existing_test.append({k: row_real11.get(k, "") for k in test_fields})

        with open(test_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=test_fields)
            writer.writeheader()
            writer.writerows(existing_test)

    # 4. Sync dataset_manifest.csv
    manifest_path = os.path.join(meta_dir, "dataset_manifest.csv")
    manifest_fields = [
        "split",
        "original_id",
        "degradation_type",
        "variant_id",
        "input_path",
        "target_path",
    ]
    existing_manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                manifest_fields = list(reader.fieldnames)
            existing_manifest = [r for r in reader if not _is_real_pair_or_blind_row(r)]

    for rec in train_records:
        for rep in range(1, train_oversample + 1):
            m_row = {
                "split": "train",
                "original_id": rec["source_id"],
                "degradation_type": "real_pairs",
                "variant_id": f"{rep:02d}",
                "input_path": rec["rel_input"],
                "target_path": rec["rel_target"],
            }
            existing_manifest.append({k: m_row.get(k, "") for k in manifest_fields})

    for rec in val_records:
        m_row = {
            "split": "val",
            "original_id": rec["source_id"],
            "degradation_type": "real_pairs",
            "variant_id": "01",
            "input_path": rec["rel_input"],
            "target_path": rec["rel_target"],
        }
        existing_manifest.append({k: m_row.get(k, "") for k in manifest_fields})

    if os.path.exists(os.path.join(dataset_root, "processed", "val", "input", "real_12.jpg")):
        m_val_blind = {
            "split": "val",
            "original_id": "real_12",
            "degradation_type": "unpaired_real",
            "variant_id": "01",
            "input_path": "processed/val/input/real_12.jpg",
            "target_path": "",
        }
        existing_manifest.append({k: m_val_blind.get(k, "") for k in manifest_fields})

    if os.path.exists(os.path.join(dataset_root, "processed", "test", "input", "real_11.jpg")):
        m_test_blind = {
            "split": "test",
            "original_id": "real_11",
            "degradation_type": "unpaired_real",
            "variant_id": "01",
            "input_path": "processed/test/input/real_11.jpg",
            "target_path": "",
        }
        existing_manifest.append({k: m_test_blind.get(k, "") for k in manifest_fields})

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(existing_manifest)

    return added_train, added_val


def prepare_real_pairs(
    real_dir: str = "datasets/real",
    dataset_root: str = "datasets",
    train_ratio: float = 1.0,
    train_oversample: int = 3,
) -> List[Dict[str, str]]:
    """
    Prepare (raw_real_input -> clean_real_target) pairs for training:
    - Strictly excludes `real_11.jpg` and `real_12.jpg` from `real_pairs` and `train.csv`
      (and removes them from `datasets/processed/real_pairs/` if accidentally present).
    - Scans `datasets/processed/real_pairs/input/` (and `datasets/real/`) for the 18 training
      `real_01..10`, `real_13..20` images + 10 `IMG20261022*.jpg` images.
    - Generates clean targets via `refine_real_clean_target` into `datasets/processed/real_pairs/target/`.
    - Adds all valid `real_pairs` to `datasets/metadata/train.csv` with `oversample x3`.
    """
    input_out_dir = os.path.join(dataset_root, "processed", "real_pairs", "input")
    target_out_dir = os.path.join(dataset_root, "processed", "real_pairs", "target")
    os.makedirs(input_out_dir, exist_ok=True)
    os.makedirs(target_out_dir, exist_ok=True)

    # 1. Prevent data leakage: remove real_11 / real_12 if accidentally placed in real_pairs
    purge_blind_eval_from_real_pairs(dataset_root=dataset_root)

    # 2. Sync valid training images from real_dir (IMG*.jpg and real_*.jpg except real_11 & real_12)
    #    into datasets/processed/real_pairs/input/ if not already there
    if os.path.exists(real_dir):
        for pat in ("IMG*.jpg", "real_*.jpg"):
            for src_p in sorted(glob.glob(os.path.join(real_dir, pat))):
                if _is_blind_eval_name(src_p):
                    continue
                dst_p = os.path.join(input_out_dir, os.path.basename(src_p))
                if not os.path.exists(dst_p):
                    shutil.copy2(src_p, dst_p)

    # 3. Collect all valid training inputs from datasets/processed/real_pairs/input/
    candidate_paths = []
    for entry in sorted(os.listdir(input_out_dir)):
        full_p = os.path.join(input_out_dir, entry)
        if not os.path.isfile(full_p):
            continue
        if _is_blind_eval_name(entry):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".jfif"):
            candidate_paths.append(full_p)

    if not candidate_paths:
        logger.warning(f"No valid real training images found in '{input_out_dir}'.")
        return []

    logger.info(
        f"Found {len(candidate_paths)} valid real training images in '{input_out_dir}' "
        f"(excluded blind-eval images: {sorted(BLIND_EVAL_BASENAMES)})."
    )
    records: List[Dict[str, str]] = []

    for idx, img_path in enumerate(candidate_paths, 1):
        raw_bgr = load_image(img_path)
        if raw_bgr is None:
            logger.warning(f"[{idx}/{len(candidate_paths)}] Could not load {img_path}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1].lower() or ".jpg"

        rel_input = f"processed/real_pairs/input/{base_name}{ext}"
        rel_target = f"processed/real_pairs/target/{base_name}.png"
        abs_input = os.path.join(dataset_root, rel_input)
        abs_target = os.path.join(dataset_root, rel_target)

        # Generate refined clean target using refine_real_clean_target
        clean_target = refine_real_clean_target(raw_bgr)
        save_image(abs_target, clean_target)

        records.append({
            "source_id": base_name,
            "raw_path": img_path,
            "input_path": abs_input,
            "target_path": abs_target,
            "rel_input": rel_input.replace("\\", "/"),
            "rel_target": rel_target.replace("\\", "/"),
            "csv_input": f"datasets/{rel_input}".replace("\\", "/"),
            "csv_target": f"datasets/{rel_target}".replace("\\", "/"),
        })
        logger.info(
            f"[{idx}/{len(candidate_paths)}] Prepared pair: {rel_input} -> {rel_target} "
            f"(shape={clean_target.shape[1]}x{clean_target.shape[0]})"
        )

    added_train, added_val = update_metadata_with_real_pairs(
        records,
        dataset_root=dataset_root,
        train_ratio=train_ratio,
        train_oversample=train_oversample,
    )
    logger.info(
        f"Metadata updated: +{added_train} rows in train.csv ({int(added_train // max(train_oversample, 1))} unique x{train_oversample} oversample), "
        f"+{added_val} paired rows in val.csv (plus unpaired real_12 in val.csv & real_11 in test.csv)."
    )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare (raw_real_input -> clean_real_target) pairs for U-Net training (excluding real_11 & real_12)"
    )
    parser.add_argument("--real-dir", type=str, default="datasets/real", help="Directory with raw real camera photos")
    parser.add_argument("--dataset-root", type=str, default="datasets", help="Root datasets directory")
    parser.add_argument("--train-ratio", type=float, default=1.0, help="Fraction of real_pairs for train split (default 1.0 = all 28 pairs in train.csv)")
    parser.add_argument("--oversample", type=int, default=3, help="Oversampling factor for real pairs in train.csv")
    args = parser.parse_args()

    prepare_real_pairs(
        real_dir=args.real_dir,
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        train_oversample=args.oversample,
    )


if __name__ == "__main__":
    main()

