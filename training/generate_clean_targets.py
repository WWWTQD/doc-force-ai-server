import os
import re
import sys
import csv
import glob
import argparse
import logging
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.restoration_dataset import (
    load_image,
    save_image,
    unsharp_mask,
    clean_target_normalization,
    infer_single_image_tiled,
)
from training.train_restoration import DocumentRestorationUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_clean_targets")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp")


def generate_master_clean_target(
    raw_gray: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    tile_size: int = 512,
    overlap: int = 128,
    use_nlm: bool = False,
) -> np.ndarray:
    """
    Generate a high-quality Master Clean Target from a raw camera photo.

    Supports Hybrid Mode (Signal Processing + Optional U-Net Prior):
    - When `model` is provided, uses U-Net's tiled shadow-free prediction as a low-frequency
      illumination prior to lift deep cast shadows (`ShadowCast`) before running the 5-step
      Multi-Scale Morphological + Sauvola Soft Matting pipeline, preserving 100% of original
      camera stroke sharpness and Vietnamese diacritics.
    - When `model` is None, runs pure 5-step Signal Processing (`clean_target_normalization`).
    """
    if raw_gray.ndim == 3:
        raw_gray = cv2.cvtColor(raw_gray, cv2.COLOR_BGR2GRAY)

    if model is not None:
        if device is None:
            device = next(model.parameters()).device
        unet_pred = infer_single_image_tiled(
            model, raw_gray, device, tile_size=tile_size, overlap=overlap
        )

        # Estimate smooth low-frequency illumination envelopes from raw photo and U-Net prior
        k_env = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        raw_env = cv2.GaussianBlur(
            cv2.morphologyEx(raw_gray, cv2.MORPH_CLOSE, k_env).astype(np.float32),
            (0, 0),
            sigmaX=21,
            sigmaY=21,
        )
        unet_env = cv2.GaussianBlur(
            cv2.morphologyEx(unet_pred, cv2.MORPH_CLOSE, k_env).astype(np.float32),
            (0, 0),
            sigmaX=21,
            sigmaY=21,
        )

        # Lift deep shadow regions using smooth U-Net illumination gain without blurring stroke edges
        shadow_gain = np.clip(unet_env / np.maximum(raw_env, 15.0), 1.0, 3.2)
        shadow_lifted = np.clip(raw_gray.astype(np.float32) * shadow_gain, 0.0, 255.0).astype(
            np.uint8
        )

        clean_hybrid = clean_target_normalization(shadow_lifted, use_nlm=use_nlm)

        # Suppress any faint shadow boundary transition where U-Net strongly predicts pure background
        clean_hybrid = np.where(
            (unet_pred > 238) & (clean_hybrid > 175), 255, clean_hybrid
        ).astype(np.uint8)
        return clean_hybrid

    return clean_target_normalization(raw_gray, use_nlm=use_nlm)


def _find_best_zoom_patch(img_gray: np.ndarray, patch_ratio: float = 0.28) -> Tuple[int, int, int, int]:
    """
    Locate a text-rich region with high gradient density for the Zoom-in comparison patch.
    Returns (x0, y0, pw, ph).
    """
    h, w = img_gray.shape[:2]
    pw = max(32, min(w, int(w * patch_ratio)))
    ph = max(32, min(h, int(h * patch_ratio)))
    if w <= pw or h <= ph:
        return 0, 0, w, h

    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    integral_grad = cv2.integral(grad_mag)
    step_y = max(1, (h - ph) // 16)
    step_x = max(1, (w - pw) // 16)

    best_score = -1.0
    best_xy = ((w - pw) // 2, (h - ph) // 2)

    for y in range(0, h - ph + 1, step_y):
        for x in range(0, w - pw + 1, step_x):
            score = float(
                integral_grad[y + ph, x + pw]
                - integral_grad[y, x + pw]
                - integral_grad[y + ph, x]
                + integral_grad[y, x]
            )
            if score > best_score:
                best_score = score
                best_xy = (x, y)

    return best_xy[0], best_xy[1], pw, ph


def create_preview_comparison(
    raw_bgr: np.ndarray,
    clean_gray: np.ndarray,
    target_height: int = 560,
) -> np.ndarray:
    """
    Create a 2-column visual comparison sheet:
    [1. Raw Raspberry Pi Photo | 2. Generated Clean Target]
    with a synchronized Zoom-in patch at the bottom to inspect Vietnamese diacritics and stroke edges.
    """
    if raw_bgr.ndim == 2:
        raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2BGR)
    clean_bgr = (
        cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR) if clean_gray.ndim == 2 else clean_gray.copy()
    )

    h, w = clean_gray.shape[:2]
    if raw_bgr.shape[:2] != (h, w):
        raw_bgr = cv2.resize(raw_bgr, (w, h), interpolation=cv2.INTER_AREA)

    zx, zy, zw, zh = _find_best_zoom_patch(clean_gray)
    raw_crop = raw_bgr[zy:zy + zh, zx:zx + zw].copy()
    clean_crop = clean_bgr[zy:zy + zh, zx:zx + zw].copy()

    raw_annotated = raw_bgr.copy()
    clean_annotated = clean_bgr.copy()
    box_thickness = max(2, min(h, w) // 250)
    cv2.rectangle(raw_annotated, (zx, zy), (zx + zw, zy + zh), (0, 140, 255), box_thickness)
    cv2.rectangle(clean_annotated, (zx, zy), (zx + zw, zy + zh), (0, 180, 0), box_thickness)

    scale = target_height / float(max(h, 1))
    disp_w = max(220, int(w * scale))

    disp_raw = cv2.resize(raw_annotated, (disp_w, target_height), interpolation=cv2.INTER_AREA)
    disp_clean = cv2.resize(clean_annotated, (disp_w, target_height), interpolation=cv2.INTER_AREA)

    zoom_h = max(140, int(target_height * 0.36))
    zoom_raw = cv2.resize(raw_crop, (disp_w, zoom_h), interpolation=cv2.INTER_CUBIC)
    zoom_clean = cv2.resize(clean_crop, (disp_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    cv2.rectangle(zoom_raw, (0, 0), (disp_w - 1, zoom_h - 1), (0, 140, 255), 2)
    cv2.rectangle(zoom_clean, (0, 0), (disp_w - 1, zoom_h - 1), (0, 180, 0), 2)

    cv2.putText(
        disp_raw,
        "1. Raw Raspberry Pi Photo",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        disp_clean,
        "2. Generated Clean Target",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 150, 0),
        2,
    )
    cv2.putText(
        zoom_raw,
        "Zoom-in: Raw Stroke & Diacritics",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 140, 255),
        2,
    )
    cv2.putText(
        zoom_clean,
        "Zoom-in: Clean Anti-Aliased Target",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 160, 0),
        2,
    )

    col_left = np.vstack([disp_raw, zoom_raw])
    col_right = np.vstack([disp_clean, zoom_clean])
    divider = np.full((col_left.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([col_left, divider, col_right])


def _get_next_original_id(original_dir: str) -> str:
    """Find next sequential AnhXX identifier in datasets/original/ (e.g. Anh23)."""
    os.makedirs(original_dir, exist_ok=True)
    max_idx = 0
    for fname in os.listdir(original_dir):
        m = re.match(r"^Anh(\d+)", fname, re.IGNORECASE)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    next_idx = max(max_idx + 1, 1)
    return f"Anh{next_idx:02d}"


def export_to_original_dir(
    clean_gray: np.ndarray,
    original_dir: str = "datasets/original",
    source_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Export the approved Clean Target to `datasets/original/AnhXX.png`.
    Returns (assigned_id, exported_path).
    """
    os.makedirs(original_dir, exist_ok=True)
    assigned_id = source_id or _get_next_original_id(original_dir)
    out_path = os.path.join(original_dir, f"{assigned_id}.png")
    save_image(out_path, clean_gray)
    return assigned_id, out_path


def register_pair_in_train_set(
    raw_bgr: np.ndarray,
    clean_gray: np.ndarray,
    base_name: str,
    dataset_root: str = "datasets",
    source_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Automatically save the (raw_photo -> clean_target) pair into the training split:
    - Saves raw photo to `datasets/processed/train/input/RealCamera_<base_name>.png`
    - Saves clean target to `datasets/processed/train/target/<base_name>.png`
    - Appends/updates rows in `datasets/metadata/train.csv` and `datasets/metadata/dataset_manifest.csv`
      without duplicating existing entries.
    """
    safe_base = re.sub(r"[^\w\-]", "_", base_name).strip("_") or "real_doc"
    src_id = source_id or safe_base

    train_in_rel = f"processed/train/input/RealCamera_{safe_base}.png"
    train_tgt_rel = f"processed/train/target/{safe_base}.png"

    train_in_abs = os.path.join(dataset_root, train_in_rel)
    train_tgt_abs = os.path.join(dataset_root, train_tgt_rel)

    save_image(train_in_abs, raw_bgr)
    save_image(train_tgt_abs, clean_gray)

    meta_dir = os.path.join(dataset_root, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # 1. Update train.csv
    train_csv_path = os.path.join(meta_dir, "train.csv")
    csv_in_val = f"datasets/{train_in_rel}".replace("\\", "/")
    csv_tgt_val = f"datasets/{train_tgt_rel}".replace("\\", "/")
    train_fieldnames = ["input", "target", "source_id", "degradation", "blur_level"]

    existing_train_rows = []
    if os.path.exists(train_csv_path):
        with open(train_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                train_fieldnames = list(reader.fieldnames)
            for r in reader:
                if r.get("input", "").strip().replace("\\", "/") != csv_in_val:
                    existing_train_rows.append(r)

    new_train_row = {
        "input": csv_in_val,
        "target": csv_tgt_val,
        "source_id": src_id,
        "degradation": "RealCamera",
        "blur_level": "RealCamera_01",
    }
    existing_train_rows.append({k: new_train_row.get(k, "") for k in train_fieldnames})

    with open(train_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_fieldnames)
        writer.writeheader()
        writer.writerows(existing_train_rows)

    # 2. Update dataset_manifest.csv
    manifest_path = os.path.join(meta_dir, "dataset_manifest.csv")
    manifest_fieldnames = [
        "split",
        "original_id",
        "degradation_type",
        "variant_id",
        "input_path",
        "target_path",
    ]
    existing_manifest_rows = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                manifest_fieldnames = list(reader.fieldnames)
            for r in reader:
                if r.get("input_path", "").strip().replace("\\", "/") != train_in_rel:
                    existing_manifest_rows.append(r)

    new_manifest_row = {
        "split": "train",
        "original_id": src_id,
        "degradation_type": "RealCamera",
        "variant_id": "01",
        "input_path": train_in_rel,
        "target_path": train_tgt_rel,
    }
    existing_manifest_rows.append({k: new_manifest_row.get(k, "") for k in manifest_fieldnames})

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(existing_manifest_rows)

    return train_in_abs, train_tgt_abs


def process_real_directory(
    input_dir: str = "datasets/real",
    model_path: str = "models/restoration_best.pth",
    use_unet: bool = True,
    use_nlm: bool = False,
    export_to_original: bool = False,
    save_to_train: bool = True,
    dataset_root: str = "datasets",
    tile_size: int = 512,
    overlap: int = 128,
) -> List[Dict[str, str]]:
    """
    Scan all raw camera photos in `input_dir`, generate Clean Targets + Preview Comparisons,
    automatically register (raw_photo -> clean_target) pairs into the training set, and
    optionally export sequential AnhXX.png files to `datasets/original/`.
    """
    os.makedirs(input_dir, exist_ok=True)
    clean_targets_dir = os.path.join(input_dir, "clean_targets")
    preview_dir = os.path.join(input_dir, "preview_comparisons")
    os.makedirs(clean_targets_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    # Collect raw image files directly inside input_dir (do not recurse into output subfolders)
    raw_files = []
    for entry in sorted(os.listdir(input_dir)):
        full_p = os.path.join(input_dir, entry)
        if os.path.isfile(full_p) and os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
            raw_files.append(full_p)

    if not raw_files:
        logger.warning(f"No raw images found in '{input_dir}'.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if use_unet and model_path and os.path.exists(model_path):
        try:
            model = DocumentRestorationUNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            logger.info(f"Loaded U-Net prior from '{model_path}' on {device} (Hybrid Mode enabled).")
        except Exception as e:
            logger.warning(f"Could not load U-Net model from '{model_path}' ({e}). Using pure Signal Processing mode.")
            model = None
    else:
        logger.info("Running in Signal Processing Master Ground-Truth mode (5-step Sauvola Soft Matting).")

    results: List[Dict[str, str]] = []
    original_dir = os.path.join(dataset_root, "original")

    for idx, img_path in enumerate(raw_files, 1):
        raw_bgr = load_image(img_path)
        if raw_bgr is None:
            logger.warning(f"[{idx}/{len(raw_files)}] Failed to load image: {img_path}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

        clean_target = generate_master_clean_target(
            raw_gray,
            model=model,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            use_nlm=use_nlm,
        )

        # 1. Save Clean Target to datasets/real/clean_targets/<base_name>.png
        clean_out_path = os.path.join(clean_targets_dir, f"{base_name}.png")
        save_image(clean_out_path, clean_target)

        # 2. Save 2-column comparison + Zoom-in patch to datasets/real/preview_comparisons/
        preview_sheet = create_preview_comparison(raw_bgr, clean_target)
        preview_out_path = os.path.join(preview_dir, f"{base_name}_preview.png")
        save_image(preview_out_path, preview_sheet)

        assigned_id = base_name
        exported_orig_path = ""
        if export_to_original:
            assigned_id, exported_orig_path = export_to_original_dir(
                clean_target, original_dir=original_dir
            )

        train_in_path = ""
        train_tgt_path = ""
        if save_to_train:
            train_in_path, train_tgt_path = register_pair_in_train_set(
                raw_bgr,
                clean_target,
                base_name=base_name,
                dataset_root=dataset_root,
                source_id=assigned_id,
            )

        rec = {
            "raw_path": img_path,
            "clean_target_path": clean_out_path,
            "preview_path": preview_out_path,
            "train_input_path": train_in_path,
            "train_target_path": train_tgt_path,
            "exported_original_path": exported_orig_path,
            "source_id": assigned_id,
        }
        results.append(rec)
        logger.info(
            f"[{idx}/{len(raw_files)}] Generated Clean Target: {clean_out_path} | Preview: {preview_out_path}"
            + (f" | Exported: {exported_orig_path}" if exported_orig_path else "")
            + (f" | Added to train.csv ({assigned_id})" if train_in_path else "")
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Master Ground-Truth Clean Target Generator for Real Camera Photos (datasets/real)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="datasets/real",
        help="Directory containing raw Raspberry Pi camera photos",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Root datasets directory containing processed/ and metadata/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/restoration_best.pth",
        help="Optional U-Net checkpoint for Hybrid Shadow Removal prior",
    )
    parser.add_argument(
        "--no-unet",
        action="store_true",
        help="Disable U-Net prior and use pure 5-step Signal Processing pipeline",
    )
    parser.add_argument(
        "--use-nlm",
        action="store_true",
        help="Use cv2.fastNlMeansDenoising instead of bilateralFilter in Step 1",
    )
    parser.add_argument(
        "--export-to-original",
        action="store_true",
        help="Automatically number next AnhXX.png and copy clean target into datasets/original/",
    )
    parser.add_argument(
        "--no-save-to-train",
        action="store_false",
        dest="save_to_train",
        default=True,
        help="Skip automatic saving of (raw_photo -> clean_target) into train split",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for U-Net prior")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap for U-Net prior")
    args = parser.parse_args()

    process_real_directory(
        input_dir=args.input_dir,
        model_path=args.model,
        use_unet=not args.no_unet,
        use_nlm=args.use_nlm,
        export_to_original=args.export_to_original,
        save_to_train=args.save_to_train,
        dataset_root=args.dataset_root,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
