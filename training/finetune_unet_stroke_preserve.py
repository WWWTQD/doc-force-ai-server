"""
Local Fast Fine-tuning Script for DocumentRestorationUNet v5 -> v5.1 (Stroke-Preserving).

Runs entirely on local CPU without external datasets or Colab:
1. Loads `models/restoration_best.pth` (creates a safety backup `models/restoration_v5_backup.pth` if absent).
2. Freezes Encoder, Bottleneck, and upper Decoder blocks (`enc1`..`up1`); unfreezes only the final
   Decoder block (`dec1`) and `out_conv` for rapid CPU adaptation.
3. Extracts ~80 patches (256x256) directly from real images in `datasets/real/`
   (prioritizing `real_11.jpg`, `real_24.jpg`, `real_36.jpg` plus diverse real camera pages),
   pre-cleaned with `clean_handwriting_and_book_gutter()` and paired with crisp clean targets.
4. Trains for 3 epochs on CPU with L1 + Multi-Scale Sobel Edge + Stroke Preservation Loss,
   logs `Epoch, Train Loss, Edge Loss, PSNR, SSIM` to `models/unet_v5_1_finetune_log.json`,
   and updates `models/restoration_best.pth` when validation quality improves.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKER_DIR = os.path.join(PROJECT_ROOT, "worker")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

from training.prepare_real_pairs import refine_real_clean_target
from training.restoration_dataset import load_image
from training.train_restoration import (
    AsymmetricStrokeDarkeningLoss,
    BleedThroughSuppressionLoss,
    DocumentRestorationUNet,
    SSIMLoss,
    SobelLoss,
    calculate_psnr,
)
from worker.preprocessor import ImagePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_unet_v5_1")

ANCHOR_IMAGES = ("real_11.jpg", "real_24.jpg", "real_36.jpg")
SMALL_LOWRES_ANCHORS = (
    "real_31.jpg",
    "real_21.jpg",
    "real_37.png",
    "real_38.jfif",
    "real_28.webp",
    "real_43.jpg",
    "real_60.jpg",
    "real_63.jfif",
)
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".jfif")


def build_stroke_preserving_target(clean_bgr: np.ndarray, prep: ImagePreprocessor) -> np.ndarray:
    """
    Create a stroke-preserving clean target from a real document image:
    - Preserves 100% of thin/faded strokes on low-resolution & yellowed scans (`500x727`, etc.)
      by avoiding bilateral smoothing before contrast estimation and using local stroke protection.
    - Suppresses yellow paper, shadows, and faint diffuse bleed-through cleanly to pure white 255.
    """
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY) if clean_bgr.ndim == 3 else clean_bgr
    h, w = gray.shape[:2]
    soft_norm, _ = prep.soft_normalize_luma(gray, mode="dilate")
    stroke_conf, local_drop, _ = prep.compute_stroke_protection(soft_norm, h, w)

    # Also protect genuine printed strokes visible in soft_norm with moderate drop on small scans
    if max(h, w) < 900:
        small_stroke = np.clip((local_drop - 14.0) / 18.0, 0.0, 1.0) * np.clip((236.0 - soft_norm) / 20.0, 0.0, 1.0)
        stroke_conf = np.maximum(stroke_conf, small_stroke)

    s_f = soft_norm / 255.0
    bg_white = np.where(
        soft_norm >= 216.0,
        np.clip((soft_norm - 216.0) / 20.0 * 0.16 + 0.84, 0.0, 1.0),
        s_f,
    )
    # Clean faint diffuse background/bleed-through where stroke_conf is near zero
    bg_white = np.where((stroke_conf < 0.15) & (soft_norm >= 205.0), 1.0, bg_white)
    # Crisp dark foreground strokes
    stroke_dark = np.power(np.clip(s_f / 0.96, 0.0, 1.0), 2.15)
    stroke_dark = np.minimum(stroke_dark, np.clip((soft_norm - 0.85 * local_drop) / 255.0, 0.0, 1.0))

    target_f = (1.0 - stroke_conf) * bg_white + stroke_conf * stroke_dark
    return np.clip(target_f * 255.0, 0.0, 255.0).astype(np.uint8)


def _extract_patches_from_image(
    img_bgr: np.ndarray,
    preprocessor: ImagePreprocessor,
    patch_size: int = 256,
    max_patches: int = 4,
    rng: random.Random | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract text-rich 256x256 (degraded_gray, clean_target_gray) patches from a real document image.
    For small/low-resolution images (max(h, w) < 900), includes upscaled 1.5x-2.0x views matching
    U-Net v5.1 inference scaling so the decoder learns sharp stroke preservation on low-res scans.
    """
    if rng is None:
        rng = random.Random(42)

    clean_bgr = preprocessor.clean_handwriting_and_book_gutter(img_bgr)
    clean_gray_full = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    tgt_full = build_stroke_preserving_target(clean_bgr, preprocessor)

    h0, w0 = clean_gray_full.shape[:2]
    max_side = max(h0, w0)
    scales: List[Tuple[int, int]] = []
    if max_side < 900:
        up_sc = min(2.0, max(1.5, 960.0 / float(max(1, max_side))))
        scales.append((int(round(w0 * up_sc)), int(round(h0 * up_sc))))
        scales.append((w0, h0))
    elif max_side > 1300:
        for target_side in (1280, 768):
            sc = float(target_side) / float(max_side)
            scales.append((max(patch_size, int(round(w0 * sc))), max(patch_size, int(round(h0 * sc)))))
    else:
        scales.append((w0, h0))
        sc = 768.0 / float(max_side)
        scales.append((max(patch_size, int(round(w0 * sc))), max(patch_size, int(round(h0 * sc)))))

    candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for nw, nh in scales:
        if (nw, nh) == (w0, h0):
            deg_s = clean_gray_full
            tgt_s = tgt_full
        elif nw > w0:
            deg_s = cv2.resize(clean_gray_full, (nw, nh), interpolation=cv2.INTER_CUBIC)
            tgt_s = cv2.resize(tgt_full, (nw, nh), interpolation=cv2.INTER_CUBIC)
        else:
            deg_s = cv2.resize(clean_gray_full, (nw, nh), interpolation=cv2.INTER_AREA)
            tgt_s = cv2.resize(tgt_full, (nw, nh), interpolation=cv2.INTER_AREA)

        hs, ws = deg_s.shape[:2]
        if hs < patch_size or ws < patch_size:
            pad_h = max(0, patch_size - hs)
            pad_w = max(0, patch_size - ws)
            deg_s = cv2.copyMakeBorder(deg_s, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            tgt_s = cv2.copyMakeBorder(tgt_s, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            hs, ws = deg_s.shape[:2]

        # Sample candidate grid + random crops and rank by stroke content & edge richness
        n_trials = 18
        for idx_t in range(n_trials):
            if idx_t < 4 and hs > patch_size and ws > patch_size:
                y = int((hs - patch_size) * (0.2 + 0.2 * idx_t))
                x = int((ws - patch_size) * (0.15 + 0.2 * (idx_t % 3)))
            else:
                y = rng.randint(0, max(0, hs - patch_size))
                x = rng.randint(0, max(0, ws - patch_size))

            d_patch = deg_s[y:y + patch_size, x:x + patch_size]
            t_patch = tgt_s[y:y + patch_size, x:x + patch_size]
            ink_ratio = float(np.mean(t_patch < 200))
            if ink_ratio < 0.02 or ink_ratio > 0.50:
                continue

            lap_score = float(np.mean(np.abs(cv2.Laplacian(t_patch, cv2.CV_32F))))
            score = lap_score * min(ink_ratio, 0.25)
            candidates.append((score, d_patch, t_patch))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: List[Tuple[np.ndarray, np.ndarray]] = []
    step = max(1, len(candidates) // max(1, max_patches))
    for i in range(0, len(candidates), step):
        selected.append((candidates[i][1], candidates[i][2]))
        if len(selected) >= max_patches:
            break
    return selected


def _extract_paired_lowres_patches(
    dataset_root: str,
    patch_size: int = 256,
    max_pairs: int = 12,
    rng: random.Random | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract exact ground-truth (degraded, clean) patches from `datasets/processed/destroyer_pairs`,
    including low-resolution downscale+upscale simulations and yellowed-paper pairs.
    """
    if rng is None:
        rng = random.Random(42)
    in_dir = os.path.join(dataset_root, "processed", "destroyer_pairs", "input")
    tgt_dir = os.path.join(dataset_root, "processed", "destroyer_pairs", "target")
    if not (os.path.isdir(in_dir) and os.path.isdir(tgt_dir)):
        return []

    in_files = sorted(f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS)
    if not in_files:
        return []
    rng.shuffle(in_files)

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for fname in in_files:
        if len(pairs) >= max_pairs:
            break
        in_p = os.path.join(in_dir, fname)
        # Match target filename prefix before '_var_' or exact name
        stem = os.path.splitext(fname)[0]
        tgt_candidates = [fname]
        if "_var_" in stem:
            base_id = stem.split("_var_", 1)[0]
            if base_id.startswith("deg_"):
                base_id = base_id[4:]
            for ext in (".png", ".jpg", ".jpeg"):
                tgt_candidates.append(base_id + ext)
                tgt_candidates.append("clean_" + base_id.replace("clean_", "") + ext)
        tgt_p = None
        for tc in tgt_candidates:
            cand = os.path.join(tgt_dir, tc)
            if os.path.exists(cand):
                tgt_p = cand
                break
        if tgt_p is None:
            continue

        deg_bgr = load_image(in_p)
        clean_bgr = load_image(tgt_p)
        if deg_bgr is None or clean_bgr is None:
            continue
        deg_g = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2GRAY)
        clean_g = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        if deg_g.shape != clean_g.shape:
            deg_g = cv2.resize(deg_g, (clean_g.shape[1], clean_g.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Simulate low-res vintage scan + bicubic upscale on half of paired samples
        h0, w0 = deg_g.shape[:2]
        if len(pairs) % 2 == 0 and max(h0, w0) >= 800:
            sc_down = rng.uniform(0.38, 0.55)
            small_d = cv2.resize(deg_g, (max(256, int(w0 * sc_down)), max(256, int(h0 * sc_down))), interpolation=cv2.INTER_AREA)
            deg_g = cv2.resize(small_d, (w0, h0), interpolation=cv2.INTER_CUBIC)

        hs, ws = deg_g.shape[:2]
        if hs < patch_size or ws < patch_size:
            continue
        for _ in range(10):
            y = rng.randint(0, hs - patch_size)
            x = rng.randint(0, ws - patch_size)
            dp = deg_g[y:y + patch_size, x:x + patch_size]
            tp = clean_g[y:y + patch_size, x:x + patch_size]
            ink_ratio = float(np.mean(tp < 200))
            if 0.03 <= ink_ratio <= 0.45:
                pairs.append((dp, tp))
                break
    return pairs


def build_local_real_patches(
    real_dir: str,
    total_patches: int = 120,
    patch_size: int = 256,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    """
    Build train/val 256x256 patch datasets from `datasets/real/` (explicitly including small low-res
    & yellowed pages such as 500x727 `real_31.jpg`) plus paired clean-GT patches from `destroyer_pairs`.
    """
    rng = random.Random(seed)
    prep = ImagePreprocessor()

    all_files = sorted(
        f for f in os.listdir(real_dir)
        if os.path.isfile(os.path.join(real_dir, f)) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )
    priority_names = list(ANCHOR_IMAGES) + [f for f in SMALL_LOWRES_ANCHORS if f not in ANCHOR_IMAGES]
    priority_paths = [os.path.join(real_dir, f) for f in priority_names if os.path.exists(os.path.join(real_dir, f))]
    other_files = [f for f in all_files if f not in priority_names]
    rng.shuffle(other_files)
    other_paths = [os.path.join(real_dir, f) for f in other_files]

    train_patches: List[Tuple[np.ndarray, np.ndarray]] = []
    val_patches: List[Tuple[np.ndarray, np.ndarray]] = []
    sources_used: Dict[str, int] = {}

    # 1. Anchor & small low-res images (including 500x727 and heavily yellowed pages)
    for p in priority_paths:
        img = load_image(p)
        if img is None:
            continue
        patches = _extract_patches_from_image(img, prep, patch_size=patch_size, max_patches=5, rng=rng)
        if len(patches) >= 3:
            val_patches.extend(patches[:1])
            train_patches.extend(patches[1:])
        else:
            train_patches.extend(patches)
        sources_used[os.path.basename(p)] = len(patches)

    # 2. Exact clean-GT paired low-res/yellowed patches from datasets/processed/destroyer_pairs
    dataset_root = os.path.dirname(os.path.abspath(real_dir))
    gt_pairs = _extract_paired_lowres_patches(dataset_root, patch_size=patch_size, max_pairs=24, rng=rng)
    if gt_pairs:
        n_gt_val = max(2, len(gt_pairs) // 5)
        val_patches.extend(gt_pairs[:n_gt_val])
        train_patches.extend(gt_pairs[n_gt_val:])
        sources_used["destroyer_pairs_gt"] = len(gt_pairs)

    # 3. Fill remaining quota from diverse real images
    val_quota = max(18, total_patches // 5)
    for p in other_paths:
        if len(train_patches) + len(val_patches) >= total_patches:
            break
        img = load_image(p)
        if img is None:
            continue
        patches = _extract_patches_from_image(img, prep, patch_size=patch_size, max_patches=3, rng=rng)
        if not patches:
            continue
        for idx_p, pair in enumerate(patches):
            if len(train_patches) + len(val_patches) >= total_patches:
                break
            if len(val_patches) < val_quota and idx_p == 0:
                val_patches.append(pair)
            else:
                train_patches.append(pair)
        sources_used[os.path.basename(p)] = len(patches)

    return train_patches, val_patches, sources_used


def freeze_encoder_and_bottleneck(model: DocumentRestorationUNet) -> Tuple[int, int]:
    """
    Freeze Encoder, Bottleneck, and upper Decoder layers (`enc1`..`up1`).
    Unfreeze only the final Decoder block (`dec1`) and `out_conv`.
    """
    for param in model.parameters():
        param.requires_grad = False

    for module in (model.dec1, model.out_conv):
        for param in module.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _compute_frozen_features(
    model: DocumentRestorationUNet,
    x_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run frozen Encoder -> Bottleneck -> up3/dec3/up2/dec2/up1 in `torch.no_grad()`,
    returning `(c1, base_logit)` ready for `dec1` + `out_conv`.
    Mathematically identical to `DocumentRestorationUNet.forward` up to `dec1`.
    """
    with torch.no_grad():
        x_safe = torch.nan_to_num(x_batch.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) * 0.996 + 0.002
        x_med = model._median_filter_3x3(x_safe)
        x_norm_31 = model._get_norm_robust(x_med, x_safe, 31)
        x_norm_101 = model._get_norm_robust(x_med, x_safe, 101)
        x_in = torch.cat([x_safe, x_norm_31, x_norm_101], dim=1)

        e1 = model.enc1(x_in)
        d1 = model.down1(e1)
        e2 = model.enc2(d1)
        d2 = model.down2(e2)
        e3 = model.enc3(d2)
        d3 = model.down3(e3)

        b = model.bottleneck(d3)

        u3 = model.up3(b)
        e3_att = model.att3(g=u3, x=e3)
        d_out3 = model.dec3(torch.cat([u3, e3_att], dim=1))

        u2 = model.up2(d_out3)
        e2_att = model.att2(g=u2, x=e2)
        d_out2 = model.dec2(torch.cat([u2, e2_att], dim=1))

        u1 = model.up1(d_out2)
        c1 = torch.cat([u1, e1], dim=1)

        base_clamped = torch.clamp(x_norm_31.float(), 1e-4, 1.0 - 1e-4)
        base_logit = torch.logit(base_clamped)
    return c1, base_logit


def _cache_batches_to_disk(
    model: DocumentRestorationUNet,
    patches: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    cache_dir: str,
    prefix: str,
) -> List[str]:
    model.eval()
    paths: List[str] = []
    for b_idx in range(0, len(patches), batch_size):
        chunk = patches[b_idx:b_idx + batch_size]
        x_np = np.stack([p[0].astype(np.float32) / 255.0 for p in chunk], axis=0)[:, None, :, :]
        y_np = np.stack([p[1].astype(np.float32) / 255.0 for p in chunk], axis=0)[:, None, :, :]
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        c1, base_logit = _compute_frozen_features(model, x_t)
        fpath = os.path.join(cache_dir, f"{prefix}_batch_{len(paths):03d}.pt")
        torch.save({"c1": c1, "base_logit": base_logit, "target": y_t}, fpath)
        paths.append(fpath)
    return paths


def _predict_from_cached(
    model: DocumentRestorationUNet,
    c1: torch.Tensor,
    base_logit: torch.Tensor,
) -> torch.Tensor:
    d_out1 = model.dec1(c1)
    delta = model.out_conv(d_out1).float()
    y_pred = torch.sigmoid(base_logit + delta)
    return torch.nan_to_num(y_pred.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _evaluate_cached(
    model: DocumentRestorationUNet,
    batch_files: List[str],
    sobel_fn: SobelLoss,
    ssim_loss_fn: SSIMLoss,
    stroke_fn: AsymmetricStrokeDarkeningLoss,
    bleed_fn: BleedThroughSuppressionLoss,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_edge = 0.0
    total_stroke = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_items = 0

    with torch.inference_mode():
        for bf in batch_files:
            data = torch.load(bf, weights_only=True)
            c1 = data["c1"]
            base_logit = data["base_logit"]
            target = data["target"]
            bs = target.shape[0]

            pred = _predict_from_cached(model, c1, base_logit)
            l1 = F.l1_loss(pred, target)
            edge = sobel_fn(pred, target)
            stroke = stroke_fn(pred, target)
            bleed = bleed_fn(pred, target)
            loss = 1.0 * l1 + 0.8 * edge + 1.8 * stroke + 0.20 * bleed

            psnr_val = calculate_psnr(pred, target)
            ssim_val = 1.0 - float(ssim_loss_fn(pred, target).item())

            total_loss += float(loss.item()) * bs
            total_edge += float(edge.item()) * bs
            total_stroke += float(stroke.item()) * bs
            total_psnr += float(psnr_val) * bs
            total_ssim += float(ssim_val) * bs
            n_items += bs

    denom = max(1, n_items)
    return {
        "loss": total_loss / denom,
        "edge_loss": total_edge / denom,
        "stroke_loss": total_stroke / denom,
        "psnr": total_psnr / denom,
        "ssim": total_ssim / denom,
    }


def finetune(
    model_path: str = "models/restoration_best.pth",
    real_dir: str = "datasets/real",
    log_path: str = "models/unet_v5_1_finetune_log.json",
    epochs: int = 4,
    batch_size: int = 8,
    total_patches: int = 120,
    patch_size: int = 256,
    lr: float = 1.8e-4,
    num_threads: int = 4,
) -> dict:
    torch.set_num_threads(num_threads)
    device = torch.device("cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Safety backup of original v5 weights before any modification
    backup_path = os.path.join(os.path.dirname(model_path), "restoration_v5_backup.pth")
    if not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        logger.info("Saved safety backup of v5 weights -> %s", backup_path)

    logger.info("Loading DocumentRestorationUNet v5 from: %s", model_path)
    model = DocumentRestorationUNet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    trainable_params, total_params = freeze_encoder_and_bottleneck(model)
    logger.info(
        "Frozen Encoder/Bottleneck/Upper-Decoder. Trainable params (dec1 + out_conv): %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / max(1, total_params),
    )

    t_start = time.perf_counter()
    train_patches, val_patches, sources_used = build_local_real_patches(
        real_dir=real_dir,
        total_patches=total_patches,
        patch_size=patch_size,
        seed=42,
    )
    logger.info(
        "Extracted %d train patches + %d val patches (%dx%d) from %d images in %s",
        len(train_patches),
        len(val_patches),
        patch_size,
        patch_size,
        len(sources_used),
        real_dir,
    )

    sobel_fn = SobelLoss().to(device)
    ssim_loss_fn = SSIMLoss().to(device)
    stroke_fn = AsymmetricStrokeDarkeningLoss(target_thresh=0.84).to(device)
    bleed_fn = BleedThroughSuppressionLoss(target_thresh=0.95, pred_floor=0.92).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    epoch_logs: List[dict] = []
    with tempfile.TemporaryDirectory() as cache_dir:
        logger.info("Caching frozen Encoder/Bottleneck/up1 features to temp disk...")
        t_cache = time.perf_counter()
        train_files = _cache_batches_to_disk(model, train_patches, batch_size, cache_dir, "train")
        val_files = _cache_batches_to_disk(model, val_patches, batch_size, cache_dir, "val")
        logger.info("Frozen feature cache ready in %.1fs.", time.perf_counter() - t_cache)

        baseline_val = _evaluate_cached(model, val_files, sobel_fn, ssim_loss_fn, stroke_fn, bleed_fn)
        logger.info(
            "Baseline v5 (Epoch 0) -> Val Loss: %.4f | Edge: %.4f | Stroke: %.4f | PSNR: %.2f dB | SSIM: %.4f",
            baseline_val["loss"],
            baseline_val["edge_loss"],
            baseline_val["stroke_loss"],
            baseline_val["psnr"],
            baseline_val["ssim"],
        )

        best_val_loss = baseline_val["loss"]
        best_val_psnr = baseline_val["psnr"]
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

        rng = random.Random(42)
        for epoch in range(1, epochs + 1):
            t_ep = time.perf_counter()
            model.dec1.train()
            model.out_conv.train()

            shuffled_files = list(train_files)
            rng.shuffle(shuffled_files)

            ep_loss = 0.0
            ep_edge = 0.0
            ep_items = 0

            for bf in shuffled_files:
                data = torch.load(bf, weights_only=True)
                c1 = data["c1"]
                base_logit = data["base_logit"]
                target = data["target"]
                bs = target.shape[0]

                optimizer.zero_grad()
                pred = _predict_from_cached(model, c1, base_logit)
                l1 = F.l1_loss(pred, target)
                edge = sobel_fn(pred, target)
                stroke = stroke_fn(pred, target)
                bleed = bleed_fn(pred, target)
                loss = 1.0 * l1 + 0.8 * edge + 1.8 * stroke + 0.20 * bleed
                loss.backward()
                optimizer.step()

                ep_loss += float(loss.item()) * bs
                ep_edge += float(edge.item()) * bs
                ep_items += bs

            scheduler.step()
            train_loss = ep_loss / max(1, ep_items)
            train_edge = ep_edge / max(1, ep_items)

            val_metrics = _evaluate_cached(model, val_files, sobel_fn, ssim_loss_fn, stroke_fn, bleed_fn)
            ep_sec = time.perf_counter() - t_ep

            rec = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "edge_loss": round(train_edge, 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_edge_loss": round(val_metrics["edge_loss"], 6),
                "val_stroke_loss": round(val_metrics["stroke_loss"], 6),
                "psnr": round(val_metrics["psnr"], 2),
                "ssim": round(val_metrics["ssim"], 4),
                "lr": round(scheduler.get_last_lr()[0], 7),
                "seconds": round(ep_sec, 2),
            }
            epoch_logs.append(rec)
            logger.info(
                "Epoch [%d/%d] Train Loss: %.4f | Edge: %.4f | Val Stroke: %.4f | Val PSNR: %.2f dB | SSIM: %.4f | %.1fs",
                epoch,
                epochs,
                rec["train_loss"],
                rec["edge_loss"],
                rec["val_stroke_loss"],
                rec["psnr"],
                rec["ssim"],
                ep_sec,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_psnr = val_metrics["psnr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    total_sec = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    logger.info("Updated weights saved -> %s (best Val PSNR: %.2f dB, total %.1fs)", model_path, best_val_psnr, total_sec)

    summary = {
        "model_version": "DocumentRestorationUNet-v5.1-StrokePreserving",
        "device": str(device),
        "num_threads": num_threads,
        "frozen_modules": ["enc1", "down1", "enc2", "down2", "enc3", "down3", "bottleneck", "up3", "att3", "dec3", "up2", "att2", "dec2", "up1"],
        "unfrozen_modules": ["dec1", "out_conv"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "patch_size": patch_size,
        "train_patches": len(train_patches),
        "val_patches": len(val_patches),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "total_seconds": round(total_sec, 2),
        "baseline_v5": {
            "val_loss": round(baseline_val["loss"], 6),
            "edge_loss": round(baseline_val["edge_loss"], 6),
            "psnr": round(baseline_val["psnr"], 2),
            "ssim": round(baseline_val["ssim"], 4),
        },
        "best_val_psnr": round(best_val_psnr, 2),
        "sources_used": sources_used,
        "log": epoch_logs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved fine-tune log -> %s", log_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fast local CPU fine-tuning for U-Net v5.1")
    parser.add_argument("--model", default="models/restoration_best.pth")
    parser.add_argument("--real-dir", default="datasets/real")
    parser.add_argument("--log-path", default="models/unet_v5_1_finetune_log.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-patches", type=int, default=120)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    finetune(
        model_path=args.model,
        real_dir=args.real_dir,
        log_path=args.log_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        total_patches=args.total_patches,
        patch_size=args.patch_size,
        lr=args.lr,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()
