import os
import csv
import logging
import random
from typing import Tuple, List, Optional, Dict
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger("restoration_dataset")


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Safely load an image supporting unicode paths and all common formats (.png, .jpg, .jfif, .webp).
    Returns BGR numpy array or None on failure.
    """
    if not os.path.exists(path):
        return None
    try:
        pil_img = Image.open(path).convert("RGB")
        img_rgb = np.array(pil_img)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None


def save_image(path: str, img: np.ndarray) -> bool:
    """
    Save an image safely supporting unicode paths.
    """
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext if ext else ".png", img)
    if result:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf)
        return True
    return False


def unsharp_mask(img: np.ndarray, amount: float = 0.8, radius: float = 1.0) -> np.ndarray:
    """
    Apply Unsharp Masking filter to sharpen text edges and contrast.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def resize_with_padding(
    img: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Resize image keeping aspect ratio and pad with zeros to reach target_size.

    Returns:
        canvas: padded image of shape (target_h, target_w, 3)
        pad_info: (pad_top, pad_left, new_h, new_w) for later unpadding
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, (pad_top, pad_left, new_h, new_w)


def remove_padding(
    img: np.ndarray,
    pad_info: Tuple[int, int, int, int],
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Remove padding and resize back to original dimensions.

    Args:
        img: padded image of shape (target_h, target_w, 3)
        pad_info: (pad_top, pad_left, new_h, new_w) from resize_with_padding
        original_size: (orig_h, orig_w) to resize back to
    """
    pad_top, pad_left, new_h, new_w = pad_info
    cropped = img[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    orig_h, orig_w = original_size
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return restored


def pad_if_needed(img: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray:
    """Pad image using BORDER_REFLECT if any dimension is smaller than patch_size."""
    h, w = img.shape[:2]
    patch_h, patch_w = patch_size
    pad_h = max(0, patch_h - h)
    pad_w = max(0, patch_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    return img


def crop_patch(
    deg_img: np.ndarray,
    clean_img: np.ndarray,
    patch_size: Tuple[int, int] = (512, 512),
    is_train: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matching patch from degraded and clean images.
    Uses RandomCrop when is_train=True, CenterCrop when is_train=False.
    Pads with BORDER_REFLECT if image is smaller than patch_size in either dimension.
    """
    deg_pad = pad_if_needed(deg_img, patch_size)
    clean_pad = pad_if_needed(clean_img, patch_size)

    h, w = deg_pad.shape[:2]
    patch_h, patch_w = patch_size

    if is_train:
        max_top = max(0, h - patch_h)
        max_left = max(0, w - patch_w)
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
    else:
        top = max(0, (h - patch_h) // 2)
        left = max(0, (w - patch_w) // 2)

    deg_crop = deg_pad[top:top + patch_h, left:left + patch_w]
    clean_crop = clean_pad[top:top + patch_h, left:left + patch_w]
    return deg_crop, clean_crop


GEOMETRIC_DISTORTION_KEYWORDS = (
    "bookbinding",
    "perspective",
    "opticaldistortion",
    "centercrop",
    "bindingsandfasteners",
)


def _sauvola_threshold_integral(
    img_f: np.ndarray,
    window_size: int = 15,
    k: float = 0.15,
    r: float = 128.0,
) -> np.ndarray:
    """
    Compute local Sauvola threshold T(x,y) = mu(x,y) * [1 + k * (sigma(x,y) / R - 1)]
    in O(1) time per pixel using OpenCV integral images.
    """
    pad = window_size // 2
    padded = cv2.copyMakeBorder(img_f, pad, pad, pad, pad, cv2.BORDER_REFLECT).astype(np.float64)
    s, s2 = cv2.integral2(padded)
    w = window_size
    area = float(w * w)

    win_sum = s[w:, w:] - s[:-w, w:] - s[w:, :-w] + s[:-w, :-w]
    win_sqsum = s2[w:, w:] - s2[:-w, w:] - s2[w:, :-w] + s2[:-w, :-w]

    mu = win_sum / area
    var = np.maximum(win_sqsum / area - (mu * mu), 0.0)
    sigma = np.sqrt(var)

    thresh = mu * (1.0 + k * ((sigma / r) - 1.0))
    return thresh.astype(np.float32)


def clean_target_normalization(
    clean_gray: np.ndarray,
    use_nlm: bool = False,
    gamma: float = 0.75,
) -> np.ndarray:
    """
    Master Ground-Truth Generator: 5-step Soft Anti-Aliased Document Mastering
    ("Multi-Scale Morphological + Sauvola Soft Matting").

    1. Edge-Preserving Pre-Denoising (Bilateral Filter or Fast Non-Local Means).
    2. Multi-Scale Morphological Background Estimation (21x21 & 45x45 elliptical closing + Gaussian sigma=15).
    3. Multi-Scale Sauvola Soft Alpha Matting (O(1) integral image with w=15,k=0.15 & w=31,k=0.20 + Soft Ramp).
    4. Topology-Aware Micro-Speckle Removal (removes isolated dust <= 2px with min_intensity > 140).
    5. Foreground Unsharp Boost (sharpens only stroke pixels < 245).
    """
    if clean_gray is None or clean_gray.size == 0:
        return clean_gray
    if clean_gray.ndim == 3:
        clean_gray = cv2.cvtColor(clean_gray, cv2.COLOR_BGR2GRAY)

    # Step 1: Edge-Preserving Pre-Denoising
    if use_nlm:
        denoised = cv2.fastNlMeansDenoising(
            clean_gray, None, h=6, templateWindowSize=7, searchWindowSize=21
        )
    else:
        denoised = cv2.bilateralFilter(clean_gray, d=5, sigmaColor=25, sigmaSpace=25)

    # Step 2: Multi-Scale Morphological Background Estimation
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    bg_small = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_small)
    bg_large = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_large)
    bg = np.maximum(bg_small, bg_large)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=15, sigmaY=15)

    norm = np.clip(
        (denoised.astype(np.float32) / np.maximum(bg.astype(np.float32), 15.0)) * 255.0,
        0.0,
        255.0,
    )

    # Step 3: Multi-Scale Sauvola Soft Alpha Matting
    t_small = _sauvola_threshold_integral(norm, window_size=15, k=0.15, r=128.0)
    t_med = _sauvola_threshold_integral(norm, window_size=31, k=0.20, r=128.0)
    t_sauvola = np.maximum(t_small, t_med)

    diff = norm - t_sauvola
    bg_mask = (diff > 6.0) | (norm > 225.0)
    fg_mask = ~bg_mask

    matted_f = np.full_like(norm, 255.0, dtype=np.float32)
    if np.any(fg_mask):
        fg_vals = norm[fg_mask]
        black_floor = float(np.clip(np.percentile(fg_vals, 18), 20.0, 145.0))
        upper_bound = np.minimum(t_sauvola + 6.0, 225.0)
        span = np.maximum(upper_bound - black_floor, 20.0)

        # Compute foreground opacity alpha in [0, 1], boost with gamma < 1.0, and map to [0, 255]
        alpha = np.clip((upper_bound - norm) / span, 0.0, 1.0)
        alpha_boosted = np.power(alpha, gamma)
        matted_f = np.where(bg_mask, 255.0, 255.0 * (1.0 - alpha_boosted))

    matted = np.clip(matted_f, 0.0, 255.0).astype(np.uint8)

    # Step 4: Topology-Aware Micro-Speckle Removal
    dark_mask = (matted < 200).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area <= 2:
            x0 = stats[lbl, cv2.CC_STAT_LEFT]
            y0 = stats[lbl, cv2.CC_STAT_TOP]
            w0 = stats[lbl, cv2.CC_STAT_WIDTH]
            h0 = stats[lbl, cv2.CC_STAT_HEIGHT]
            roi_labels = labels[y0:y0 + h0, x0:x0 + w0]
            roi_img = matted[y0:y0 + h0, x0:x0 + w0]
            comp_mask = roi_labels == lbl
            if int(np.min(roi_img[comp_mask])) > 140:
                roi_img[comp_mask] = 255

    # Step 5: Foreground Unsharp Boost (only on stroke pixels < 245)
    sharpened = unsharp_mask(matted, amount=0.65, radius=1.0)
    out = np.where(matted < 245, sharpened, 255).astype(np.uint8)
    return out


# Alias for backward compatibility across training and evaluation
normalize_clean_target = clean_target_normalization


def apply_hard_augmentation(
    deg_patch: np.ndarray,
    clean_patch: np.ndarray,
    is_real_pair: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    On-the-fly hard augmentation for training patches:
    - 50% probability: synchronous HorizontalFlip on both degraded and clean patches.
    - For synthetic/standard samples (`is_real_pair=False`), 30% probability: synthetic local shadow
      patch or mild Gaussian blur on degraded patch.
    - For `real_pairs` (`is_real_pair=True`), preserves authentic camera degradation (only horizontal flip).
    """
    # 1. 50% chance: synchronous HorizontalFlip
    if random.random() < 0.5:
        deg_patch = cv2.flip(deg_patch, 1)
        clean_patch = cv2.flip(clean_patch, 1)

    if is_real_pair:
        return deg_patch, clean_patch

    # 2. 30% chance: mild Gaussian Blur and/or Synthetic Local Shadow on deg_patch
    if random.random() < 0.3:
        aug_mode = random.choice(["blur", "shadow", "both"])
        deg_f = deg_patch.astype(np.float32)

        if aug_mode in ("blur", "both"):
            ksize = random.choice([3, 5])
            sigma = random.uniform(0.6, 1.3)
            deg_f = cv2.GaussianBlur(deg_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        if aug_mode in ("shadow", "both"):
            h, w = deg_f.shape[:2]
            # Create a smooth local shadow mask darkening a random region by factor [0.45, 0.75]
            shadow_mask = np.ones((h, w), dtype=np.float32)
            cx = random.randint(int(w * 0.2), int(w * 0.8))
            cy = random.randint(int(h * 0.2), int(h * 0.8))
            rx = random.randint(int(w * 0.25), int(w * 0.6))
            ry = random.randint(int(h * 0.25), int(h * 0.6))
            angle = random.uniform(0, 180)
            darkness = random.uniform(0.45, 0.75)
            cv2.ellipse(shadow_mask, (cx, cy), (rx, ry), angle, 0, 360, darkness, -1)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=25, sigmaY=25)
            deg_f = deg_f * shadow_mask

        deg_patch = np.clip(deg_f, 0, 255).astype(np.uint8)

    return deg_patch, clean_patch


def infer_single_image_tiled(
    model: torch.nn.Module,
    img_gray: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
    refine_glyphs: bool = False,
) -> np.ndarray:
    """
    Sliding Window Tiling Inference with 2D Gaussian feather blending.
    Runs 100% pure neural network forward passes through DocumentRestorationUNet.
    refine_glyphs: parameter kept for compatibility; no external morphological/heuristic
    filter or background clipping is performed (100% pure neural network output).
    """
    h_orig, w_orig = img_gray.shape[:2]

    # On CPU, if image is ultra-high resolution (>1800px, e.g. 9MP-12MP camera raw),
    # run tiled U-Net at 1600px working scale and upsample back
    if device.type == "cpu" and max(h_orig, w_orig) > 1800 and tile_size <= 512:
        scale = 1600.0 / float(max(h_orig, w_orig))
        w_work = max(64, int(round(w_orig * scale)))
        h_work = max(64, int(round(h_orig * scale)))
        img_work = cv2.resize(img_gray, (w_work, h_work), interpolation=cv2.INTER_AREA)
        pred_work = infer_single_image_tiled(
            model, img_work, device, tile_size=tile_size, overlap=min(overlap, 64), refine_glyphs=refine_glyphs
        )
        return cv2.resize(pred_work, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # Small image fast path: pad to multiple of 8 and run single forward pass
    if h_orig <= tile_size and w_orig <= tile_size:
        pad_h = (8 - (h_orig % 8)) % 8
        pad_w = (8 - (w_orig % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img_gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            img_pad = img_gray
        tensor_in = torch.from_numpy(img_pad).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            pred = model(tensor_in)
        out = (torch.clamp(pred, 0.0, 1.0).squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        return out[:h_orig, :w_orig]

    # Pad any dimension smaller than tile_size
    pad_h = max(0, tile_size - h_orig)
    pad_w = max(0, tile_size - w_orig)
    if pad_h > 0 or pad_w > 0:
        img_pad = cv2.copyMakeBorder(img_gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        img_pad = img_gray
    h_pad, w_pad = img_pad.shape[:2]

    # 2D Gaussian feather blending window (size: tile_size x tile_size)
    coords = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    g1d = np.exp(-2.0 * (coords ** 2))
    weight_2d = np.outer(g1d, g1d).astype(np.float32)

    stride = max(1, tile_size - overlap)
    y_starts = list(range(0, h_pad - tile_size + 1, stride))
    if not y_starts or y_starts[-1] != h_pad - tile_size:
        y_starts.append(h_pad - tile_size)

    x_starts = list(range(0, w_pad - tile_size + 1, stride))
    if not x_starts or x_starts[-1] != w_pad - tile_size:
        x_starts.append(w_pad - tile_size)

    acc_pred = np.zeros((h_pad, w_pad), dtype=np.float32)
    acc_weight = np.zeros((h_pad, w_pad), dtype=np.float32)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                tile = img_pad[y:y + tile_size, x:x + tile_size]
                tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
                pred_t = model(tile_t)
                pred_np = torch.clamp(pred_t, 0.0, 1.0).squeeze().cpu().numpy()
                acc_pred[y:y + tile_size, x:x + tile_size] += pred_np * weight_2d
                acc_weight[y:y + tile_size, x:x + tile_size] += weight_2d

    merged = acc_pred / np.maximum(acc_weight, 1e-6)
    return (np.clip(merged[:h_orig, :w_orig], 0.0, 1.0) * 255.0).astype(np.uint8)


class DocumentRestorationDataset(Dataset):
    """
    Dataset loader for Document Image Restoration model.
    Reads paired degraded/clean images from a CSV metadata file.
    Supports Grayscale 1-channel patch-based training (RandomCrop) and validation (CenterCrop).

    Two CSV formats are supported (auto-detected from the header):

    1. Manifest (``datasets/metadata/dataset_manifest.csv``) - many inputs -> one target.
       Columns: split, original_id, degradation_type, variant_id, input_path, target_path.
       Paths are resolved relative to ``root_dir`` (default: parent of the CSV's folder, i.e.
       the ``datasets`` directory). Input and target names are NOT assumed to match.
       Use ``split`` ('train'/'val'/'test') and ``degradation_types`` to filter rows.

    2. Legacy: columns input, target (+ optional source_id, degradation, blur_*...).
    """

    def __init__(
        self,
        metadata_csv: str,
        image_size: Tuple[int, int] = (512, 512),
        patch_size: Optional[Tuple[int, int]] = None,
        transform: Optional[T.Compose] = None,
        return_metadata: bool = False,
        split: Optional[str] = None,
        root_dir: Optional[str] = None,
        degradation_types: Optional[List[str]] = None,
        target_cache_size: int = 64,
        is_train: Optional[bool] = None,
        filter_geometric: Optional[bool] = None,
        include_unpaired: bool = False,
    ):
        if patch_size is not None:
            self.patch_size = patch_size
        else:
            self.patch_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.image_size = self.patch_size
        self.include_unpaired = include_unpaired

        # Infer effective split ('train', 'val', 'test') when split is not explicitly passed
        csv_basename = os.path.basename(metadata_csv).lower() if metadata_csv else ""
        if split is not None:
            effective_split = split.strip().lower()
        elif "test" in csv_basename:
            effective_split = "test"
        elif "val" in csv_basename:
            effective_split = "val"
        elif "train" in csv_basename:
            effective_split = "train"
        else:
            effective_split = "train" if (is_train is None or is_train) else "val"
        self.effective_split = effective_split

        if is_train is not None:
            self.is_train = is_train
        else:
            self.is_train = (effective_split == "train")

        if filter_geometric is not None:
            self.filter_geometric = filter_geometric
        else:
            self.filter_geometric = effective_split in ("train", "val")

        self.return_metadata = return_metadata
        self.target_cache_size = target_cache_size
        self._target_cache: Dict[str, np.ndarray] = {}

        # Read CSV metadata
        self.pairs: List[Dict[str, str]] = []
        self.unpaired_samples: List[Dict[str, str]] = []
        self.total_rows = 0
        self.missing_count = 0
        self.filtered_count = 0
        self.unpaired_count = 0

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),  # converts [H,W] uint8 to [1,H,W] float [0,1]
            ])

        if not os.path.exists(metadata_csv):
            logger.error(f"Metadata CSV file not found: '{metadata_csv}'")
            return

        if root_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(metadata_csv)))
        wanted_types = set(degradation_types) if degradation_types else None

        with open(metadata_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            is_manifest = "input_path" in (reader.fieldnames or [])
            for row in reader:
                if is_manifest:
                    row_split = row.get("split", "").strip()
                    if split is not None and row_split != split:
                        continue
                    deg_str = row.get("degradation_type", "").strip()
                    if wanted_types is not None and deg_str not in wanted_types:
                        continue
                    self.total_rows += 1
                    raw_in = (row.get("input_path") or "").strip()
                    raw_tgt = (row.get("target_path") or "").strip()
                    input_path = os.path.normpath(os.path.join(root_dir, raw_in)) if raw_in else ""
                    target_path = os.path.normpath(os.path.join(root_dir, raw_tgt)) if raw_tgt else ""
                    src_id = row.get("original_id", "").strip()
                    active_split = (row_split or effective_split).lower()
                else:
                    self.total_rows += 1
                    raw_in = (row.get("input") or "").strip()
                    raw_tgt = (row.get("target") or "").strip()
                    input_path = os.path.normpath(raw_in) if raw_in else ""
                    target_path = os.path.normpath(raw_tgt) if raw_tgt else ""
                    deg_str = (row.get("degradation") or row.get("blur_level") or "").strip()
                    src_id = (row.get("source_id") or row.get("original_id") or "").strip()
                    active_split = (row.get("split") or effective_split).strip().lower()

                in_stem = os.path.splitext(os.path.basename(input_path))[0].lower() if input_path else ""
                is_blind_real = (
                    in_stem in ("real_11", "real_12")
                    or deg_str.lower() in ("unpaired_real", "blind_real")
                    or (raw_tgt == "" and bool(raw_in))
                )

                # Never allow blind evaluation images (real_11, real_12) into the train split
                if active_split == "train" and (in_stem in ("real_11", "real_12") or is_blind_real):
                    self.filtered_count += 1
                    continue

                if is_blind_real:
                    if input_path and os.path.exists(input_path):
                        self.unpaired_count += 1
                        unpaired_rec = {
                            "input": input_path,
                            "target": "",
                            "source_id": src_id or in_stem,
                            "degradation": deg_str or "unpaired_real",
                            "split": active_split,
                            "is_real_pair": False,
                            "is_unpaired": True,
                        }
                        self.unpaired_samples.append(unpaired_rec)
                        if self.include_unpaired and not self.is_train:
                            self.pairs.append(unpaired_rec)
                    else:
                        self.missing_count += 1
                    continue

                if not (input_path and target_path and os.path.exists(input_path) and os.path.exists(target_path)):
                    self.missing_count += 1
                    continue

                # Filter geometric distortions (BookBinding, Perspective, OpticalDistortion) in train & val
                # and filter Anh06 in train (open book on wooden table)
                if self.filter_geometric and active_split in ("train", "val"):
                    check_text = f"{os.path.basename(input_path)} {deg_str}".lower()
                    if any(kw in check_text for kw in GEOMETRIC_DISTORTION_KEYWORDS):
                        self.filtered_count += 1
                        continue
                    if active_split == "train":
                        id_text = f"{src_id} {os.path.basename(input_path)} {os.path.basename(target_path)}".lower()
                        if "anh06" in id_text:
                            self.filtered_count += 1
                            continue

                is_real_pair = (
                    "real_pairs" in input_path.replace("\\", "/").lower()
                    or "real_pairs" in target_path.replace("\\", "/").lower()
                    or deg_str.lower() in ("real_pairs", "realcamera")
                )

                if is_manifest:
                    self.pairs.append({
                        "input": input_path,
                        "target": target_path,
                        "source_id": src_id,
                        "original_id": src_id,
                        "degradation": deg_str,
                        "variant_id": row.get("variant_id", ""),
                        "split": row.get("split", ""),
                        "is_real_pair": is_real_pair,
                        "is_unpaired": False,
                    })
                else:
                    self.pairs.append({
                        "input": input_path,
                        "target": target_path,
                        "source_id": src_id,
                        "degradation": deg_str,
                        "is_real_pair": is_real_pair,
                        "is_unpaired": False,
                    })

        if self.missing_count > 0:
            logger.warning(
                f"Missing {self.missing_count}/{self.total_rows} image pair(s) in '{metadata_csv}'. "
                f"Loaded {len(self.pairs)} valid pair(s)."
            )
        elif self.total_rows == 0:
            logger.warning(f"Metadata CSV '{metadata_csv}' is empty (split={split}).")
        elif self.filtered_count > 0 or self.unpaired_count > 0:
            logger.info(
                f"Split='{effective_split}': Loaded {len(self.pairs)} paired sample(s), "
                f"{self.unpaired_count} unpaired blind-eval sample(s), "
                f"filtered {self.filtered_count} geometric/non-flat pair(s)."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_target(self, path: str, skip_normalization: bool = False) -> np.ndarray:
        """
        Load full-resolution clean target image as Grayscale (cached).
        For `real_pairs` targets (already carefully refined with bleed-through & watermark removal),
        skips re-running simple `clean_target_normalization`.
        """
        cached = self._target_cache.get(path)
        if cached is not None:
            return cached
        clean_bgr = load_image(path)
        if clean_bgr is None:
            raise FileNotFoundError(f"Failed to load clean target image: {path}")
        clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        is_real_target = skip_normalization or ("real_pairs" in path.replace("\\", "/").lower())
        if not is_real_target:
            clean_gray = clean_target_normalization(clean_gray)
        if len(self._target_cache) < self.target_cache_size:
            self._target_cache[path] = clean_gray
        return clean_gray

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        is_real_pair = bool(pair.get("is_real_pair", False))
        is_unpaired = bool(pair.get("is_unpaired", False)) or not pair.get("target")

        # Load degraded image as numpy BGR -> Grayscale
        deg_bgr = load_image(pair["input"])
        if deg_bgr is None:
            raise FileNotFoundError(f"Failed to load degraded input image: {pair['input']}")

        deg_gray = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2GRAY)

        if is_unpaired:
            deg_patch, _ = crop_patch(deg_gray, deg_gray, patch_size=self.patch_size, is_train=False)
            deg_tensor = self.transform(deg_patch)
            if deg_tensor.ndim == 2:
                deg_tensor = deg_tensor.unsqueeze(0)
            if self.return_metadata:
                return deg_tensor, None, pair
            return deg_tensor, deg_tensor

        clean_gray = self._load_target(pair["target"], skip_normalization=is_real_pair)

        # Align dimensions if input and clean target differ in resolution (e.g. LongestMaxSize)
        # Always resize degraded input to match clean target resolution
        if deg_gray.shape[:2] != clean_gray.shape[:2]:
            deg_gray = cv2.resize(deg_gray, (clean_gray.shape[1], clean_gray.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Patch-based extraction at full resolution (RandomCrop for train, CenterCrop for eval)
        deg_patch, clean_patch = crop_patch(deg_gray, clean_gray, patch_size=self.patch_size, is_train=self.is_train)

        # On-the-fly Hard Augmentation (HorizontalFlip 50%; skip synthetic blur/shadow for real_pairs)
        if self.is_train:
            deg_patch, clean_patch = apply_hard_augmentation(
                deg_patch, clean_patch, is_real_pair=is_real_pair
            )

        # Apply transforms: converts uint8 [0,255] [H,W] to float [0,1] [1,H,W]
        deg_tensor = self.transform(deg_patch)
        clean_tensor = self.transform(clean_patch)

        # Ensure tensor is shape [1, H, W]
        if deg_tensor.ndim == 2:
            deg_tensor = deg_tensor.unsqueeze(0)
        if clean_tensor.ndim == 2:
            clean_tensor = clean_tensor.unsqueeze(0)

        if self.return_metadata:
            return deg_tensor, clean_tensor, pair

        return deg_tensor, clean_tensor


