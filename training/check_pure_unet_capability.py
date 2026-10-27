import os
import sys
import glob
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train_restoration import DocumentRestorationUNet
from training.restoration_dataset import load_image, save_image, infer_single_image_tiled


def create_pure_comparison_sheet(
    raw_bgr: np.ndarray,
    zeroshot_gray: np.ndarray,
    finetuned_gray: np.ndarray,
    display_height: int = 720,
) -> np.ndarray:
    """
    Build a 3-column pure AI comparison sheet:
    [1. Anh Real Goc (Input duy nhat) | 2. Pure U-Net (Chua hoc anh Real - Zero-shot) | 3. Pure U-Net (Da Fine-tune)]
    """
    if raw_bgr.ndim == 2:
        raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2BGR)
    zs_bgr = cv2.cvtColor(zeroshot_gray, cv2.COLOR_GRAY2BGR) if zeroshot_gray.ndim == 2 else zeroshot_gray
    ft_bgr = cv2.cvtColor(finetuned_gray, cv2.COLOR_GRAY2BGR) if finetuned_gray.ndim == 2 else finetuned_gray

    def _resize_col(img: np.ndarray, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = th / float(max(h, 1))
        nw = max(240, int(round(w * scale)))
        return cv2.resize(img, (nw, th), interpolation=cv2.INTER_AREA)

    def _add_header(img: np.ndarray, title: str, color: tuple) -> np.ndarray:
        header_h = 46
        canvas = np.full((img.shape[0] + header_h, img.shape[1], 3), 248, dtype=np.uint8)
        canvas[header_h:, :] = img
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return canvas

    c1 = _add_header(_resize_col(raw_bgr, display_height), "1. Anh Real Goc (Input duy nhat)", (0, 0, 200))
    c2 = _add_header(_resize_col(zs_bgr, display_height), "2. Pure U-Net (Chua hoc anh Real - Zero-shot)", (180, 50, 0))
    c3 = _add_header(_resize_col(ft_bgr, display_height), "3. Pure U-Net (Da Fine-tune)", (0, 140, 0))

    divider = np.full((c1.shape[0], 4, 3), 210, dtype=np.uint8)
    return np.hstack([c1, divider, c2, divider, c3])


def run_pure_capability_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Pure U-Net Capability Check on: {device}")

    zs_weights = "models/restoration_pretrained_v4_backup.pth"
    ft_weights = "models/restoration_best.pth"

    assert os.path.exists(zs_weights), f"Missing {zs_weights}"
    assert os.path.exists(ft_weights), f"Missing {ft_weights}"

    print(f"Loading Zero-shot Model from: {zs_weights}")
    model_zs = DocumentRestorationUNet().to(device)
    model_zs.load_state_dict(torch.load(zs_weights, map_location=device, weights_only=True))
    model_zs.eval()

    print(f"Loading Fine-tuned Model from: {ft_weights}")
    model_ft = DocumentRestorationUNet().to(device)
    model_ft.load_state_dict(torch.load(ft_weights, map_location=device, weights_only=True))
    model_ft.eval()

    out_zs_dir = "datasets/real/pure_unet_zeroshot"
    out_ft_dir = "datasets/real/pure_unet_finetuned"
    out_cmp_dir = "datasets/real/pure_unet_capability_check"
    os.makedirs(out_zs_dir, exist_ok=True)
    os.makedirs(out_ft_dir, exist_ok=True)
    os.makedirs(out_cmp_dir, exist_ok=True)

    img_paths = sorted(glob.glob("datasets/real/IMG20261022*.jpg"))
    print(f"Found {len(img_paths)} real camera images to evaluate.")

    metrics = []

    for idx, img_p in enumerate(img_paths, 1):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        raw_bgr = load_image(img_p)
        assert raw_bgr is not None, f"Cannot load {img_p}"
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        h, w = raw_gray.shape

        t0 = time.time()
        # 1. Pure Zero-Shot forward pass (NO target, NO post-processing filters, NO thresholding)
        zs_gray = infer_single_image_tiled(model_zs, raw_gray, device, tile_size=512, overlap=128, refine_glyphs=False)
        t_zs = time.time() - t0

        t1 = time.time()
        # 2. Pure Fine-Tuned forward pass (NO target, NO post-processing filters, NO thresholding)
        ft_gray = infer_single_image_tiled(model_ft, raw_gray, device, tile_size=512, overlap=128, refine_glyphs=False)
        t_ft = time.time() - t1

        # Save pure outputs directly
        save_image(os.path.join(out_zs_dir, f"{stem}_zeroshot.png"), zs_gray)
        save_image(os.path.join(out_ft_dir, f"{stem}_finetuned.png"), ft_gray)

        # 3. Create 3-column comparison sheet
        cmp_sheet = create_pure_comparison_sheet(raw_bgr, zs_gray, ft_gray)
        cmp_path = os.path.join(out_cmp_dir, f"{stem}_comparison.png")
        save_image(cmp_path, cmp_sheet)

        # Calculate statistics
        # Estimate background mask from raw image (> 180 intensity)
        bg_mask = raw_gray > 180
        # Estimate stroke mask from raw image (< 90 intensity)
        stroke_mask = raw_gray < 90

        raw_bg_mean = float(np.mean(raw_gray[bg_mask])) if np.any(bg_mask) else float(np.mean(raw_gray))
        zs_bg_mean = float(np.mean(zs_gray[bg_mask])) if np.any(bg_mask) else float(np.mean(zs_gray))
        ft_bg_mean = float(np.mean(ft_gray[bg_mask])) if np.any(bg_mask) else float(np.mean(ft_gray))

        raw_stroke_min = int(np.min(raw_gray))
        zs_stroke_min = int(np.min(zs_gray))
        ft_stroke_min = int(np.min(ft_gray))

        zs_white_ratio = float(np.mean(zs_gray > 245)) * 100.0
        ft_white_ratio = float(np.mean(ft_gray > 245)) * 100.0

        rec = {
            "stem": stem,
            "res": f"{w}x{h}",
            "raw_bg_mean": round(raw_bg_mean, 1),
            "zs_bg_mean": round(zs_bg_mean, 1),
            "ft_bg_mean": round(ft_bg_mean, 1),
            "raw_stroke_min": raw_stroke_min,
            "zs_stroke_min": zs_stroke_min,
            "ft_stroke_min": ft_stroke_min,
            "zs_white_pct": round(zs_white_ratio, 1),
            "ft_white_pct": round(ft_white_ratio, 1),
            "time_zs": round(t_zs, 1),
            "time_ft": round(t_ft, 1),
        }
        metrics.append(rec)
        print(
            f"[{idx:2d}/{len(img_paths)}] {stem} ({w}x{h}) | "
            f"BG Mean: Raw={rec['raw_bg_mean']} -> ZS={rec['zs_bg_mean']} -> FT={rec['ft_bg_mean']} | "
            f"White (>245): ZS={rec['zs_white_pct']}% -> FT={rec['ft_white_pct']}%"
        )

    print("\n" + "=" * 90)
    print("SUMMARY METRICS: PURE U-NET REAL-WORLD CAPABILITY")
    print("=" * 90)
    print(f"{'Image':<20} {'Size':<12} {'Raw BG':<8} {'ZS BG':<8} {'FT BG':<8} {'ZS White%':<10} {'FT White%':<10}")
    print("-" * 90)
    for m in metrics:
        print(f"{m['stem']:<20} {m['res']:<12} {m['raw_bg_mean']:<8.1f} {m['zs_bg_mean']:<8.1f} {m['ft_bg_mean']:<8.1f} {m['zs_white_pct']:<10.1f} {m['ft_white_pct']:<10.1f}")
    print("=" * 90)


if __name__ == "__main__":
    run_pure_capability_check()
