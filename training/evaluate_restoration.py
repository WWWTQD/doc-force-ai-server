import os
import sys
import argparse
import csv
import cv2
import numpy as np
import torch
import math
import json
from PIL import Image

# Add parent directory to path to import training modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train_restoration import DocumentRestorationUNet
from training.restoration_dataset import (
    load_image,
    save_image,
    unsharp_mask,
    resize_with_padding,
    remove_padding,
    normalize_clean_target,
    clean_target_normalization,
    infer_single_image_tiled,
    GEOMETRIC_DISTORTION_KEYWORDS,
)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two uint8 images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two uint8 images."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def _load_pairs_from_csv(csv_path: str, default_split: str = "test"):
    """Load valid image pairs and unpaired blind evaluation entries from a split CSV (test.csv or val.csv)."""
    pairs = []
    unpaired = []
    if not csv_path or not os.path.exists(csv_path):
        return pairs, unpaired
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_in = (row.get('input') or '').strip()
            raw_tgt = (row.get('target') or '').strip()
            in_p = os.path.normpath(raw_in) if raw_in else ''
            tgt_p = os.path.normpath(raw_tgt) if raw_tgt else ''
            deg_str = (row.get('degradation') or row.get('blur_level') or '').strip().lower()
            in_stem = os.path.splitext(os.path.basename(in_p))[0].lower() if in_p else ''

            is_blind = (
                in_stem in ('real_11', 'real_12')
                or deg_str in ('unpaired_real', 'blind_real')
                or (raw_tgt == '' and bool(raw_in))
            )
            if is_blind:
                if in_p and os.path.exists(in_p):
                    row_copy = dict(row)
                    row_copy['input'] = in_p
                    row_copy['target'] = ''
                    row_copy['split'] = row.get('split', default_split) or default_split
                    row_copy['is_unpaired'] = True
                    unpaired.append(row_copy)
                continue

            if in_p and tgt_p and os.path.exists(in_p) and os.path.exists(tgt_p):
                row_copy = dict(row)
                row_copy['input'] = in_p
                row_copy['target'] = tgt_p
                row_copy['split'] = row.get('split', default_split) or default_split
                row_copy['is_unpaired'] = False
                pairs.append(row_copy)
    return pairs, unpaired


def _collect_blind_real_samples(existing_unpaired):
    """Ensure datasets/processed/test/input/real_*.jpg and datasets/processed/val/input/real_*.jpg are included."""
    seen_paths = {os.path.abspath(item['input']) for item in existing_unpaired if item.get('input')}
    blind_list = list(existing_unpaired)

    candidates = [
        ('test', 'datasets/processed/test/input/real_11.jpg'),
        ('val', 'datasets/processed/val/input/real_12.jpg'),
    ]
    for split_name, folder in [('test', 'datasets/processed/test/input'), ('val', 'datasets/processed/val/input')]:
        if os.path.exists(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.lower().startswith('real_') and os.path.splitext(fn)[1].lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    candidates.append((split_name, os.path.join(folder, fn)))

    for split_name, rel_p in candidates:
        norm_p = os.path.normpath(rel_p)
        if os.path.exists(norm_p):
            abs_p = os.path.abspath(norm_p)
            if abs_p not in seen_paths:
                seen_paths.add(abs_p)
                stem = os.path.splitext(os.path.basename(norm_p))[0]
                blind_list.append({
                    'input': norm_p,
                    'target': '',
                    'source_id': stem,
                    'degradation': 'unpaired_real',
                    'blur_level': f'real_blind_{split_name}',
                    'split': split_name,
                    'is_unpaired': True,
                })
    return blind_list


def create_two_column_blind_comparison(raw_bgr: np.ndarray, pure_unet_gray: np.ndarray, target_h: int = 680) -> np.ndarray:
    """
    Create an honest 2-column comparison sheet for Unpaired Real Evaluation:
    [1. Anh Real Goc (Input duy nhat) | 2. Ket qua 100% Pure U-Net (Output tu AI)]
    """
    if raw_bgr.ndim == 2:
        raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2BGR)
    if pure_unet_gray.ndim == 2:
        unet_bgr = cv2.cvtColor(pure_unet_gray, cv2.COLOR_GRAY2BGR)
    else:
        unet_bgr = pure_unet_gray.copy()

    def _resize_col(img: np.ndarray, th: int) -> np.ndarray:
        hi, wi = img.shape[:2]
        scale = th / float(max(hi, 1))
        nw = max(240, int(wi * scale))
        return cv2.resize(img, (nw, th), interpolation=cv2.INTER_AREA)

    def _add_header(img: np.ndarray, title: str, color: tuple) -> np.ndarray:
        header_h = 46
        canvas = np.full((img.shape[0] + header_h, img.shape[1], 3), 245, dtype=np.uint8)
        canvas[header_h:, :] = img
        cv2.putText(canvas, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        return canvas

    col1 = _add_header(
        _resize_col(raw_bgr, target_h),
        "1. Anh Real Goc (Input duy nhat)",
        (0, 0, 210),
    )
    col2 = _add_header(
        _resize_col(unet_bgr, target_h),
        "2. Ket qua 100% Pure U-Net (Output tu AI)",
        (180, 60, 0),
    )
    divider = np.full((col1.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([col1, divider, col2])


def _summarize_group(items):
    """Compute average PSNR/SSIM summary for a subset of evaluated samples."""
    if not items:
        return {
            'count': 0,
            'restored_psnr': 0.0,
            'blur_psnr': 0.0,
            'psnr_improvement': 0.0,
            'restored_ssim': 0.0,
            'blur_ssim': 0.0,
            'ssim_improvement': 0.0,
        }
    r_psnr = float(np.mean([m['restored_psnr'] for m in items]))
    b_psnr = float(np.mean([m['blur_psnr'] for m in items]))
    r_ssim = float(np.mean([m['restored_ssim'] for m in items]))
    b_ssim = float(np.mean([m['blur_ssim'] for m in items]))
    return {
        'count': len(items),
        'restored_psnr': round(r_psnr, 2),
        'blur_psnr': round(b_psnr, 2),
        'psnr_improvement': round(r_psnr - b_psnr, 2),
        'restored_ssim': round(r_ssim, 4),
        'blur_ssim': round(b_ssim, 4),
        'ssim_improvement': round(r_ssim - b_ssim, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Document Restoration AI Model")
    parser.add_argument('--model', type=str, default='models/restoration_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str, default='datasets/metadata/test.csv',
                        help='Path to test split CSV')
    parser.add_argument('--val-csv', type=str, default='datasets/metadata/val.csv',
                        help='Path to val split CSV')
    parser.add_argument('--include-val', action='store_true', default=True,
                        help='Also export comparison sheets for val.csv')
    parser.add_argument('--no-val', action='store_false', dest='include_val',
                        help='Evaluate only test.csv without val.csv')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Target model tile size for sliding window inference')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between sliding window tiles')
    parser.add_argument('--blind-out-dir', type=str, default='outputs/restoration/real_blind_test',
                        help='Output directory for 100% Pure U-Net blind test images (real_11.jpg & real_12.jpg)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Model
    model = DocumentRestorationUNet().to(device)

    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Loaded model from {args.model}")
    else:
        print(f"ERROR: Model not found at {args.model}.")
        return

    model.eval()

    output_dir = 'outputs/restoration/test'
    comp_dir = 'outputs/restoration/comparison'
    blind_dir = args.blind_out_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(blind_dir, exist_ok=True)

    # Remove old comparison images so only current evaluation results exist
    for old_f in os.listdir(comp_dir):
        if old_f.endswith("_comparison.png"):
            try:
                os.remove(os.path.join(comp_dir, old_f))
            except OSError:
                pass

    # Read test CSV and val CSV
    test_pairs, test_unpaired = _load_pairs_from_csv(args.test_csv, default_split="test")
    val_pairs, val_unpaired = _load_pairs_from_csv(args.val_csv, default_split="val") if args.include_val else ([], [])

    eval_pairs = test_pairs + val_pairs
    blind_samples = _collect_blind_real_samples(test_unpaired + val_unpaired)

    if len(eval_pairs) == 0:
        manifest_p = "datasets/metadata/dataset_manifest.csv"
        if os.path.exists(manifest_p):
            print(f"No valid files in '{args.test_csv}'. Loading evaluation pairs from '{manifest_p}'...")
            with open(manifest_p, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("split", "").strip() not in ("test", "val"):
                        continue
                    raw_in = r.get("input_path", "").strip()
                    raw_tgt = r.get("target_path", "").strip()
                    if not raw_in or not raw_tgt:
                        continue
                    in_p = os.path.normpath(os.path.join("datasets", raw_in))
                    tgt_p = os.path.normpath(os.path.join("datasets", raw_tgt))
                    if os.path.exists(in_p) and os.path.exists(tgt_p):
                        eval_pairs.append({
                            'input': in_p,
                            'target': tgt_p,
                            'source_id': r.get('original_id', 'doc'),
                            'blur_level': r.get('degradation_type', 'degraded') + '_' + r.get('variant_id', '01'),
                            'split': r.get('split', 'test'),
                        })
        elif len(blind_samples) == 0:
            print("ERROR: No valid test pairs found.")
            return

    print(
        f"Total paired samples to evaluate: {len(eval_pairs)} (test={len(test_pairs)}, val={len(val_pairs)}) | "
        f"Unpaired Blind Real samples: {len(blind_samples)}"
    )
    all_metrics = []

    for pair in eval_pairs:
        blur_path = os.path.normpath(pair['input'].strip())
        orig_path = os.path.normpath(pair['target'].strip())
        source_id = pair.get('source_id') or pair.get('original_id') or 'unknown'
        blur_level = pair.get('blur_level') or pair.get('degradation') or os.path.splitext(os.path.basename(blur_path))[0]
        split_tag = pair.get('split', 'test')

        blur_img = load_image(blur_path)
        orig_img = load_image(orig_path)
        if blur_img is None or orig_img is None:
            print(f"  SKIP {source_id} ({blur_level}): failed to load images")
            continue

        # Convert to Grayscale and apply synchronized clean_target_normalization on Ground Truth
        blur_gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        orig_gray = clean_target_normalization(orig_gray)

        # Align dimensions if input and clean target differ in resolution (e.g. LongestMaxSize)
        # Always resize degraded input to match clean target resolution
        if blur_gray.shape[:2] != orig_gray.shape[:2]:
            blur_gray = cv2.resize(blur_gray, (orig_gray.shape[1], orig_gray.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Sliding Window Tiling Inference (512x512 with 128 overlap & Gaussian feather blending)
        restored_cropped = infer_single_image_tiled(
            model, blur_gray, device, tile_size=args.image_size, overlap=args.overlap
        )

        # Post-processing: Unsharp Masking
        restored_final = unsharp_mask(restored_cropped, amount=0.8, radius=1.0)

        # Compute metrics on grayscale images
        l1 = float(np.mean(np.abs(orig_gray.astype(float) - restored_final.astype(float))))
        mse = float(np.mean((orig_gray.astype(float) - restored_final.astype(float)) ** 2))
        psnr = calculate_psnr(orig_gray, restored_final)
        ssim = calculate_ssim(orig_gray, restored_final)

        # Baseline metrics (degraded vs normalized clean GT)
        blur_psnr = calculate_psnr(orig_gray, blur_gray)
        blur_ssim = calculate_ssim(orig_gray, blur_gray)

        check_text = f"{os.path.basename(blur_path)} {blur_level}".lower()
        is_geometric = any(kw in check_text for kw in GEOMETRIC_DISTORTION_KEYWORDS)

        metrics = {
            'split': split_tag,
            'source_id': source_id,
            'blur_level': blur_level,
            'is_geometric': is_geometric,
            'restored_l1': round(l1, 4),
            'restored_mse': round(mse, 4),
            'restored_psnr': round(psnr, 2),
            'restored_ssim': round(ssim, 4),
            'blur_psnr': round(blur_psnr, 2),
            'blur_ssim': round(blur_ssim, 4),
            'psnr_improvement': round(psnr - blur_psnr, 2),
        }
        all_metrics.append(metrics)
        geom_tag = "GEOM" if is_geometric else "ALIGNED"
        print(f"  [{split_tag.upper()}:{geom_tag} | {source_id}/{blur_level}] "
              f"Restored PSNR: {psnr:.2f} dB (Degraded: {blur_psnr:.2f} dB, diff={psnr-blur_psnr:+.2f} dB) | "
              f"SSIM: {ssim:.4f} (Degraded: {blur_ssim:.4f})")

        # Save images in structured folders
        sample_name = f"{source_id}_{blur_level}"
        save_dir = os.path.join(output_dir, sample_name)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'input_blur.png'), blur_gray)
        cv2.imwrite(os.path.join(save_dir, 'restored.png'), restored_final)
        cv2.imwrite(os.path.join(save_dir, 'original.png'), orig_gray)

        # Create comparison contact sheet: Original (GT) | Degraded Input | U-Net Restored
        target_h = 450
        def resize_for_display(img, th):
            hi, wi = img.shape[:2]
            scale = th / hi
            return cv2.resize(img, (int(wi * scale), th), interpolation=cv2.INTER_AREA)

        def add_label(img, label):
            img_c = img.copy()
            cv2.putText(img_c, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return img_c

        disp_orig = add_label(resize_for_display(cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR), target_h), "1. Ground Truth")
        disp_blur = add_label(resize_for_display(cv2.cvtColor(blur_gray, cv2.COLOR_GRAY2BGR), target_h), "2. Degraded Input")
        disp_rest = add_label(resize_for_display(cv2.cvtColor(restored_final, cv2.COLOR_GRAY2BGR), target_h), "3. U-Net Restored")

        contact_sheet = np.hstack([disp_orig, disp_blur, disp_rest])
        comparison_path = os.path.join(comp_dir, f"{sample_name}_comparison.png")
        cv2.imwrite(comparison_path, contact_sheet)
        print(f"     -> Saved comparison contact sheet: {comparison_path}")

    # ======================== UNPAIRED REAL BLIND EVALUATION (100% PURE U-NET) ========================
    blind_results = []
    if blind_samples:
        print("\n" + "=" * 68)
        print("UNPAIRED REAL BLIND EVALUATION (100% PURE U-NET INFERENCE - NO TARGET)")
        print("=" * 68)
        for b_item in blind_samples:
            raw_path = os.path.normpath(b_item['input'])
            split_tag = b_item.get('split', 'test')
            stem = os.path.splitext(os.path.basename(raw_path))[0]
            raw_bgr = load_image(raw_path)
            if raw_bgr is None:
                print(f"  [BLIND-{split_tag.upper()}] SKIP {raw_path}: failed to load")
                continue

            raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

            # 100% Pure U-Net Inference directly from raw_gray (NO post-processing, NO unsharp mask, NO fake target)
            pure_unet_gray = infer_single_image_tiled(
                model, raw_gray, device, tile_size=args.image_size, overlap=args.overlap, refine_glyphs=False
            )

            pure_out_path = os.path.join(blind_dir, f"{stem}_pure_unet.png")
            comp_2col_path = os.path.join(blind_dir, f"{stem}_blind_comparison.png")

            two_col_sheet = create_two_column_blind_comparison(raw_bgr, pure_unet_gray, target_h=680)
            save_image(pure_out_path, pure_unet_gray)
            save_image(comp_2col_path, two_col_sheet)

            blind_rec = {
                'split': split_tag,
                'source_id': stem,
                'input_path': raw_path,
                'pure_unet_output_path': pure_out_path,
                'comparison_2col_path': comp_2col_path,
                'mode': '100% Pure U-Net (Unpaired Blind Evaluation - No Target)',
            }
            blind_results.append(blind_rec)
            print(
                f"  [BLIND-{split_tag.upper()} | {stem}] 100% Pure U-Net Output -> {pure_out_path} | "
                f"2-Column Comparison -> {comp_2col_path}"
            )

    # Summary separated into pixel_aligned_metrics and geometric_distorted_metrics
    if all_metrics or blind_results:
        test_only = [m for m in all_metrics if m.get('split') == 'test'] or all_metrics
        pixel_aligned = [m for m in test_only if not m.get('is_geometric', False)]
        geom_distorted = [m for m in test_only if m.get('is_geometric', False)]

        overall_summary = _summarize_group(test_only)
        aligned_summary = _summarize_group(pixel_aligned)
        geom_summary = _summarize_group(geom_distorted)

        print("\n" + "=" * 68)
        print("TEST RESULTS SUMMARY (Separated by Alignment)")
        print("=" * 68)
        print(f"1. Pixel-Aligned Samples ({aligned_summary['count']} samples - True Pixel Restoration):")
        print(f"   Avg Restored PSNR: {aligned_summary['restored_psnr']:.2f} dB (Blur: {aligned_summary['blur_psnr']:.2f} dB, Diff: {aligned_summary['psnr_improvement']:+.2f} dB)")
        print(f"   Avg Restored SSIM: {aligned_summary['restored_ssim']:.4f} (Blur: {aligned_summary['blur_ssim']:.4f}, Diff: {aligned_summary['ssim_improvement']:+.4f})")
        print("-" * 68)
        print(f"2. Geometric-Distorted Samples ({geom_summary['count']} samples - Perspective/BookBinding/Optical):")
        print(f"   Avg Restored PSNR: {geom_summary['restored_psnr']:.2f} dB (Blur: {geom_summary['blur_psnr']:.2f} dB, Diff: {geom_summary['psnr_improvement']:+.2f} dB)")
        print(f"   Avg Restored SSIM: {geom_summary['restored_ssim']:.4f} (Blur: {geom_summary['blur_ssim']:.4f}, Diff: {geom_summary['ssim_improvement']:+.4f})")
        print("-" * 68)
        print(f"3. Overall Paired Test Samples ({overall_summary['count']} test samples, {len(all_metrics)} total incl. val):")
        print(f"   Avg Restored PSNR: {overall_summary['restored_psnr']:.2f} dB | Avg Restored SSIM: {overall_summary['restored_ssim']:.4f}")
        print("-" * 68)
        print(f"4. Unpaired Real Blind Evaluation ({len(blind_results)} samples - 100% Pure U-Net, No Target): {[b['source_id'] for b in blind_results]}")
        print("=" * 68)

        metrics_path = 'outputs/restoration/test_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'pixel_aligned_metrics': aligned_summary,
                'geometric_distorted_metrics': geom_summary,
                'average': overall_summary,
                'per_image': all_metrics,
                'blind_real_samples': blind_results,
                'total_samples': len(all_metrics),
            }, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()


