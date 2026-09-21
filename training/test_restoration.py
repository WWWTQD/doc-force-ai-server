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

# Add parent directory to path to import training.train_restoration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train_restoration import DocumentRestorationUNet


def resize_with_padding(img, target_size=(512, 512)):
    """Resize keeping aspect ratio and pad with zeros."""
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


def remove_padding(img, pad_info):
    """Remove padding added by resize_with_padding."""
    pad_top, pad_left, new_h, new_w = pad_info
    return img[pad_top:pad_top + new_h, pad_left:pad_left + new_w]


def unsharp_mask(img: np.ndarray, amount: float = 0.8, radius: float = 1.0) -> np.ndarray:
    """Apply Unsharp Masking filter to sharpen text edges and contrast."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def calculate_psnr(img1, img2):
    """Calculate PSNR between two uint8 images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM between two uint8 images (manual implementation)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
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
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def load_image(path):
    """Load image safely supporting unicode paths and all formats."""
    try:
        pil_img = Image.open(path).convert("RGB")
        img_rgb = np.array(pil_img)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Document Restoration AI Model")
    parser.add_argument('--model', type=str, default='models/restoration_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str, default='datasets/metadata/test.csv',
                        help='Path to test split CSV')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Target model input image size')
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
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    # Read test CSV
    if not os.path.exists(args.test_csv):
        print(f"ERROR: Test CSV not found at {args.test_csv}.")
        return

    test_pairs = []
    with open(args.test_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_pairs.append(row)

    print(f"Test pairs: {len(test_pairs)}")
    all_metrics = []

    target_size = (args.image_size, args.image_size)

    for pair in test_pairs:
        blur_path = os.path.normpath(pair['input'].strip())
        orig_path = os.path.normpath(pair['target'].strip())
        source_id = pair.get('source_id', 'unknown')
        blur_level = pair.get('blur_level', 'unknown')

        if not os.path.exists(blur_path):
            print(f"  SKIP {source_id} ({blur_level}): blur image not found at {blur_path}")
            continue
        if not os.path.exists(orig_path):
            print(f"  SKIP {source_id} ({blur_level}): original image not found at {orig_path}")
            continue

        blur_img = load_image(blur_path)
        orig_img = load_image(orig_path)
        if blur_img is None or orig_img is None:
            print(f"  SKIP {source_id} ({blur_level}): failed to load images")
            continue

        h, w = blur_img.shape[:2]

        # Preprocess: resize with padding
        canvas, pad_info = resize_with_padding(blur_img, target_size)

        # Convert BGR to RGB for model
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(canvas_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Post-process: tensor to numpy, RGB back to BGR
        output_rgb = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        # Remove padding and resize to original dimensions
        restored_cropped = remove_padding(output_bgr, pad_info)
        restored_final = cv2.resize(restored_cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        restored_final = unsharp_mask(restored_final, amount=0.8, radius=1.0)

        # Resize original to same size for fair comparison
        orig_resized = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Compute metrics
        l1 = float(np.mean(np.abs(orig_resized.astype(float) - restored_final.astype(float))))
        mse = float(np.mean((orig_resized.astype(float) - restored_final.astype(float)) ** 2))
        psnr = calculate_psnr(orig_resized, restored_final)
        ssim = calculate_ssim(orig_resized, restored_final)

        # Baseline metrics
        blur_resized = cv2.resize(blur_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        blur_psnr = calculate_psnr(orig_resized, blur_resized)
        blur_ssim = calculate_ssim(orig_resized, blur_resized)

        metrics = {
            'source_id': source_id,
            'blur_level': blur_level,
            'restored_l1': round(l1, 4),
            'restored_mse': round(mse, 4),
            'restored_psnr': round(psnr, 2),
            'restored_ssim': round(ssim, 4),
            'blur_psnr': round(blur_psnr, 2),
            'blur_ssim': round(blur_ssim, 4),
            'psnr_improvement': round(psnr - blur_psnr, 2),
        }
        all_metrics.append(metrics)
        print(f"  [{source_id}/{blur_level}] "
              f"Restored PSNR: {psnr:.2f} (blur: {blur_psnr:.2f}, diff={psnr-blur_psnr:+.2f}) | "
              f"SSIM: {ssim:.4f} (blur: {blur_ssim:.4f})")

        # Save images in structured folders
        sample_name = f"{source_id}_{blur_level}"
        save_dir = os.path.join(output_dir, sample_name)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'input_blur.png'), blur_img)
        cv2.imwrite(os.path.join(save_dir, 'restored.png'), restored_final)
        cv2.imwrite(os.path.join(save_dir, 'original.png'), orig_img)

        # Create comparison contact sheet: Original | Blur | Restored
        target_h = 400
        def resize_for_display(img, target_h):
            h_img, w_img = img.shape[:2]
            scale = target_h / h_img
            return cv2.resize(img, (int(w_img * scale), target_h), interpolation=cv2.INTER_AREA)

        def add_label(img, label):
            img_c = img.copy()
            cv2.putText(img_c, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img_c

        disp_orig = add_label(resize_for_display(orig_img, target_h), "Original")
        disp_blur = add_label(resize_for_display(blur_img, target_h), "Blur")
        disp_rest = add_label(resize_for_display(restored_final, target_h), "Restored")

        contact_sheet = np.hstack([disp_orig, disp_blur, disp_rest])
        cv2.imwrite(os.path.join(comp_dir, f"{sample_name}_comparison.png"), contact_sheet)

    # Summary
    if all_metrics:
        avg_psnr = np.mean([m['restored_psnr'] for m in all_metrics])
        avg_ssim = np.mean([m['restored_ssim'] for m in all_metrics])
        avg_blur_psnr = np.mean([m['blur_psnr'] for m in all_metrics])
        avg_blur_ssim = np.mean([m['blur_ssim'] for m in all_metrics])
        avg_improvement = np.mean([m['psnr_improvement'] for m in all_metrics])

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total test samples: {len(all_metrics)}")
        print(f"Avg Restored PSNR: {avg_psnr:.2f} dB")
        print(f"Avg Blur PSNR:     {avg_blur_psnr:.2f} dB")
        print(f"Avg PSNR Improvement: {avg_improvement:+.2f} dB")
        print(f"Avg Restored SSIM: {avg_ssim:.4f}")
        print(f"Avg Blur SSIM:     {avg_blur_ssim:.4f}")
        print("=" * 60)

        metrics_path = 'outputs/restoration/test_metrics.json'
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'per_image': all_metrics,
                'average': {
                    'restored_psnr': round(avg_psnr, 2),
                    'blur_psnr': round(avg_blur_psnr, 2),
                    'psnr_improvement': round(avg_improvement, 2),
                    'restored_ssim': round(avg_ssim, 4),
                    'blur_ssim': round(avg_blur_ssim, 4),
                },
                'total_samples': len(all_metrics),
            }, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
