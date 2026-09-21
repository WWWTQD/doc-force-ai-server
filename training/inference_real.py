import os
import sys
import argparse
import cv2
import numpy as np
import torch
import glob
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


def load_image(path):
    """Load image safely supporting unicode paths and all image formats."""
    try:
        pil_img = Image.open(path).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


def save_image(path, img):
    """Save image safely supporting unicode paths."""
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext if ext else '.png', img)
    if result:
        with open(path, 'wb') as f:
            f.write(buf)


def main():
    parser = argparse.ArgumentParser(description="Run Restoration on Real Blur Images")
    parser.add_argument('--model', type=str, default='models/restoration_best.pth')
    parser.add_argument('--input-dir', type=str, default='datasets/images/real/blur/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DocumentRestorationUNet().to(device)

    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Loaded model from {args.model}")
    else:
        print(f"ERROR: Model not found at {args.model}.")
        return

    model.eval()

    output_dir = 'outputs/restoration_real'
    comp_dir = 'outputs/restoration_real/comparison'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.jfif')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))

    # Deduplicate
    image_paths = list(set(image_paths))
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    processed = 0
    for idx, img_path in enumerate(sorted(image_paths), 1):
        blur_img = load_image(img_path)
        if blur_img is None:
            print(f"  [{idx}/{len(image_paths)}] SKIP: Could not load image")
            continue

        h, w = blur_img.shape[:2]
        canvas, pad_info = resize_with_padding(blur_img)

        # Convert BGR to RGB for model input
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(canvas_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Post-process: RGB tensor back to BGR for OpenCV
        output_rgb = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        restored_cropped = remove_padding(output_bgr, pad_info)
        restored_final = cv2.resize(restored_cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        restored_final = unsharp_mask(restored_final, amount=0.8, radius=1.0)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        filename = f"real_{idx}_{base_name}.png"
        
        save_image(os.path.join(output_dir, filename), restored_final)

        # Create side-by-side comparison
        target_h = 400

        def resize_for_display(img, th):
            hi, wi = img.shape[:2]
            sc = th / hi
            return cv2.resize(img, (int(wi * sc), th), interpolation=cv2.INTER_AREA)

        def add_label(img, label):
            img_c = img.copy()
            cv2.putText(img_c, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img_c

        disp_input = add_label(resize_for_display(blur_img, target_h), "Input Blur")
        disp_rest = add_label(resize_for_display(restored_final, target_h), "Restored")
        contact_sheet = np.hstack([disp_input, disp_rest])

        save_image(os.path.join(comp_dir, f"comparison_{idx}.png"), contact_sheet)
        processed += 1
        print(f"  [{idx}/{len(image_paths)}] Processed successfully -> {filename.encode('ascii', 'ignore').decode('ascii')}")

    print(f"\nFinished processing {processed} images.")
    print(f"Restored images: {output_dir}")
    print(f"Comparisons: {comp_dir}")


if __name__ == '__main__':
    main()
