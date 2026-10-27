import os
import cv2
import numpy as np
import random
import argparse
import csv
import json
from pathlib import Path
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="Generate synthetic blur dataset")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--variants', type=int, default=3, help="Number of variants per image")
    parser.add_argument('--original-dir', type=str, default='datasets/images/original/', help="Original images directory")
    parser.add_argument('--output-dir', type=str, default='datasets/images/synthetic/blur/', help="Output directory for blurred images")
    return parser.parse_args()

def generate_motion_blur_kernel(kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        t = i - center
        x = int(center + t * np.cos(np.radians(angle)))
        y = int(center + t * np.sin(np.radians(angle)))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    kernel /= np.sum(kernel)
    return kernel

def process_image(img_path, output_dir, original_dir):
    try:
        # Read image
        if img_path.suffix.lower() in ['.webp', '.jfif']:
            pil_img = Image.open(img_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                pil_img = Image.open(img_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if img is None:
             raise ValueError("Image could not be read.")
    except Exception as e:
        return None, str(e)
    
    stem = img_path.stem
    ext = img_path.suffix.lower()
    if ext == '.jfif':
        ext = '.jpg'

    records = []
    
    # Light blur
    k_light = random.choice([3, 5])
    s_light = random.uniform(0.5, 1.0)
    img_light = cv2.GaussianBlur(img, (k_light, k_light), s_light)
    out_light_name = f"{stem}_blur_light{ext}"
    out_light_path = output_dir / out_light_name
    cv2.imwrite(str(out_light_path), img_light)
    records.append({
        'input': str(out_light_path).replace('\\', '/'),
        'target': str(img_path).replace('\\', '/'),
        'source_id': stem,
        'degradation': 'blur',
        'blur_level': 'light',
        'blur_type': 'gaussian',
        'kernel_size': k_light,
        'sigma': round(s_light, 3)
    })

    # Medium blur
    k_medium = random.choice([5, 7])
    s_medium = random.uniform(1.0, 2.0)
    img_medium = cv2.GaussianBlur(img, (k_medium, k_medium), s_medium)
    out_medium_name = f"{stem}_blur_medium{ext}"
    out_medium_path = output_dir / out_medium_name
    cv2.imwrite(str(out_medium_path), img_medium)
    records.append({
        'input': str(out_medium_path).replace('\\', '/'),
        'target': str(img_path).replace('\\', '/'),
        'source_id': stem,
        'degradation': 'blur',
        'blur_level': 'medium',
        'blur_type': 'gaussian',
        'kernel_size': k_medium,
        'sigma': round(s_medium, 3)
    })

    # Strong blur
    if random.random() < 0.5:
        k_strong = random.choice([7, 9])
        s_strong = random.uniform(2.0, 3.0)
        img_strong = cv2.GaussianBlur(img, (k_strong, k_strong), s_strong)
        btype = 'gaussian'
        k_val = k_strong
        s_val = round(s_strong, 3)
    else:
        k_strong = random.randint(9, 15)
        if k_strong % 2 == 0:
            k_strong += 1
        angle = random.uniform(0, 180)
        kernel = generate_motion_blur_kernel(k_strong, angle)
        img_strong = cv2.filter2D(img, -1, kernel)
        btype = 'motion'
        k_val = k_strong
        s_val = ""

    out_strong_name = f"{stem}_blur_strong{ext}"
    out_strong_path = output_dir / out_strong_name
    cv2.imwrite(str(out_strong_path), img_strong)
    records.append({
        'input': str(out_strong_path).replace('\\', '/'),
        'target': str(img_path).replace('\\', '/'),
        'source_id': stem,
        'degradation': 'blur',
        'blur_level': 'strong',
        'blur_type': btype,
        'kernel_size': k_val,
        'sigma': s_val
    })

    return records, None

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    orig_dir = Path(args.original_dir)
    out_dir = Path(args.output_dir)
    meta_dir = Path('datasets/metadata')

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {'.jpg', '.jpeg', '.png', '.webp', '.jfif'}
    
    if not orig_dir.exists():
        print(f"Original directory {orig_dir} does not exist.")
        return

    all_images = [p for p in orig_dir.rglob('*') if p.is_file() and p.suffix.lower() in allowed_exts]
    
    all_records = []
    failed = 0
    details = []

    for img_path in all_images:
        records, err = process_image(img_path, out_dir, orig_dir)
        if err:
            failed += 1
            details.append({"file": str(img_path), "error": err})
            print(f"Failed to process {img_path}: {err}")
        else:
            all_records.extend(records)
            details.append({"file": str(img_path), "status": "success", "variants": len(records)})

    csv_path = meta_dir / 'blur_pairs.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'target', 'source_id', 'degradation', 'blur_level', 'blur_type', 'kernel_size', 'sigma'])
        writer.writeheader()
        writer.writerows(all_records)

    report = {
        "original_count": len(all_images),
        "blur_generated": len(all_records),
        "failed": failed,
        "seed": args.seed,
        "variants_per_image": args.variants,
        "details": details
    }

    report_path = meta_dir / 'blur_generation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print("Summary:")
    print(f"Original images processed: {len(all_images)}")
    print(f"Blur variants generated: {len(all_records)}")
    print(f"Failed: {failed}")
    print(f"Metadata saved to: {csv_path}")
    print(f"Report saved to: {report_path}")

if __name__ == '__main__':
    main()
