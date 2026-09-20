"""
Dataset Synthesis & Degradation Script cho Doc Forge.

Tạo dataset tổng hợp (Synthetic Dataset) huấn luyện các mô hình AI:
  - De-warping (xoay/nắn thẳng tài liệu)
  - Document Shadow Removal (khử bóng mờ)
  - Document Denoising & Enhancement (khử nhiễu, tăng tương phản)

Script đọc ảnh tài liệu sắc nét từ --input-dir, áp dụng chuỗi biến đổi
suy hao thực tế và xuất ra từng cặp ảnh ({name}_input.jpg, {name}_target.jpg) vào --output-dir.

Sử dụng:
  python scripts/synthesize_dataset.py --input-dir ./data/raw --output-dir ./data/synthetic --num-samples 50
"""

import argparse
import glob
import logging
import math
import os
import random
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("synthesize_dataset")


class DocumentDegrader:
    """Áp dụng các hiệu ứng suy hao tài liệu."""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def apply_perspective_skew(self, img: np.ndarray) -> np.ndarray:
        """Tạo góc nghiêng biến dạng hình học 3D nhẹ (Perspective Warp)."""
        h, w = img.shape[:2]
        max_offset = min(w, h) * 0.08  # Tối đa 8% chiều rộng/cao

        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(0, max_offset), random.uniform(0, max_offset)],
            [w - random.uniform(0, max_offset), random.uniform(0, max_offset)],
            [random.uniform(0, max_offset), h - random.uniform(0, max_offset)],
            [w - random.uniform(0, max_offset), h - random.uniform(0, max_offset)],
        ])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    def add_shadow(self, img: np.ndarray) -> np.ndarray:
        """Tạo bóng mờ che phủ dạng gradient (Gradient Shadow)."""
        h, w = img.shape[:2]

        # Tạo gradient mask
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Chọn góc hướng của bóng mờ ngẫu nhiên
        angle = random.uniform(0, 2 * math.pi)
        gradient = xx * math.cos(angle) + yy * math.sin(angle)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)

        # Mức độ tối của bóng
        shadow_intensity = random.uniform(0.35, 0.75)
        mask = 1.0 - shadow_intensity * gradient
        mask_3ch = np.dstack([mask] * 3)

        shadowed = (img.astype(np.float32) * mask_3ch).clip(0, 255).astype(np.uint8)
        return shadowed

    def add_blur_and_noise(self, img: np.ndarray) -> np.ndarray:
        """Thêm nhiễu muỗi Gaussian và mờ chuyển động (Motion/Gaussian Blur)."""
        out = img.copy()

        # 1. Motion blur
        if random.random() > 0.4:
            kernel_size = random.choice([3, 5, 7])
            kernel = np.zeros((kernel_size, kernel_size))
            if random.random() > 0.5:
                kernel[int((kernel_size - 1) / 2), :] = 1.0
            else:
                kernel[:, int((kernel_size - 1) / 2)] = 1.0
            kernel /= kernel_size
            out = cv2.filter2D(out, -1, kernel)

        # 2. Gaussian noise
        if random.random() > 0.3:
            sigma = random.uniform(10.0, 30.0)
            gauss = np.random.normal(0, sigma, out.shape).astype(np.float32)
            out = (out.astype(np.float32) + gauss).clip(0, 255).astype(np.uint8)

        return out

    def adjust_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Giảm độ sáng và tương phản (giả lập camera thiếu sáng)."""
        alpha = random.uniform(0.6, 0.9)  # Contrast
        beta = random.uniform(-40, 10)     # Brightness
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted

    def add_crease_line(self, img: np.ndarray) -> np.ndarray:
        """Tạo vết nếp gấp/gập trang giấy (Page Crease Line)."""
        h, w = img.shape[:2]
        out = img.copy()

        # Tạo đường nếp gấp nằm ngang hoặc dọc
        if random.random() > 0.5:
            y = random.randint(int(h * 0.2), int(h * 0.8))
            thickness = random.randint(2, 5)
            cv2.line(out, (0, y), (w, y), (180, 180, 180), thickness)
        else:
            x = random.randint(int(w * 0.2), int(w * 0.8))
            thickness = random.randint(2, 5)
            cv2.line(out, (x, 0), (x, h), (180, 180, 180), thickness)

        return out

    def degrade(self, img: np.ndarray) -> np.ndarray:
        """Áp dụng ngẫu nhiên chuỗi biến đổi suy hao."""
        out = img.copy()

        if random.random() > 0.2:
            out = self.apply_perspective_skew(out)
        if random.random() > 0.2:
            out = self.add_shadow(out)
        if random.random() > 0.3:
            out = self.adjust_brightness_contrast(out)
        if random.random() > 0.2:
            out = self.add_blur_and_noise(out)
        if random.random() > 0.5:
            out = self.add_crease_line(out)

        return out


def main():
    parser = argparse.ArgumentParser(description="Tạo dataset tổng hợp suy hao tài liệu (Synthetic Document Dataset Generator)")
    parser.add_argument("--input-dir", type=str, default="./data/raw", help="Thư mục chứa ảnh gốc sắc nét")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic", help="Thư mục lưu cặp ảnh kết quả")
    parser.add_argument("--num-samples", type=int, default=20, help="Số lượng mẫu tổng hợp cần tạo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cho tính tái lập")
    args = parser.parse_args()

    LOG.info("╔══════════════════════════════════════════════════════════╗")
    LOG.info("║    Doc Forge — Synthetic Dataset Generator Script        ║")
    LOG.info("╚══════════════════════════════════════════════════════════╝")
    LOG.info("Input dir : %s", args.input-dir if hasattr(args, 'input-dir') else args.input_dir)
    LOG.info("Output dir: %s", args.output_dir)
    LOG.info("Samples   : %d", args.num_samples)

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Tìm các file ảnh gốc
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        LOG.warning("Không tìm thấy ảnh gốc trong %s. Đang tạo ảnh sample thử nghiệm...", input_dir)
        os.makedirs(input_dir, exist_ok=True)
        sample_path = os.path.join(input_dir, "sample_doc.jpg")

        # Tạo một ảnh tài liệu văn bản mẫu
        canvas = np.ones((1200, 800, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "DOC FORGE AI — SAMPLE DOCUMENT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, "Clean high resolution document page for dataset generation.", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        for i in range(10):
            cv2.putText(canvas, f"Line {i+1}: Synthetic text pattern with Vietnamese accent test.", (50, 220 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(sample_path, canvas)
        image_paths.append(sample_path)
        LOG.info("Đã tạo mẫu %s", sample_path)

    degrader = DocumentDegrader(seed=args.seed)
    generated_count = 0

    for idx in range(args.num_samples):
        src_path = random.choice(image_paths)
        img = cv2.imread(src_path)
        if img is None:
            continue

        degraded_img = degrader.degrade(img)

        base_name = f"sample_{idx+1:04d}"
        target_path = os.path.join(output_dir, f"{base_name}_target.jpg")
        input_path = os.path.join(output_dir, f"{base_name}_input.jpg")

        cv2.imwrite(target_path, img)
        cv2.imwrite(input_path, degraded_img)
        generated_count += 1

        if (idx + 1) % 5 == 0 or (idx + 1) == args.num_samples:
            LOG.info("  -> Đã tạo %d/%d cặp ảnh synthetic", idx + 1, args.num_samples)

    LOG.info("══ Hoàn tất! Đã lưu %d cặp ảnh vào %s ══", generated_count, output_dir)


if __name__ == "__main__":
    main()
