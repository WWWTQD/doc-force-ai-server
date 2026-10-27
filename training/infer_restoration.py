import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch

# Add parent directory to path to import training modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train_restoration import DocumentRestorationUNet
from training.restoration_dataset import (
    load_image,
    save_image,
    unsharp_mask,
    resize_with_padding,
    remove_padding,
    infer_single_image_tiled,
)


def restore_image_for_yolov8(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
    preserve_color: bool = False,
    refine_glyphs: bool = False,
    apply_postprocess: bool = False,
) -> np.ndarray:
    """
    AI-Centric Document Restoration for YOLOv8:
    1. Passes image directly through `DocumentRestorationUNet` (`infer_single_image_tiled`).
    2. Default mode (preserve_color=False, apply_postprocess=False):
       Output is 100% Pure U-Net output tensor converted to BGR uint8,
       WITHOUT any background clipping (e.g. np.where(y < 245, ..., 255)) or unsharp masking.
    3. If preserve_color=True: merges U-Net's raw prediction with original chrominance (Cr, Cb).
    4. If apply_postprocess=True: optionally applies stroke unsharp masking and white floor clipping.
    """
    if img_bgr.ndim == 2:
        gray = img_bgr
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 100% Pure U-Net forward pass
    y_restored = infer_single_image_tiled(
        model, gray, device, tile_size=tile_size, overlap=overlap, refine_glyphs=refine_glyphs
    )

    if apply_postprocess:
        y_proc = np.where(y_restored < 245, unsharp_mask(y_restored, amount=0.65, radius=1.0), 255).astype(np.uint8)
    else:
        y_proc = y_restored

    if not preserve_color:
        return cv2.cvtColor(y_proc, cv2.COLOR_GRAY2BGR)

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr_chan, cb_chan = cv2.split(ycrcb)
    chroma_sat = np.hypot(cr_chan.astype(np.float32) - 128.0, cb_chan.astype(np.float32) - 128.0)
    color_fg_mask = (y_proc < 245) & (chroma_sat > 18.0)
    cr_clean = np.where(color_fg_mask, cr_chan, 128).astype(np.uint8)
    cb_clean = np.where(color_fg_mask, cb_chan, 128).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([y_proc, cr_clean, cb_clean]), cv2.COLOR_YCrCb2BGR)


def create_three_column_preview(
    raw_bgr: np.ndarray,
    target_gray_or_bgr: np.ndarray,
    unet_bgr: np.ndarray,
    display_height: int = 680,
) -> np.ndarray:
    """
    Build a 3-column comparison sheet:
    [1. Anh Real Goc (Input) | 2. Dap an Mau (Target) | 3. Ket qua tu Model U-Net (Output cho YOLO)]
    """
    if raw_bgr.ndim == 2:
        raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2BGR)
    if target_gray_or_bgr.ndim == 2:
        target_bgr = cv2.cvtColor(target_gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        target_bgr = target_gray_or_bgr.copy()
    if unet_bgr.ndim == 2:
        unet_bgr = cv2.cvtColor(unet_bgr, cv2.COLOR_GRAY2BGR)

    def _resize_col(img: np.ndarray, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = th / float(max(h, 1))
        nw = max(220, int(w * scale))
        return cv2.resize(img, (nw, th), interpolation=cv2.INTER_AREA)

    def _add_header(img: np.ndarray, title: str, color: tuple) -> np.ndarray:
        header_h = 44
        canvas = np.full((img.shape[0] + header_h, img.shape[1], 3), 245, dtype=np.uint8)
        canvas[header_h:, :] = img
        cv2.putText(canvas, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        return canvas

    col1 = _add_header(_resize_col(raw_bgr, display_height), "1. Anh Real Goc (Input)", (0, 0, 210))
    col2 = _add_header(_resize_col(target_bgr, display_height), "2. Dap an Mau (Target)", (0, 140, 0))
    col3 = _add_header(_resize_col(unet_bgr, display_height), "3. Ket qua tu Model U-Net (Output cho YOLO)", (180, 60, 0))

    # Match column widths if needed
    divider = np.full((col1.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([col1, divider, col2, divider, col3])


def create_two_column_blind_preview(
    raw_bgr: np.ndarray,
    unet_bgr: np.ndarray,
    display_height: int = 680,
) -> np.ndarray:
    """
    Build an honest 2-column blind evaluation sheet for unpaired real images (e.g., real_11.jpg, real_12.jpg):
    [1. Anh Real Goc (Input duy nhat) | 2. Ket qua 100% Pure U-Net (Output tu AI)]
    """
    if raw_bgr.ndim == 2:
        raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_GRAY2BGR)
    if unet_bgr.ndim == 2:
        unet_bgr = cv2.cvtColor(unet_bgr, cv2.COLOR_GRAY2BGR)

    def _resize_col(img: np.ndarray, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = th / float(max(h, 1))
        nw = max(240, int(w * scale))
        return cv2.resize(img, (nw, th), interpolation=cv2.INTER_AREA)

    def _add_header(img: np.ndarray, title: str, color: tuple) -> np.ndarray:
        header_h = 44
        canvas = np.full((img.shape[0] + header_h, img.shape[1], 3), 245, dtype=np.uint8)
        canvas[header_h:, :] = img
        cv2.putText(canvas, title, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        return canvas

    col1 = _add_header(
        _resize_col(raw_bgr, display_height),
        "1. Anh Real Goc (Input duy nhat)",
        (0, 0, 210),
    )
    col2 = _add_header(
        _resize_col(unet_bgr, display_height),
        "2. Ket qua 100% Pure U-Net (Output tu AI)",
        (180, 60, 0),
    )

    divider = np.full((col1.shape[0], 4, 3), 200, dtype=np.uint8)
    return np.hstack([col1, divider, col2])


def main():
    parser = argparse.ArgumentParser(description="Run DocumentRestorationUNet Inference for YOLOv8")
    parser.add_argument("--model", type=str, default="models/restoration_best.pth")
    parser.add_argument("--input-dir", "--input", dest="input_dir", type=str, default="datasets/real")
    parser.add_argument("--target-dir", type=str, default="datasets/processed/real_pairs/target")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=str, default="outputs/restoration_real")
    parser.add_argument("--preview-dir", type=str, default="datasets/real/unet_results_preview")
    parser.add_argument("--blind-out-dir", type=str, default="outputs/restoration/real_blind_test")
    parser.add_argument("--tile-size", type=int, default=512, help="Sliding window tile size")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles")
    parser.add_argument(
        "--preserve-color",
        action="store_true",
        default=False,
        help="Preserve foreground color via YCrCb space while restoring luminance Y with U-Net (default: False for Pure U-Net Grayscale)",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        default=False,
        help="Apply stroke unsharp mask and background thresholding (default: False for Pure U-Net)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DocumentRestorationUNet().to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Loaded model from {args.model}")
    else:
        print(f"ERROR: Model not found at {args.model}.")
        return

    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)
    os.makedirs(args.blind_out_dir, exist_ok=True)
    comp_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    extensions = (".jpg", ".jpeg", ".png", ".webp", ".jfif")
    image_paths = []
    if os.path.exists(args.input_dir):
        for entry in sorted(os.listdir(args.input_dir)):
            full_p = os.path.join(args.input_dir, entry)
            if os.path.isfile(full_p) and os.path.splitext(entry)[1].lower() in extensions:
                image_paths.append(full_p)

    print(f"Found {len(image_paths)} images in {args.input_dir}")

    processed = 0
    for idx, img_path in enumerate(image_paths, 1):
        raw_bgr = load_image(img_path)
        if raw_bgr is None:
            print(f"  [{idx}/{len(image_paths)}] SKIP: Could not load image {img_path}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        is_blind_eval = base_name.lower() in ("real_11", "real_12")

        # For blind evaluation images (real_11, real_12), strictly enforce 100% Pure U-Net
        restored_bgr = restore_image_for_yolov8(
            model,
            raw_bgr,
            device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            preserve_color=False if is_blind_eval else args.preserve_color,
            refine_glyphs=False,
            apply_postprocess=False if is_blind_eval else args.postprocess,
        )

        filename = f"{base_name}_unet_yolo.png"
        save_image(os.path.join(args.output_dir, filename), restored_bgr)

        target_path = os.path.join(args.target_dir, f"{base_name}.png")
        has_valid_target = (not is_blind_eval) and os.path.exists(target_path)

        if has_valid_target:
            target_img = load_image(target_path)
            preview_sheet = create_three_column_preview(raw_bgr, target_img, restored_bgr)
        else:
            preview_sheet = create_two_column_blind_preview(raw_bgr, restored_bgr)
            save_image(os.path.join(args.blind_out_dir, f"{base_name}_pure_unet.png"), restored_bgr)
            save_image(os.path.join(args.blind_out_dir, f"{base_name}_blind_comparison.png"), preview_sheet)

        preview_path = os.path.join(args.preview_dir, f"{base_name}_unet_preview.png")
        save_image(preview_path, preview_sheet)
        save_image(os.path.join(comp_dir, f"comparison_{idx}_{base_name}.png"), preview_sheet)

        processed += 1
        mode_tag = "2-Col Blind Pure U-Net" if not has_valid_target else "3-Col Paired Preview"
        print(f"  [{idx}/{len(image_paths)}] [{mode_tag}] Processed -> {preview_path}")

    print(f"\nFinished processing {processed} images.")
    print(f"Restored YOLOv8 images: {args.output_dir}")
    print(f"U-Net Previews: {args.preview_dir}")
    print(f"Blind Test Pure U-Net Previews: {args.blind_out_dir}")


if __name__ == "__main__":
    main()

