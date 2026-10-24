import os
import csv
import json
import numpy as np
import pytest
import cv2
import torch

from training.train_restoration import (
    DocumentRestorationUNet,
    DocumentCleanLoss,
    BimodalStrokeLoss,
    SobelLoss,
    SSIMLoss,
    calculate_psnr,
    train_restoration_model,
)
from training.restoration_dataset import (
    DocumentRestorationDataset,
    resize_with_padding,
    remove_padding,
    unsharp_mask,
    load_image,
    save_image,
    infer_single_image_tiled,
    apply_hard_augmentation,
    clean_target_normalization,
)
from training.infer_restoration import restore_image_for_yolov8
from training.generate_clean_targets import (
    generate_master_clean_target,
    create_preview_comparison,
    process_real_directory,
)


def create_dummy_image(file_path: str, color=(128, 128, 128), size=(64, 64)):
    """Helper to create a solid color dummy image for testing."""
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    save_image(file_path, img)
    return file_path


def test_unet_architecture_and_forward():
    """Verify 3-stage U-Net with 2-channel Local Illumination Normalization and +0.2 bias init."""
    model = DocumentRestorationUNet(in_channels=1, out_channels=1)
    assert hasattr(model, "down3") and hasattr(model, "up3") and hasattr(model, "dec3")
    assert model.enc1[0].in_channels == 2, "First conv in enc1 must accept 2 channels [x, x_norm]"
    last_conv = model.dec1[-2]
    assert torch.allclose(last_conv.bias.data, torch.full_like(last_conv.bias.data, 0.2)), "Last conv bias must be initialized to 0.2"
    dummy_input = torch.rand(2, 1, 128, 128, dtype=torch.float32)
    output = model(dummy_input)

    assert output.shape == (2, 1, 128, 128), f"Expected shape (2, 1, 128, 128), got {output.shape}"
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output must be bounded in [0, 1]"


def test_sobel_loss_properties():
    """Verify SobelLoss gives 0 on identical inputs and positive on different inputs for 1-channel."""
    loss_fn = SobelLoss()
    img1 = torch.rand(1, 1, 64, 64, dtype=torch.float32)
    img2 = img1.clone()

    loss_identical = loss_fn(img1, img2)
    assert torch.isclose(loss_identical, torch.tensor(0.0), atol=1e-6), "Loss on identical tensors must be ~0"

    img_diff = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    # Add high frequency edge
    img_diff[:, :, 30:34, :] = 1.0
    loss_different = loss_fn(img1, img_diff)
    assert loss_different.item() > 0.0, "Loss on different tensors must be positive"


def test_document_clean_loss_properties():
    """Verify DocumentCleanLoss combines stroke-weighted Charbonnier, BCE, Sobel, SSIM, and BimodalStrokeLoss."""
    loss_fn = DocumentCleanLoss()
    bimodal_fn = BimodalStrokeLoss()
    target = torch.ones(1, 1, 32, 32, dtype=torch.float32)
    target[:, :, 10:20, 10:20] = 0.0  # black text region
    pred_good = target.clone()
    pred_bad = torch.full((1, 1, 32, 32), 0.5, dtype=torch.float32)

    assert torch.isclose(bimodal_fn(pred_good, target), torch.tensor(0.0), atol=1e-6)
    assert bimodal_fn(pred_bad, target).item() > 0.2

    loss_good = loss_fn(pred_good, target)
    loss_bad = loss_fn(pred_bad, target)
    assert loss_good.item() < loss_bad.item(), "Clean prediction must have strictly lower loss than noisy prediction"


def test_tiled_inference_and_hard_augmentation():
    """Verify sliding window tiled inference and on-the-fly hard augmentation preserve shapes."""
    model = DocumentRestorationUNet(in_channels=1, out_channels=1).eval()
    large_gray = np.random.randint(0, 256, size=(160, 200), dtype=np.uint8)
    restored = infer_single_image_tiled(model, large_gray, torch.device("cpu"), tile_size=128, overlap=32)
    assert restored.shape == (160, 200)
    assert restored.dtype == np.uint8

    deg_p = np.full((64, 64), 200, dtype=np.uint8)
    clean_p = np.full((64, 64), 255, dtype=np.uint8)
    aug_deg, aug_clean = apply_hard_augmentation(deg_p, clean_p)
    assert aug_deg.shape == (64, 64) and aug_clean.shape == (64, 64)


def test_psnr_calculation():
    """Verify PSNR utility calculation."""
    t1 = torch.ones(1, 1, 32, 32)
    t2 = torch.ones(1, 1, 32, 32)
    psnr_same = calculate_psnr(t1, t2)
    assert psnr_same == 100.0, "PSNR on identical images should be 100.0"

    t3 = torch.zeros(1, 1, 32, 32)
    psnr_diff = calculate_psnr(t1, t3)
    assert 0.0 <= psnr_diff < 50.0, "PSNR on completely inverted images should be low"


def test_dataset_loader_and_missing_file_warning(tmp_path):
    """Verify dataset loads valid pairs, skips missing ones, and tracks counts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    img1 = create_dummy_image(str(data_dir / "input1.png"), color=(100, 100, 100))
    img2 = create_dummy_image(str(data_dir / "target1.png"), color=(200, 200, 200))
    missing_input = str(data_dir / "missing_in.png")
    missing_target = str(data_dir / "missing_tgt.png")

    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target", "source_id", "degradation"])
        writer.writeheader()
        writer.writerow({"input": img1, "target": img2, "source_id": "doc1", "degradation": "blur"})
        writer.writerow({"input": missing_input, "target": missing_target, "source_id": "doc2", "degradation": "blur"})

    dataset = DocumentRestorationDataset(metadata_csv=str(csv_path), image_size=(64, 64))

    assert len(dataset) == 1, "Only the 1 existing pair should be loaded"
    assert dataset.total_rows == 2, "Total rows in CSV should be 2"
    assert dataset.missing_count == 1, "Missing count should be 1"

    deg_tensor, clean_tensor = dataset[0]
    assert deg_tensor.shape == (1, 64, 64)
    assert clean_tensor.shape == (1, 64, 64)
    assert deg_tensor.dtype == torch.float32


def test_train_raises_error_on_empty_dataset(tmp_path):
    """Verify train_restoration_model raises ValueError when dataset has 0 valid pairs."""
    empty_csv = tmp_path / "empty.csv"
    with open(empty_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target"])
        writer.writeheader()

    with pytest.raises(ValueError, match="No training pairs found"):
        train_restoration_model(
            train_csv=str(empty_csv),
            val_csv=str(empty_csv),
            output_dir=str(tmp_path / "models"),
            epochs=1,
        )


def test_train_one_epoch_and_checkpoint_saving(tmp_path):
    """Verify full 1-epoch training loop, checkpoint saving, and log validity."""
    data_dir = tmp_path / "samples"
    data_dir.mkdir()

    # Create 2 train pairs, 1 val pair
    train_rows = []
    for i in range(2):
        in_p = create_dummy_image(str(data_dir / f"tr_in_{i}.png"), color=(50 * (i + 1), 50, 50))
        tgt_p = create_dummy_image(str(data_dir / f"tr_tgt_{i}.png"), color=(50 * (i + 1), 60, 60))
        train_rows.append({"input": in_p, "target": tgt_p, "source_id": f"tr_{i}"})

    val_rows = []
    v_in = create_dummy_image(str(data_dir / "val_in.png"), color=(80, 80, 80))
    v_tgt = create_dummy_image(str(data_dir / "val_tgt.png"), color=(90, 90, 90))
    val_rows.append({"input": v_in, "target": v_tgt, "source_id": "val_0"})

    train_csv = tmp_path / "train.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target", "source_id"])
        writer.writeheader()
        writer.writerows(train_rows)

    val_csv = tmp_path / "val.csv"
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target", "source_id"])
        writer.writeheader()
        writer.writerows(val_rows)

    output_dir = tmp_path / "checkpoints"

    # Train 1 epoch on CPU
    train_restoration_model(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        lr=1e-4,
        image_size=64,
    )

    best_ckpt = output_dir / "restoration_best.pth"
    last_ckpt = output_dir / "restoration_last.pth"
    log_file = output_dir / "training_log.json"

    assert best_ckpt.exists(), "restoration_best.pth must be created"
    assert last_ckpt.exists(), "restoration_last.pth must be created"
    assert log_file.exists(), "training_log.json must be created"

    # Verify JSON log content
    with open(log_file, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    assert log_data["epochs"] == 1
    assert log_data["train_samples"] == 2
    assert log_data["val_samples"] == 1
    assert log_data["best_val_psnr"] is not None
    assert isinstance(log_data["best_val_psnr"], (int, float))

    # Verify checkpoint can be loaded for inference
    model = DocumentRestorationUNet()
    state_dict = torch.load(str(best_ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        infer_out = model(torch.rand(1, 1, 64, 64))
    assert infer_out.shape == (1, 1, 64, 64)


def test_image_helpers():
    """Verify padding, unpadding, and unsharp masking utilities."""
    sample = np.zeros((100, 200, 3), dtype=np.uint8)
    sample[20:80, 40:160] = 255

    padded, pad_info = resize_with_padding(sample, target_size=(128, 128))
    assert padded.shape == (128, 128, 3)

    unpadded = remove_padding(padded, pad_info, original_size=(100, 200))
    assert unpadded.shape == (100, 200, 3)

    sharpened = unsharp_mask(sample, amount=0.5, radius=1.0)
    assert sharpened.shape == sample.shape
    assert sharpened.dtype == np.uint8


def test_geometric_keywords_and_size_alignment(tmp_path):
    """Verify new geometric keywords ('centercrop', 'bindingsandfasteners') and deg-to-clean resizing."""
    from training.restoration_dataset import GEOMETRIC_DISTORTION_KEYWORDS

    assert "centercrop" in GEOMETRIC_DISTORTION_KEYWORDS
    assert "bindingsandfasteners" in GEOMETRIC_DISTORTION_KEYWORDS

    # Verify that when deg and clean have different sizes, deg is resized to match clean
    data_dir = tmp_path / "size_data"
    data_dir.mkdir()
    deg_img = create_dummy_image(str(data_dir / "deg_small.png"), size=(40, 50))
    clean_img = create_dummy_image(str(data_dir / "clean_large.png"), size=(80, 100))

    csv_path = tmp_path / "size_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "target", "source_id", "degradation"])
        writer.writeheader()
        writer.writerow({"input": deg_img, "target": clean_img, "source_id": "test_doc", "degradation": "none"})

    ds = DocumentRestorationDataset(metadata_csv=str(csv_path), patch_size=(32, 32), is_train=False)
    deg_t, clean_t = ds[0]
    assert deg_t.shape == (1, 32, 32)
    assert clean_t.shape == (1, 32, 32)


def test_clean_target_normalization_sauvola_soft_matting():
    """
    Verify 5-step Multi-Scale Morphological + Sauvola Soft Matting:
    - Uneven lighting/shadow background becomes pure white (255).
    - Text stroke centers become deep black (<= 25) while preserving diacritics.
    """
    h, w = 160, 240
    # Create synthetic shaded paper background (gradient from 140 to 230)
    x_grid = np.linspace(140, 230, w, dtype=np.float32)
    shaded_bg = np.tile(x_grid, (h, 1)).astype(np.uint8)

    # Draw dark text strokes and a small Vietnamese diacritic dot (3x3)
    doc_gray = shaded_bg.copy()
    cv2.putText(doc_gray, "TIENG VIET", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 25, 3, cv2.LINE_AA)
    doc_gray[52:55, 80:83] = 35  # Small diacritic mark

    master_clean = clean_target_normalization(doc_gray)
    assert master_clean.shape == (h, w)
    assert master_clean.dtype == np.uint8

    # Background border should be pure white 255
    assert np.all(master_clean[:20, :] == 255), "Shaded background must be normalized to pure white 255"
    # Stroke centers must be deep black (<= 25)
    assert int(np.min(master_clean)) <= 25, "Text stroke centers must be deep black (0..25)"
    # Small diacritic mark must be preserved (dark < 180)
    assert int(np.min(master_clean[50:57, 78:85])) < 180, "Small Vietnamese diacritic must be preserved"


def test_generate_clean_targets_and_auto_train_registration(tmp_path):
    """
    Verify generate_clean_targets.py generates clean_targets/, preview_comparisons/,
    exports to datasets/original/AnhXX.png, and automatically saves (raw_photo -> clean_target)
    into train.csv and dataset_manifest.csv.
    """
    dataset_root = tmp_path / "datasets"
    real_dir = dataset_root / "real"
    orig_dir = dataset_root / "original"
    real_dir.mkdir(parents=True)
    orig_dir.mkdir(parents=True)

    # Create a dummy existing Anh22.jpg in original_dir so export assigns Anh23.png
    create_dummy_image(str(orig_dir / "Anh22.jpg"), color=(240, 240, 240), size=(80, 80))

    # Create a raw camera image in datasets/real/
    raw_img = np.full((120, 160, 3), 190, dtype=np.uint8)
    cv2.putText(raw_img, "RPi Cam", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    raw_path = real_dir / "rpi_capture_01.jpg"
    save_image(str(raw_path), raw_img)

    results = process_real_directory(
        input_dir=str(real_dir),
        model_path=str(tmp_path / "nonexistent_model.pth"),
        use_unet=False,
        export_to_original=True,
        save_to_train=True,
        dataset_root=str(dataset_root),
    )

    assert len(results) == 1
    rec = results[0]
    assert os.path.exists(rec["clean_target_path"])
    assert os.path.exists(rec["preview_path"])
    assert os.path.exists(rec["exported_original_path"])
    assert os.path.basename(rec["exported_original_path"]) == "Anh23.png"
    assert os.path.exists(rec["train_input_path"])
    assert os.path.exists(rec["train_target_path"])

    # Verify train.csv and dataset_manifest.csv contain the new pair
    train_csv = dataset_root / "metadata" / "train.csv"
    manifest_csv = dataset_root / "metadata" / "dataset_manifest.csv"
    assert train_csv.exists()
    assert manifest_csv.exists()

    with open(train_csv, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    assert len(train_rows) == 1
    assert train_rows[0]["source_id"] == "Anh23"
    assert train_rows[0]["degradation"] == "RealCamera"


def test_restore_image_for_yolov8():
    """Verify restore_image_for_yolov8 outputs 3-channel BGR (with and without preserve_color)."""
    model = DocumentRestorationUNet(in_channels=1, out_channels=1).eval()
    sample_bgr = np.random.randint(50, 220, size=(96, 128, 3), dtype=np.uint8)

    out_bgr = restore_image_for_yolov8(model, sample_bgr, torch.device("cpu"), tile_size=64, overlap=16, preserve_color=False)
    assert out_bgr.shape == (96, 128, 3)
    assert out_bgr.dtype == np.uint8

    out_color = restore_image_for_yolov8(model, sample_bgr, torch.device("cpu"), tile_size=64, overlap=16, preserve_color=True)
    assert out_color.shape == (96, 128, 3)
    assert out_color.dtype == np.uint8


def test_real_pairs_pipeline_and_bleed_through_loss():
    """Verify BleedThroughSuppressionLoss, refine_real_clean_target, and 3-column U-Net preview."""
    from training.train_restoration import BleedThroughSuppressionLoss
    from training.prepare_real_pairs import refine_real_clean_target
    from training.infer_restoration import create_three_column_preview

    # 1. BleedThroughSuppressionLoss penalizes gray output (<0.92) where target > 0.95
    bt_loss = BleedThroughSuppressionLoss()
    white_target = torch.ones((1, 1, 32, 32), dtype=torch.float32)
    clean_pred = torch.full((1, 1, 32, 32), 0.99, dtype=torch.float32)
    bleed_pred = torch.full((1, 1, 32, 32), 0.75, dtype=torch.float32)
    assert bt_loss(clean_pred, white_target).item() < 0.01
    assert bt_loss(bleed_pred, white_target).item() > 0.20

    # 2. refine_real_clean_target removes bleed-through (>210, low gradient) while keeping sharp text
    raw_bgr = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.putText(raw_bgr, "SHARP", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2)
    target_gray = refine_real_clean_target(raw_bgr)
    assert target_gray.shape == (120, 160)
    assert np.mean(target_gray == 255) > 0.70
    assert int(np.min(target_gray)) <= 30

    # 3. 3-column preview sheet generation
    preview = create_three_column_preview(raw_bgr, target_gray, raw_bgr)
    assert preview.ndim == 3 and preview.shape[2] == 3


def test_unpaired_blind_real_evaluation_and_zero_leakage(tmp_path):
    """
    Verify that real_11.jpg and real_12.jpg are excluded from real_pairs and train.csv,
    and that DocumentRestorationDataset + 2-column blind preview handle unpaired real samples cleanly.
    """
    from training.prepare_real_pairs import prepare_real_pairs
    from training.infer_restoration import create_two_column_blind_preview
    from training.evaluate_restoration import create_two_column_blind_comparison

    ds_root = tmp_path / "datasets"
    real_dir = ds_root / "real"
    rp_in = ds_root / "processed" / "real_pairs" / "input"
    val_in = ds_root / "processed" / "val" / "input"
    test_in = ds_root / "processed" / "test" / "input"
    meta_dir = ds_root / "metadata"
    for d in (real_dir, rp_in, val_in, test_in, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create real_01 (train), real_11 (blind test), real_12 (blind val)
    create_dummy_image(str(real_dir / "real_01.jpg"), color=(210, 210, 210), size=(80, 80))
    create_dummy_image(str(real_dir / "real_11.jpg"), color=(200, 200, 200), size=(80, 80))
    create_dummy_image(str(real_dir / "real_12.jpg"), color=(200, 200, 200), size=(80, 80))
    create_dummy_image(str(test_in / "real_11.jpg"), color=(200, 200, 200), size=(80, 80))
    create_dummy_image(str(val_in / "real_12.jpg"), color=(200, 200, 200), size=(80, 80))
    # Even if real_11.jpg was accidentally copied into real_pairs/input, prepare_real_pairs must purge it
    create_dummy_image(str(rp_in / "real_11.jpg"), color=(200, 200, 200), size=(80, 80))

    for csv_name in ("train.csv", "val.csv", "test.csv"):
        with open(meta_dir / csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["input", "target", "source_id", "degradation", "blur_level"])
            writer.writeheader()

    records = prepare_real_pairs(real_dir=str(real_dir), dataset_root=str(ds_root), train_ratio=1.0, train_oversample=3)
    assert len(records) == 1
    assert records[0]["source_id"] == "real_01"
    assert not (rp_in / "real_11.jpg").exists(), "Leaked real_11.jpg must be purged from real_pairs/input"
    assert not (ds_root / "processed" / "real_pairs" / "target" / "real_11.png").exists()
    assert not (ds_root / "processed" / "real_pairs" / "target" / "real_12.png").exists()

    # Check train.csv has 0 occurrences of real_11 or real_12
    train_text = (meta_dir / "train.csv").read_text(encoding="utf-8")
    assert "real_11" not in train_text and "real_12" not in train_text

    # Check DocumentRestorationDataset handles unpaired row in test.csv without missing_count error
    raw_bgr = np.full((80, 100, 3), 210, dtype=np.uint8)
    pure_gray = np.full((80, 100), 250, dtype=np.uint8)
    sheet1 = create_two_column_blind_preview(raw_bgr, pure_gray)
    sheet2 = create_two_column_blind_comparison(raw_bgr, pure_gray)
    assert sheet1.ndim == 3 and sheet1.shape[2] == 3
    assert sheet2.ndim == 3 and sheet2.shape[2] == 3


