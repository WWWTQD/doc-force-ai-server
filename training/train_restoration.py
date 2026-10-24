import os
import sys
import argparse
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.restoration_dataset import DocumentRestorationDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_restoration")


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class SobelLoss(nn.Module):
    """Multi-Scale Sobel Gradient Loss to enforce sharp text edge reconstruction across scales."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad_l1(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p_gx = F.conv2d(p, self.sobel_x, padding=1)
        p_gy = F.conv2d(p, self.sobel_y, padding=1)
        t_gx = F.conv2d(t, self.sobel_x, padding=1)
        t_gy = F.conv2d(t, self.sobel_y, padding=1)
        return F.l1_loss(p_gx, t_gx) + F.l1_loss(p_gy, t_gy)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 3:
            pred_g = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_g = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_g = pred
            target_g = target

        loss_scale1 = self._grad_l1(pred_g, target_g)
        if pred_g.shape[-2] >= 8 and pred_g.shape[-1] >= 8:
            pred_half = F.avg_pool2d(pred_g, kernel_size=2, stride=2)
            target_half = F.avg_pool2d(target_g, kernel_size=2, stride=2)
            loss_scale2 = self._grad_l1(pred_half, target_half)
            return loss_scale1 + 0.5 * loss_scale2
        return loss_scale1


class BimodalStrokeLoss(nn.Module):
    """
    Penalizes washed-out gray predictions on text strokes while preserving soft anti-aliased
    target edges. Forces stroke cores to be crisp dark ink rather than muddy mid-gray.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        # 1. Bimodal distribution alignment: match p * (1 - p) of the clean target
        pred_bimodal = pred * (1.0 - pred)
        target_bimodal = target * (1.0 - target)
        loss_bimodal = F.l1_loss(pred_bimodal, target_bimodal)

        # 2. Asymmetric stroke darkening penalty: penalize faded/gray predictions (pred > target) on stroke pixels
        stroke_mask = (target < 0.65).float()
        faded_stroke_penalty = torch.mean(stroke_mask * (F.relu(pred - target) ** 2))

        return loss_bimodal + 2.0 * faded_stroke_penalty


class SSIMLoss(nn.Module):
    """Differentiable SSIM Loss (1 - SSIM) using an 11x11 Gaussian window."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window_2d)
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        channels = pred.shape[1]
        win = self.window.expand(channels, 1, self.window_size, self.window_size)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, win, padding=pad, groups=channels)
        mu2 = F.conv2d(target, win, padding=pad, groups=channels)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, win, padding=pad, groups=channels) - mu1_mu2

        ssim_map = ((2.0 * mu1_mu2 + self.c1) * (2.0 * sigma12 + self.c2)) / (
            (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2) + 1e-8
        )
        return torch.clamp(1.0 - ssim_map.mean(), min=0.0)


class BleedThroughSuppressionLoss(nn.Module):
    """
    Penalizes residual gray predictions (pred < 0.92) in regions where target > 0.95
    (pure white background or back-side bleed-through/watermark regions that must be erased).
    Teaches U-Net to distinguish and suppress back-side bleed-through without affecting front strokes.
    """

    def __init__(self, target_thresh: float = 0.95, pred_floor: float = 0.92):
        super().__init__()
        self.target_thresh = target_thresh
        self.pred_floor = pred_floor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        bg_or_bleed_mask = (target > self.target_thresh).float()
        # Heavy quadratic penalty when U-Net predicts gray < 0.92 on clean background/bleed-through regions
        below_floor = F.relu(self.pred_floor - pred)
        white_gap = F.relu(target - pred)
        return torch.mean(bg_or_bleed_mask * (10.0 * (below_floor ** 2) + 0.5 * white_gap))


class DocumentCleanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelLoss()
        self.ssim = SSIMLoss()
        self.bimodal = BimodalStrokeLoss()
        self.bleed_suppression = BleedThroughSuppressionLoss(target_thresh=0.95, pred_floor=0.92)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        # Trọng số liên tục ưu tiên bảo vệ nét chữ (không ép 1.5x cho nền trắng để tránh over-bleaching)
        stroke_weight = 1.0 + 6.0 * ((1.0 - target) ** 2)

        # 1. Weighted Charbonnier Loss
        charbonnier = torch.sqrt((pred - target) ** 2 + 1e-6)
        loss_charbonnier = torch.mean(stroke_weight * charbonnier)

        # 2. Weighted BCE Loss
        pred_clamped = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        bce_map = F.binary_cross_entropy(pred_clamped, target, reduction="none")
        loss_bce = torch.mean(stroke_weight * bce_map)

        # 3. Multi-Scale Sobel Edge Gradient Loss (0.8) để tái tạo sắc nét biên chữ khi mờ hoặc trong bóng râm
        loss_sobel = self.sobel(pred, target)

        # 4. SSIM Loss (0.2) bảo toàn cấu trúc cục bộ của ký tự
        loss_ssim = self.ssim(pred, target)

        # 5. BimodalStrokeLoss (0.25) phạt vùng chữ bị xám mờ, tăng tương phản nét chữ cho YOLOv8
        loss_bimodal = self.bimodal(pred, target)

        # 6. BleedThroughSuppressionLoss (0.35) phạt nặng nếu vùng target > 0.95 mà U-Net xuất < 0.92
        loss_bleed = self.bleed_suppression(pred, target)

        total_loss = (
            loss_charbonnier
            + 0.3 * loss_bce
            + 0.8 * loss_sobel
            + 0.2 * loss_ssim
            + 0.25 * loss_bimodal
            + 0.35 * loss_bleed
        )
        return total_loss


class DocumentRestorationUNet(nn.Module):
    """
    3-Stage U-Net architecture for Multi-Degradation Document Image Restoration.
    Accepts [B, 1, H, W] input, computes internal Local Illumination Normalization `x_norm`,
    concatenates `[x, x_norm]` into 2 channels for `enc1`, and uses Sigmoid output activation
    with +0.2 bias init to preserve faint/blurred text strokes.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        enc_in_channels = in_channels * 2

        # Encoder level 1 (receives concatenated [x, x_norm] -> 2 channels when in_channels=1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(enc_in_channels, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
        )
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        # Encoder level 2
        self.enc2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Encoder level 3
        self.enc3 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.down3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # Bottleneck (512 channels at H/8, W/8)
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
        )

        # Decoder level 3: upsample 512->256, concat with enc3 skip (256) = 512 channels
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1),  # 1x1 conv: 512 -> 256
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Decoder level 2: upsample 256->128, concat with enc2 skip (128) = 256 channels
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=1),  # 1x1 conv: 256 -> 128
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Decoder level 1: upsample 128->64, concat with enc1 skip (64) = 128 channels
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1),  # 1x1 conv: 128 -> 64
            ResidualBlock(64),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Direct clean output in [0, 1]
        )

        # Hạ khởi tạo bias của tầng Conv2d cuối cùng xuống +0.2 để tránh tẩy trắng nhầm chữ bị mờ
        last_conv = self.dec1[-2]  # Lớp nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        if hasattr(last_conv, "bias") and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, 0.2)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Allow seamless loading of legacy 1-channel enc1.0.weight checkpoints if needed."""
        w_key = prefix + "enc1.0.weight"
        if w_key in state_dict:
            w_val = state_dict[w_key]
            expected_shape = self.enc1[0].weight.shape
            if w_val.ndim == 4 and w_val.shape[1] == 1 and expected_shape[1] == 2:
                zeros = torch.zeros_like(w_val)
                state_dict[w_key] = torch.cat([w_val, zeros], dim=1)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local Illumination Normalization (tương phản cục bộ giúp lộ rõ chữ trong bóng râm ShadowCast)
        bg = F.avg_pool2d(x, kernel_size=31, stride=1, padding=15)
        x_norm = torch.clamp(x / (bg + 1e-4), 0.0, 1.0)
        x_in = torch.cat([x, x_norm], dim=1)  # [B, 2, H, W]

        # Encoder
        e1 = self.enc1(x_in)     # [B, 64, H, W]
        d1 = self.down1(e1)      # [B, 128, H/2, W/2]

        e2 = self.enc2(d1)       # [B, 128, H/2, W/2]
        d2 = self.down2(e2)      # [B, 256, H/4, W/4]

        e3 = self.enc3(d2)       # [B, 256, H/4, W/4]
        d3 = self.down3(e3)      # [B, 512, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(d3)  # [B, 512, H/8, W/8]

        # Decoder level 3
        u3 = self.up3(b)                 # [B, 256, H/4, W/4]
        c3 = torch.cat([u3, e3], dim=1)  # [B, 512, H/4, W/4]
        d_out3 = self.dec3(c3)           # [B, 256, H/4, W/4]

        # Decoder level 2
        u2 = self.up2(d_out3)            # [B, 128, H/2, W/2]
        c2 = torch.cat([u2, e2], dim=1)  # [B, 256, H/2, W/2]
        d_out2 = self.dec2(c2)           # [B, 128, H/2, W/2]

        # Decoder level 1
        u1 = self.up1(d_out2)            # [B, 64, H, W]
        c1 = torch.cat([u1, e1], dim=1)  # [B, 128, H, W]
        out = self.dec1(c1)              # [B, out_channels, H, W]

        return out


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two tensors with values in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def train_restoration_model(
    train_csv: str = "datasets/metadata/train.csv",
    val_csv: str = "datasets/metadata/val.csv",
    output_dir: str = "models",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    image_size: int = 512,
    manifest: str = None,
    dataset_root: str = None,
    resume_weights: str = "models/restoration_best.pth",
):
    """Train or Fine-Tune Document Restoration UNet with validation and best checkpoint saving.

    If `manifest` is given (datasets/metadata/dataset_manifest.csv), train/val pairs are read from it
    (many inputs -> one target) and `train_csv`/`val_csv` are ignored.
    If `resume_weights` exists on disk, warm-starts the U-Net from those pretrained weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Document Restoration Training on device: {device}")

    # Create datasets
    img_size = (image_size, image_size)
    if manifest:
        train_csv = val_csv = manifest
        train_dataset = DocumentRestorationDataset(manifest, image_size=img_size, split="train", is_train=True, root_dir=dataset_root)
        val_dataset = DocumentRestorationDataset(manifest, image_size=img_size, split="val", is_train=False, root_dir=dataset_root)
    else:
        train_dataset = DocumentRestorationDataset(metadata_csv=train_csv, image_size=img_size, split="train", is_train=True)
        val_dataset = DocumentRestorationDataset(metadata_csv=val_csv, image_size=img_size, split="val", is_train=False)

        # Automatic fallback to manifest if default train_csv has no valid image pairs on disk
        if ("datasets" in train_csv and "train.csv" in train_csv) and len(train_dataset) == 0:
            manifest_fallback = "datasets/metadata/dataset_manifest.csv"
            if os.path.exists(manifest_fallback):
                logger.info(f"No valid image pairs in '{train_csv}', falling back to manifest: {manifest_fallback}")
                train_dataset = DocumentRestorationDataset(manifest_fallback, image_size=img_size, split="train", is_train=True, root_dir=dataset_root)
                val_dataset = DocumentRestorationDataset(manifest_fallback, image_size=img_size, split="val", is_train=False, root_dir=dataset_root)

    if len(train_dataset) == 0:
        logger.error(f"No training pairs found in '{train_csv}'. Training aborted.")
        raise ValueError(f"No training pairs found in '{train_csv}'. Training aborted.")
    if len(val_dataset) == 0:
        logger.warning(f"No validation pairs found in '{val_csv}'. Will train without validation.")

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = DocumentRestorationUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    if resume_weights and os.path.exists(resume_weights):
        state_dict = torch.load(resume_weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Warm-Start (Fine-Tune): Loaded pretrained weights from '{resume_weights}'")

    # Loss and optimizer: Multi-purpose DocumentCleanLoss (Weighted Charbonnier + BCE + Sobel + SSIM + Bimodal + BleedThroughSuppression)
    criterion = DocumentCleanLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Checkpoint directory
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "restoration_best.pth")
    last_model_path = os.path.join(output_dir, "restoration_last.pth")

    best_val_psnr = -float("inf")
    training_log = []

    for epoch in range(1, epochs + 1):
        # ======================== TRAIN ========================
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        train_steps = 0

        for deg, clean in train_loader:
            deg, clean = deg.to(device), clean.to(device)
            optimizer.zero_grad()

            pred = model(deg)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                train_psnr += calculate_psnr(pred, clean)
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)
        avg_train_psnr = train_psnr / max(train_steps, 1)

        # ======================== VALIDATION ========================
        avg_val_loss = 0.0
        avg_val_psnr = 0.0

        if len(val_dataset) > 0:
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_steps = 0

            with torch.no_grad():
                for deg, clean in val_loader:
                    deg, clean = deg.to(device), clean.to(device)
                    pred = model(deg)
                    loss = criterion(pred, clean)
                    val_loss += loss.item()
                    val_psnr += calculate_psnr(pred, clean)
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            avg_val_psnr = val_psnr / max(val_steps, 1)

        # Log
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "train_psnr": round(avg_train_psnr, 2),
            "val_loss": round(avg_val_loss, 6),
            "val_psnr": round(avg_val_psnr, 2),
        }
        training_log.append(epoch_info)

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.2f} dB | "
            f"Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB"
        )

        # Save last checkpoint
        torch.save(model.state_dict(), last_model_path)

        # Save best checkpoint based on validation PSNR
        if avg_val_psnr > best_val_psnr and len(val_dataset) > 0:
            best_val_psnr = avg_val_psnr
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best model saved (Val PSNR: {best_val_psnr:.2f} dB)")

    # If no validation data, save last as best
    if len(val_dataset) == 0:
        torch.save(model.state_dict(), best_model_path)
        logger.info("No validation data. Saved last model as best.")

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    val_psnr_record = round(best_val_psnr, 2) if best_val_psnr != -float("inf") else None
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "image_size": image_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "best_val_psnr": val_psnr_record,
            "log": training_log,
        }, f, indent=2)

    logger.info(f"Training complete. Best Val PSNR: {val_psnr_record if val_psnr_record is not None else 'N/A'} dB")
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"Last model: {last_model_path}")
    logger.info(f"Training log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Document Image Restoration UNet")
    parser.add_argument("--train-csv", default="datasets/metadata/train.csv",
                        help="Path to training metadata CSV")
    parser.add_argument("--val-csv", default="datasets/metadata/val.csv",
                        help="Path to validation metadata CSV")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Target image size (square)")
    parser.add_argument("--manifest", default=None,
                        help="dataset_manifest.csv (many inputs -> one target); overrides --train-csv/--val-csv")
    parser.add_argument("--dataset-root", default=None,
                        help="Root that manifest paths are relative to (default: datasets/)")
    parser.add_argument("--resume-weights", default="models/restoration_best.pth",
                        help="Path to pretrained weights for warm-start fine-tuning")
    args = parser.parse_args()

    try:
        train_restoration_model(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            manifest=args.manifest,
            dataset_root=args.dataset_root,
            resume_weights=args.resume_weights,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
