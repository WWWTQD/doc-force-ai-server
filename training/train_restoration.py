import os
import sys
import argparse
import math
import json
import logging
import torch
import torch.nn as nn
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
    """Sobel Gradient Loss to enforce sharp text edge reconstruction."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Grayscale conversion: 0.299 R + 0.587 G + 0.114 B
        pred_g = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_g = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]

        pred_gx = nn.functional.conv2d(pred_g, self.sobel_x, padding=1)
        pred_gy = nn.functional.conv2d(pred_g, self.sobel_y, padding=1)
        target_gx = nn.functional.conv2d(target_g, self.sobel_x, padding=1)
        target_gy = nn.functional.conv2d(target_g, self.sobel_y, padding=1)

        loss_gx = nn.functional.l1_loss(pred_gx, target_gx)
        loss_gy = nn.functional.l1_loss(pred_gy, target_gy)
        return loss_gx + loss_gy


class DocumentRestorationUNet(nn.Module):
    """
    Lightweight Residual U-Net architecture for Document Image Restoration.
    Uses Global Residual Connection: output = clamp(x + residual, 0, 1)
    predicting edge/contrast corrections rather than learning from scratch.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
        )
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
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
            nn.Tanh(),  # Residual delta output [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)        # [B, 64, H, W]
        d1 = self.down1(e1)      # [B, 128, H/2, W/2]

        e2 = self.enc2(d1)       # [B, 128, H/2, W/2]
        d2 = self.down2(e2)      # [B, 256, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(d2)  # [B, 256, H/4, W/4]

        # Decoder level 2
        u2 = self.up2(b)         # [B, 128, H/2, W/2]
        c2 = torch.cat([u2, e2], dim=1)  # [B, 256, H/2, W/2]
        d_out2 = self.dec2(c2)   # [B, 128, H/2, W/2]

        # Decoder level 1
        u1 = self.up1(d_out2)    # [B, 64, H, W]
        c1 = torch.cat([u1, e1], dim=1)  # [B, 128, H, W]
        res = self.dec1(c1)      # [B, 3, H, W] residual map

        # Global residual connection: add residual to input image and clamp
        out = torch.clamp(x + res, 0.0, 1.0)
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
):
    """Train Document Restoration UNet with validation and best checkpoint saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Document Restoration Training on device: {device}")

    # Create datasets
    img_size = (image_size, image_size)
    train_dataset = DocumentRestorationDataset(metadata_csv=train_csv, image_size=img_size)
    val_dataset = DocumentRestorationDataset(metadata_csv=val_csv, image_size=img_size)

    if len(train_dataset) == 0:
        logger.error(f"No training pairs found in '{train_csv}'. Training aborted.")
        return
    if len(val_dataset) == 0:
        logger.warning(f"No validation pairs found in '{val_csv}'. Will train without validation.")

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = DocumentRestorationUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_sobel = SobelLoss().to(device)
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
            loss_l1 = criterion_l1(pred, clean)
            loss_mse = criterion_mse(pred, clean)
            loss_sobel = criterion_sobel(pred, clean)

            # Combined loss: L1 + 0.5*MSE + 2.0*Sobel (Sobel forces sharp text edges)
            loss = loss_l1 + 0.5 * loss_mse + 2.0 * loss_sobel
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
                    loss = criterion_l1(pred, clean) + 0.5 * criterion_mse(pred, clean) + 2.0 * criterion_sobel(pred, clean)
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
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "image_size": image_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "best_val_psnr": round(best_val_psnr, 2),
            "log": training_log,
        }, f, indent=2)

    logger.info(f"Training complete. Best Val PSNR: {best_val_psnr:.2f} dB")
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
    args = parser.parse_args()

    train_restoration_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
    )
