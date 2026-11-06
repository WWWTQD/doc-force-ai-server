"""
Document Image Restoration Model Training Script (U-Net v5).

Architecture (DocumentRestorationUNet v5):
- Multi-Scale Robust Input (3 channels): [x, x_norm_robust_31, x_norm_robust_101] with 3x3 median pre-filtering
- Internal layers: GroupNorm (8 groups) throughout, replacing BatchNorm to eliminate tile brightness shifts
- Bilinear Upsampling + Conv3x3 (up3, up2, up1) replacing ConvTranspose2d to eliminate checkerboard artifacts
- Bottleneck: 1 ResidualBlock(512) followed by ASPP(512) with dilations 1, 2, 4, 8 + Global Average Pooling
- Selective Skip Connections: AttentionGate on levels 2 and 3; direct skip connection on level 1 to preserve 100% fine accents
- Logit-Space Residual Output Head: delta = out_conv(d1), output = sigmoid(logit(clamp(x_norm_robust_31)) + delta)
  with zero weight/bias initialization so training starts cleanly at the normalized state.

Loss (DocumentCleanLoss):
- Weighted Charbonnier Loss (1.0)
- Multi-Scale Sobel Edge Gradient Loss (0.8)
- SSIM Loss (0.2)
- Asymmetric Stroke Darkening Loss (0.25)
- Bleed-Through Suppression Loss (0.15)
- FFT Frequency L1 Loss (0.05)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.restoration_dataset import DocumentRestorationDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_restoration")


# ==============================================================================
# ARCHITECTURAL BLOCKS (U-Net v5)
# ==============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and GroupNorm (8 groups)."""

    def __init__(self, channels: int):
        super().__init__()
        num_groups = min(8, channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018):
    Filters background noise and back-side bleed-through before skip connections.
    F_g: channels of gating signal from decoder
    F_l: channels of skip connection from encoder
    F_int: intermediate channel dimension
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        num_groups = min(8, F_int)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-receptive field context in the bottleneck.
    Dilations: 1, 2, 4, 8 + Global Average Pooling.
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 512):
        super().__init__()
        branch_ch = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, branch_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(8, branch_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, branch_ch),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, branch_ch),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.GroupNorm(8, branch_ch),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, branch_ch),
            nn.ReLU(inplace=True),
        )
        total_in = branch_ch * 6
        self.project = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        f4 = self.conv4(x)
        f5 = self.conv5(x)
        gp = self.global_pool(x)
        gp_up = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        cat = torch.cat([f1, f2, f3, f4, f5, gp_up], dim=1)
        return self.project(cat)


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class SobelLoss(nn.Module):
    """Multi-Scale Sobel Gradient Loss to enforce sharp text edge reconstruction across scales."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
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


class AsymmetricStrokeDarkeningLoss(nn.Module):
    """
    Asymmetric stroke darkening penalty:
    Penalizes faded or gray predictions (pred > target) specifically on text stroke pixels
    where target < target_thresh (default 0.70). Does NOT penalize dark strokes.
    """

    def __init__(self, target_thresh: float = 0.70):
        super().__init__()
        self.target_thresh = target_thresh

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        stroke_mask = (target < self.target_thresh).float()
        faded_penalty = torch.mean(stroke_mask * (F.relu(pred - target) ** 2))
        return faded_penalty


class BleedThroughSuppressionLoss(nn.Module):
    """
    Penalizes residual gray predictions (pred < pred_floor) in regions where target > target_thresh
    (pure white background or back-side bleed-through/watermark regions that must be erased).
    """

    def __init__(self, target_thresh: float = 0.95, pred_floor: float = 0.92):
        super().__init__()
        self.target_thresh = target_thresh
        self.pred_floor = pred_floor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        bg_or_bleed_mask = (target > self.target_thresh).float()
        below_floor = F.relu(self.pred_floor - pred)
        white_gap = F.relu(target - pred)
        return torch.mean(bg_or_bleed_mask * (10.0 * (below_floor ** 2) + 0.5 * white_gap))


class FFTLoss(nn.Module):
    """Frequency-domain L1 loss using 2D Real FFT to preserve fine text texture and suppress noise grids."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)


class DocumentCleanLoss(nn.Module):
    """
    Comprehensive Multi-Task Loss for Document Restoration (v5):
    - Weighted Charbonnier Loss (1.0)
    - Multi-Scale Sobel Gradient Edge Loss (0.8)
    - Structural Similarity SSIM Loss (0.2)
    - Asymmetric Stroke Darkening Loss (0.25)
    - Bleed-Through Suppression Loss (0.15)
    - FFT Frequency L1 Loss (0.05)
    """

    def __init__(self):
        super().__init__()
        self.sobel = SobelLoss()
        self.ssim = SSIMLoss()
        self.stroke_darkening = AsymmetricStrokeDarkeningLoss(target_thresh=0.70)
        self.bleed_suppression = BleedThroughSuppressionLoss(target_thresh=0.95, pred_floor=0.92)
        self.fft_loss = FFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        # Weighted Charbonnier prioritizing stroke reconstruction
        stroke_weight = 1.0 + 6.0 * ((1.0 - target) ** 2)
        charbonnier = torch.sqrt((pred - target) ** 2 + 1e-6)
        loss_charbonnier = torch.mean(stroke_weight * charbonnier)

        loss_sobel = self.sobel(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_darkening = self.stroke_darkening(pred, target)
        loss_bleed = self.bleed_suppression(pred, target)
        loss_fft = self.fft_loss(pred, target)

        total_loss = (
            1.0 * loss_charbonnier
            + 0.8 * loss_sobel
            + 0.2 * loss_ssim
            + 0.25 * loss_darkening
            + 0.15 * loss_bleed
            + 0.05 * loss_fft
        )
        return total_loss


# ==============================================================================
# MODEL ARCHITECTURE: DocumentRestorationUNet (v5)
# ==============================================================================

class DocumentRestorationUNet(nn.Module):
    """
    DocumentRestorationUNet v5:
    - Multi-Scale Robust Input (3 channels): [x, x_norm_robust_31, x_norm_robust_101]
    - GroupNorm(8) throughout internal layers (no batch-size dependence or tile illumination shifts)
    - Bilinear Upsampling + Conv3x3 (no checkerboard artifacts)
    - Bottleneck: 1 ResidualBlock(512) + ASPP(512)
    - Selective Skip Connections: AttentionGate on levels 2 & 3; direct skip on level 1
    - Logit-Space Residual Output Head: delta = out_conv(d1), y = sigmoid(logit(x_norm_31) + delta)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 3-channel input: [x, x_norm_robust_31, x_norm_robust_101]
        enc_in_channels = in_channels * 3

        # Encoder level 1
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

        # Bottleneck: 1 ResidualBlock + ASPP
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ASPP(512, 512),
        )

        # Decoder level 3 (Attention Gate on e3)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
        )
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Decoder level 2 (Attention Gate on e2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
        )
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=1),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Decoder level 1 (Direct skip connection on e1 to preserve 100% fine accents)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1),
            ResidualBlock(64),
        )

        # Logit-space residual head
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _median_filter_3x3(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        patches = F.unfold(x_pad, kernel_size=3, stride=1)
        med = patches.median(dim=1).values.view(b, c, h, w)
        return med

    @staticmethod
    def _get_norm_robust(x_med: torch.Tensor, x_raw: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        x_pad = F.pad(x_med, (pad, pad, pad, pad), mode="reflect")
        bg_max = F.max_pool2d(
            F.max_pool2d(x_pad, kernel_size=(1, k), stride=1, padding=0),
            kernel_size=(k, 1),
            stride=1,
            padding=0,
        )
        bg_pad = F.pad(bg_max, (pad, pad, pad, pad), mode="reflect")
        bg_smooth = F.avg_pool2d(
            F.avg_pool2d(bg_pad, kernel_size=(1, k), stride=1, padding=0),
            kernel_size=(k, 1),
            stride=1,
            padding=0,
        )
        norm = torch.clamp(x_raw / (bg_smooth + 1e-4), 0.0, 1.0)
        return norm

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Allow seamless loading of legacy v4 checkpoints if encountered."""
        w_key = prefix + "enc1.0.weight"
        if w_key in state_dict:
            w_val = state_dict[w_key]
            expected_shape = self.enc1[0].weight.shape
            if w_val.ndim == 4 and w_val.shape[1] in (1, 2) and expected_shape[1] == 3:
                channels_needed = 3 - w_val.shape[1]
                zeros = torch.zeros(w_val.shape[0], channels_needed, w_val.shape[2], w_val.shape[3], device=w_val.device, dtype=w_val.dtype)
                state_dict[w_key] = torch.cat([w_val, zeros], dim=1)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure FP32 safe logit input (matches Colab training: float32 + [0.002, 0.998] range protection)
        x_safe = torch.nan_to_num(x.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0) * 0.996 + 0.002

        # Pre-filtering with 3x3 median filter
        x_med = self._median_filter_3x3(x_safe)

        # 2 scales: k=31 and k=101
        x_norm_31 = self._get_norm_robust(x_med, x_safe, 31)
        x_norm_101 = self._get_norm_robust(x_med, x_safe, 101)

        x_in = torch.cat([x_safe, x_norm_31, x_norm_101], dim=1)  # [B, 3, H, W]

        # Encoder
        e1 = self.enc1(x_in)    # [B, 64, H, W]
        d1 = self.down1(e1)     # [B, 128, H/2, W/2]

        e2 = self.enc2(d1)      # [B, 128, H/2, W/2]
        d2 = self.down2(e2)     # [B, 256, H/4, W/4]

        e3 = self.enc3(d2)      # [B, 256, H/4, W/4]
        d3 = self.down3(e3)     # [B, 512, H/8, W/8]

        # Bottleneck (ResBlock + ASPP)
        b = self.bottleneck(d3) # [B, 512, H/8, W/8]

        # Decoder level 3 (Attention Gate on e3)
        u3 = self.up3(b)
        e3_att = self.att3(g=u3, x=e3)
        c3 = torch.cat([u3, e3_att], dim=1)
        d_out3 = self.dec3(c3)

        # Decoder level 2 (Attention Gate on e2)
        u2 = self.up2(d_out3)
        e2_att = self.att2(g=u2, x=e2)
        c2 = torch.cat([u2, e2_att], dim=1)
        d_out2 = self.dec2(c2)

        # Decoder level 1 (Direct skip on e1)
        u1 = self.up1(d_out2)
        c1 = torch.cat([u1, e1], dim=1)
        d_out1 = self.dec1(c1)

        # Logit-space residual head (FP32 safe logit)
        delta = self.out_conv(d_out1).float()
        base_clamped = torch.clamp(x_norm_31.float(), 1e-4, 1.0 - 1e-4)
        base_logit = torch.logit(base_clamped)
        y_pred = torch.sigmoid(base_logit + delta)
        return torch.nan_to_num(y_pred.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two tensors with values in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train_restoration_model(
    train_csv: str = "datasets/metadata/train.csv",
    val_csv: str = "datasets/metadata/val.csv",
    output_dir: str = "models",
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 2e-4,
    image_size: int = 512,
    manifest: Optional[str] = None,
    dataset_root: Optional[str] = None,
    resume_weights: Optional[str] = None,
):
    """
    Train or Fine-Tune Document Restoration UNet v5 with validation and best checkpoint saving.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting Document Restoration UNet v5 Training on device: {device}")

    img_size = (image_size, image_size)
    if manifest:
        train_csv = val_csv = manifest
        train_dataset = DocumentRestorationDataset(manifest, image_size=img_size, split="train", is_train=True, root_dir=dataset_root)
        val_dataset = DocumentRestorationDataset(manifest, image_size=img_size, split="val", is_train=False, root_dir=dataset_root)
    else:
        train_dataset = DocumentRestorationDataset(metadata_csv=train_csv, image_size=img_size, split="train", is_train=True)
        val_dataset = DocumentRestorationDataset(metadata_csv=val_csv, image_size=img_size, split="val", is_train=False)

    if len(train_dataset) == 0:
        logger.error(f"No training pairs found in '{train_csv}'. Training aborted.")
        raise ValueError(f"No training pairs found in '{train_csv}'. Training aborted.")

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else []

    # Model
    model = DocumentRestorationUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"U-Net v5 Model parameters: {total_params:,}")

    if resume_weights and os.path.exists(resume_weights):
        try:
            state_dict = torch.load(resume_weights, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from '{resume_weights}'")
        except Exception as e:
            logger.warning(f"Could not load checkpoint '{resume_weights}' directly ({e}). Initializing fresh v5 weights.")

    criterion = DocumentCleanLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "restoration_best.pth")
    last_model_path = os.path.join(output_dir, "restoration_last.pth")

    best_val_psnr = -float("inf")
    training_log = []

    for epoch in range(1, epochs + 1):
        # TRAIN
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

        scheduler.step()
        avg_train_loss = train_loss / max(train_steps, 1)
        avg_train_psnr = train_psnr / max(train_steps, 1)

        # VALIDATION
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

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "train_psnr": round(avg_train_psnr, 2),
            "val_loss": round(avg_val_loss, 6),
            "val_psnr": round(avg_val_psnr, 2),
            "lr": round(scheduler.get_last_lr()[0], 7),
        }
        training_log.append(epoch_info)

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.2f} dB | "
            f"Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        torch.save(model.state_dict(), last_model_path)

        if avg_val_psnr > best_val_psnr and len(val_dataset) > 0:
            best_val_psnr = avg_val_psnr
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best model saved (Val PSNR: {best_val_psnr:.2f} dB)")

    if len(val_dataset) == 0:
        torch.save(model.state_dict(), best_model_path)
        logger.info("No validation data. Saved last model as best.")

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


def main():
    parser = argparse.ArgumentParser(description="Train Document Image Restoration UNet v5")
    parser.add_argument("--train-csv", default="datasets/metadata/train.csv")
    parser.add_argument("--val-csv", default="datasets/metadata/val.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--resume-weights", default=None)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
