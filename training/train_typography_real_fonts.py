# -*- coding: utf-8 -*-
"""
Huấn luyện lại DeepDocumentTypographyResNet trên 22 font TrueType thật
(Times New Roman, Georgia, Arial, Calibri, Tahoma, Verdana - đủ Regular/Bold/Italic/Bold-Italic)
kết hợp mô phỏng quang học camera chụp tài liệu (`camera_text_generator.py`).

Thay thế hoàn toàn dữ liệu vector OpenCV cũ; tuyệt đối không dùng câu cố định.
Lưu checkpoint tương thích 100% với `ocr_engine.py` tại `models/typography_real_best.pth`.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_text_generator import (
    apply_camera_optics,
    discover_font_catalog,
    random_vietnamese_line,
)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class DeepDocumentTypographyResNet(nn.Module):
    """Kiến trúc khớp hoàn toàn với `ocr_engine.py`."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res1 = _ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res2 = _ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.res3 = _ResidualBlock(128)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.res4 = _ResidualBlock(256)
        self.res5 = _ResidualBlock(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)

        self.head_bold = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self.head_italic = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self.head_serif = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )
        self.head_size = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        x = self.res1(self.stem(x))
        x = self.res2(self.down1(x))
        x = self.res3(self.down2(x))
        x = self.res5(self.res4(self.down3(x)))
        feat = self.dropout(self.global_pool(x).flatten(1))
        return (
            self.head_bold(feat),
            self.head_italic(feat),
            self.head_serif(feat),
            self.head_size(feat).squeeze(-1),
        )


def preprocess_crop_for_typo(crop: np.ndarray) -> np.ndarray:
    """Tiền xử lý giống hệt `_infer_typography` trong `ocr_engine.py`."""
    if crop.size == 0:
        return np.full((32, 128), 255, dtype=np.uint8)
    ch, cw = crop.shape[:2]
    scale = min(28.0 / max(1, ch), 120.0 / max(1, cw))
    nw, nh = max(4, int(cw * scale)), max(4, int(ch * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((32, 128), 255, dtype=np.uint8)
    oy, ox = (32 - nh) // 2, (128 - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def synthesize_typo_sample(
    catalog: dict,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[np.ndarray, int, int, int, float]:
    keys = [k for k, paths in catalog.items() if paths]
    is_serif, is_bold, is_italic = rng.choice(keys)
    fpath = rng.choice(catalog[(is_serif, is_bold, is_italic)])

    size_pt = rng.choice([11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0, 16.0, 17.0, 18.0])
    # Mô phỏng ảnh chụp tài liệu ~150-220 DPI => font_px tỷ lệ thuận nhẹ với size_pt
    dpi_factor = rng.uniform(1.6, 2.2)
    font_px = max(18, int(round(size_pt * dpi_factor)))

    try:
        font = _get_font(fpath, font_px)
    except Exception:
        font = ImageFont.load_default()

    # Một đoạn văn ngắn 2-6 từ (giống crop DBNet 1 dòng hoặc cụm từ tiêu đề)
    n_words = rng.randint(2, 6)
    text = random_vietnamese_line(
        rng,
        min_words=n_words,
        max_words=max(n_words, n_words + 1),
        allow_upper=(size_pt >= 14.0 or rng.random() < 0.2),
        allow_numbering=(rng.random() < 0.25),
    )
    if len(text) > 36:
        text = text[:36].strip()

    dummy = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = max(24, bbox[2] - bbox[0])
    th = max(18, bbox[3] - bbox[1])

    pad_x = rng.randint(4, 14)
    pad_y = rng.randint(3, 10)
    w = tw + pad_x * 2
    h = th + pad_y * 2
    bg = rng.randint(195, 248)
    ink = rng.randint(18, 85)

    img = Image.new("L", (w, h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink, font=font)

    gray = apply_camera_optics(np.array(img, dtype=np.uint8), rng, np_rng, max_skew_deg=2.0)

    # Giả lập crop lệch nhỏ giống DBNet
    cy1 = rng.randint(0, min(3, max(0, gray.shape[0] - 12)))
    cy2 = gray.shape[0] - rng.randint(0, min(3, max(0, gray.shape[0] - cy1 - 12)))
    crop = gray[cy1:cy2, :]
    canvas = preprocess_crop_for_typo(crop)
    return canvas, is_bold, is_italic, is_serif, float(size_pt)


class TypographyDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int, int, int, float]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        canvas, b, it, sf, sz = self.samples[idx]
        x = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        return (
            x,
            torch.tensor(b, dtype=torch.long),
            torch.tensor(it, dtype=torch.long),
            torch.tensor(sf, dtype=torch.long),
            torch.tensor(sz, dtype=torch.float32),
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    tot = 0
    ok_b = ok_i = ok_s = 0
    mae_sz = 0.0
    with torch.no_grad():
        for x, b, it, sf, sz in loader:
            x = x.to(device)
            pb, pi, ps, psz = model(x)
            ok_b += int((pb.argmax(1).cpu() == b).sum().item())
            ok_i += int((pi.argmax(1).cpu() == it).sum().item())
            ok_s += int((ps.argmax(1).cpu() == sf).sum().item())
            mae_sz += float(torch.abs(psz.cpu() - sz).sum().item())
            tot += int(x.size(0))
    return ok_b / tot, ok_i / tot, ok_s / tot, mae_sz / tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=1600)
    parser.add_argument("--val-samples", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    args = parser.parse_args()

    try:
        torch.set_num_threads(4)
    except Exception:
        pass

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "typography_real_best.pth")

    catalog = discover_font_catalog(os.path.join(models_dir, "fonts"))
    print(f"[TypographyAI] Số lượng TrueType fonts khả dụng: {sum(len(v) for v in catalog.values())}")

    t0 = time.time()
    tr_rng = random.Random(20261116)
    tr_np = np.random.default_rng(20261116)
    va_rng = random.Random(77665544)
    va_np = np.random.default_rng(77665544)

    train_samples = [synthesize_typo_sample(catalog, tr_rng, tr_np) for _ in range(args.train_samples)]
    val_samples = [synthesize_typo_sample(catalog, va_rng, va_np) for _ in range(args.val_samples)]
    print(f"[TypographyAI] Đã sinh {len(train_samples)} mẫu train + {len(val_samples)} mẫu val trong {time.time()-t0:.2f}s.")

    train_loader = DataLoader(TypographyDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TypographyDataset(val_samples), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDocumentTypographyResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_avg_acc = -1.0
    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        loss_sum = 0.0
        n_tot = 0
        for x, b, it, sf, sz in train_loader:
            x = x.to(device)
            b, it, sf, sz = b.to(device), it.to(device), sf.to(device), sz.to(device)
            optimizer.zero_grad()
            pb, pi, ps, psz = model(x)
            loss = (
                F.cross_entropy(pb, b, label_smoothing=0.03)
                + F.cross_entropy(pi, it, label_smoothing=0.03)
                + F.cross_entropy(ps, sf, label_smoothing=0.03)
                + 0.15 * F.smooth_l1_loss(psz, sz)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * x.size(0)
            n_tot += int(x.size(0))
        scheduler.step()

        ab, ai, asf, mae = evaluate(model, val_loader, device)
        avg_acc = (ab + ai + asf) / 3.0
        print(
            f"Epoch {ep:02d}/{args.epochs} ({time.time()-ep_t0:.1f}s) | loss={loss_sum/n_tot:.4f} | "
            f"val Bold={ab:.4f} Italic={ai:.4f} Serif={asf:.4f} SizeMAE={mae:.2f}pt"
        )
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": ep,
                    "val_bold_acc": ab,
                    "val_italic_acc": ai,
                    "val_serif_acc": asf,
                    "val_size_mae": mae,
                },
                out_path,
            )
    print(f"[TypographyAI] Hoàn tất! Đã lưu model vào {out_path} (avg_cls_acc={best_avg_acc:.4f}).")


if __name__ == "__main__":
    main()
