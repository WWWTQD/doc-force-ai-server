# -*- coding: utf-8 -*-
"""
Huấn luyện mạng nơ-ron sâu nhận diện hướng trang tài liệu tiếng Việt (0°, 90°, 180°, 270°)
dành cho hệ thống scan ảnh chụp trực tiếp từ camera (`DocumentOrientationResNet`).

Đặc điểm thiết kế AI:
  - Dữ liệu huấn luyện 100% tổng hợp từ ngữ âm tiếng Việt ngẫu nhiên + 22 font TrueType thật
    + mô phỏng quang học camera (bóng gáy sách, ánh sáng lệch, nhòe tiêu cự, nghiêng ±2.8°, JPEG).
  - KHÔNG chứa bất kỳ câu chữ cố định hay dữ liệu trang tài liệu mẫu nào của người dùng.
  - Kiến trúc `DocumentOrientationResNet` (đầu vào 128x128) dùng pooling không gian 2x2
    (`AdaptiveAvgPool2d((2, 2))`) giữ thông tin bất đối xứng trên/dưới/trái/phải của dấu thanh,
    chữ hoa và dòng đầu đoạn, phân biệt chính xác 0° (xuôi) và 180° (lộn ngược).
  - Lưu trọng số tại `models/orientation_net.pth`.
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


class _OrientResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(x + out)


class DocumentOrientationResNet(nn.Module):
    """
    CNN phân loại 4 hướng đọc chuẩn của patch chữ tiếng Việt (1, 128, 128):
      0 -> 0° (Xuôi, đọc được bình thường)
      1 -> 90° CW needed
      2 -> 180° Upside down
      3 -> 270° CW needed
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            _OrientResBlock(64),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            _OrientResBlock(96),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(96 * 4, 96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(self.pool(x).flatten(1))


def render_upright_doc_patch(
    catalog: dict,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Sinh 1 ảnh patch (128x128 uint8) chứa 4-6 dòng chữ tiếng Việt xuôi với quang học camera."""
    canvas_size = rng.choice([224, 256, 300])
    bg_val = rng.randint(205, 248)
    pil_img = Image.new("L", (canvas_size, canvas_size), color=bg_val)
    draw = ImageDraw.Draw(pil_img)

    is_serif = rng.choice([0, 1])
    body_fonts = catalog[(is_serif, 0, 0)] or catalog[(1, 0, 0)]
    bold_fonts = catalog[(is_serif, 1, 0)] or body_fonts
    italic_fonts = catalog[(is_serif, 0, 1)] or body_fonts

    font_px = rng.randint(max(18, canvas_size // 14), max(24, canvas_size // 9))
    line_step = int(font_px * rng.uniform(1.18, 1.50))

    y = rng.randint(-font_px // 4, int(canvas_size * 0.14))
    has_header = rng.random() < 0.28
    first_line = True

    while y < canvas_size - font_px // 2:
        mode = "body"
        if first_line and has_header:
            mode = "heading"
        elif rng.random() < 0.14:
            mode = "italic"

        if mode == "heading":
            fpath = rng.choice(bold_fonts)
            cur_px = int(font_px * rng.uniform(1.05, 1.22))
            text = random_vietnamese_line(rng, min_words=2, max_words=6, allow_upper=True, allow_numbering=True)
        elif mode == "italic":
            fpath = rng.choice(italic_fonts)
            cur_px = font_px
            text = random_vietnamese_line(rng, min_words=3, max_words=8, allow_upper=False, allow_numbering=False)
        else:
            fpath = rng.choice(body_fonts)
            cur_px = font_px
            text = random_vietnamese_line(rng, min_words=4, max_words=10, allow_upper=True, allow_numbering=True)

        try:
            font = _get_font(fpath, cur_px)
        except Exception:
            font = ImageFont.load_default()

        x_start = rng.randint(4, max(8, canvas_size // 10))
        if first_line and rng.random() < 0.45 and mode != "heading":
            x_start += rng.randint(10, 26)

        ink_val = rng.randint(18, 82)
        draw.text((x_start, y), text, fill=ink_val, font=font)
        y += line_step
        first_line = False

    gray = np.array(pil_img, dtype=np.uint8)
    degraded = apply_camera_optics(gray, rng, np_rng, max_skew_deg=2.8)
    return cv2.resize(degraded, (128, 128), interpolation=cv2.INTER_AREA)


class OrientationPatchDataset(Dataset):
    def __init__(self, patches: List[np.ndarray], augment: bool = True, seed: int = 42):
        self.patches = patches
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.patches) * 4

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base = self.patches[idx // 4]
        k = idx % 4
        if k == 1:
            img = cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif k == 2:
            img = cv2.rotate(base, cv2.ROTATE_180)
        elif k == 3:
            img = cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)
        else:
            img = base.copy()

        if self.augment:
            alpha = self.rng.uniform(0.82, 1.18)
            beta = self.rng.uniform(-16.0, 16.0)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if self.rng.random() < 0.25:
                img = cv2.GaussianBlur(img, (3, 3), sigmaX=self.rng.uniform(0.3, 0.7))

        tensor = torch.from_numpy((img.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0)
        return tensor, k


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    pair_0_180_total = 0
    pair_0_180_correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.size(0))

            mask_02 = (y == 0) | (y == 2)
            if mask_02.any():
                sub_logits = logits[mask_02]
                sub_y = y[mask_02]
                pred_02 = torch.where(sub_logits[:, 0] >= sub_logits[:, 2], 0, 2)
                pair_0_180_correct += int((pred_02 == sub_y).sum().item())
                pair_0_180_total += int(sub_y.size(0))
    return (
        loss_sum / max(1, total),
        correct / max(1, total),
        pair_0_180_correct / max(1, pair_0_180_total),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bases", type=int, default=640)
    parser.add_argument("--val-bases", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=12)
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
    out_path = os.path.join(models_dir, "orientation_net.pth")

    catalog = discover_font_catalog(os.path.join(models_dir, "fonts"))
    font_count = sum(len(v) for v in catalog.values())
    print(f"[OrientationAI] Phát hiện {font_count} TrueType fonts trong {os.path.join(models_dir, 'fonts')}")

    t0 = time.time()
    train_rng = random.Random(20261116)
    train_np = np.random.default_rng(20261116)
    val_rng = random.Random(99887766)
    val_np = np.random.default_rng(99887766)

    print(f"[OrientationAI] Đang tổng hợp {args.train_bases} train patches + {args.val_bases} val patches từ camera simulator...")
    train_patches = [render_upright_doc_patch(catalog, train_rng, train_np) for _ in range(args.train_bases)]
    val_patches = [render_upright_doc_patch(catalog, val_rng, val_np) for _ in range(args.val_bases)]
    print(f"[OrientationAI] Sinh dữ liệu hoàn tất trong {time.time() - t0:.2f}s "
          f"({len(train_patches)*4} mẫu train, {len(val_patches)*4} mẫu val).")

    train_ds = OrientationPatchDataset(train_patches, augment=True, seed=11)
    val_ds = OrientationPatchDataset(val_patches, augment=False, seed=22)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentOrientationResNet(num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.04)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * y.size(0)
            tr_correct += int((logits.argmax(dim=1) == y).sum().item())
            tr_total += int(y.size(0))
        scheduler.step()

        val_loss, val_acc, pair_02_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} ({time.time()-ep_t0:.1f}s) | "
            f"train_loss={tr_loss/tr_total:.4f} train_acc={tr_correct/tr_total:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} (0_vs_180={pair_02_acc:.4f})"
        )
        score = val_acc * 0.6 + pair_02_acc * 0.4
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "pair_0_180_acc": pair_02_acc,
                    "epoch": epoch,
                    "input_size": 128,
                },
                out_path,
            )
    print(f"[OrientationAI] Đã lưu model tốt nhất vào {out_path} (best_score={best_score:.4f}).")


if __name__ == "__main__":
    main()
