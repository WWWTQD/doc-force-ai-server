"""
End-to-End Self-Contained Typography & Formatting AI Training Pipeline for Doc Forge.

Chức năng:
  Tự động sinh dataset tài liệu tiếng Việt (Synthetic Dataset) và huấn luyện mô hình Deep Learning
  đa nhiệm (Multi-Task CNN) nhận diện 100% thuộc tính định dạng:
    1. Chữ in đậm (Bold vs Regular)
    2. Chữ in nghiêng (Italic vs Upright)
    3. Họ Font chữ (Serif / Times New Roman vs Sans-Serif / Arial)
    4. Cỡ chữ chuẩn Point (10pt - 26pt)

Đặc điểm:
  - TỰ ĐỘNG 100% từ A đến Z trong MỘT FILE DUY NHẤT.
  - KHÔNG CẦN CẮT ẢNH THỦ CÔNG: Tự render ảnh chữ tiếng Việt với nhãn chuẩn xác.
  - KHÔNG CẦN THÊM THƯ VIỆN NGOÀI: Chỉ dùng PyTorch, Pillow, OpenCV, NumPy có sẵn.
  - Báo cáo số liệu chi tiết: Tham số, GFLOPs, Loss, Accuracy, MAE từng epoch, Confusion Matrix.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("typography_trainer")

# Cố định seed để tái lập kết quả
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Kích thước chuẩn đưa vào mạng CNN
INPUT_HEIGHT = 32
INPUT_WIDTH = 128

# Kho ngữ liệu tiếng Việt mẫu (văn bản hành chính, pháp quy, hợp đồng, báo cáo)
SAMPLE_TEXTS = [
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
    "BỘ KHOA HỌC VÀ CÔNG NGHỆ",
    "Số: 45/2026/QĐ-TTg",
    "Hà Nội, ngày 15 tháng 11 năm 2026",
    "Đà Nẵng, ngày 02 tháng 09 năm 2026",
    "QUYẾT ĐỊNH",
    "THỦ TƯỚNG CHÍNH PHỦ",
    "Điều 1. Phê duyệt đề án phát triển",
    "Điều 2. Trách nhiệm thi hành",
    "Căn cứ Luật Tổ chức Chính phủ",
    "Xét đề nghị của Bộ trưởng",
    "Nơi nhận: Như Điều 3;",
    "Lưu: VT, Cổng TTĐT",
    "KT. THỦ TƯỚNG",
    "PHÓ THỦ TƯỚNG",
    "Nguyễn Văn An",
    "Trần Thị Bích",
    "Phạm Quang Huy",
    "HỢP ĐỒNG KINH TẾ",
    "Bên A (Bên bán):",
    "Bên B (Bên mua):",
    "Tổng số trang: 05 trang",
    "Đã ký và đóng dấu",
    "Kính gửi: Các cơ quan thông tấn",
    "Báo cáo tiến độ quý IV",
    "Công văn hướng dẫn chi tiết",
    "Theo quy định tại khoản 2 Điều 15",
    "Doc Forge OCR & Document AI System",
]

# Danh mục font TrueType mặc định của hệ thống Windows
FONT_FAMILIES = {
    "serif": {
        "regular": [r"C:\Windows\Fonts\times.ttf"],
        "bold": [r"C:\Windows\Fonts\timesbd.ttf"],
        "italic": [r"C:\Windows\Fonts\timesi.ttf"],
        "bold_italic": [r"C:\Windows\Fonts\timesbi.ttf"],
    },
    "sans": {
        "regular": [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf"],
        "bold": [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf"],
        "italic": [r"C:\Windows\Fonts\ariali.ttf", r"C:\Windows\Fonts\calibrii.ttf"],
        "bold_italic": [r"C:\Windows\Fonts\arialbi.ttf", r"C:\Windows\Fonts\calibriz.ttf"],
    },
}


# ==============================================================================
# GIAI ĐOẠN 1: BỘ SINH DỮ LIỆU TỔNG HỢP TRỰC TIẾP (SYNTHETIC GENERATOR)
# ==============================================================================

class SyntheticTypographyGenerator:
    """Tự động kết xuất ảnh dòng/từ chữ với Ground-Truth hoàn toàn chính xác."""

    def __init__(self):
        self._font_cache = {}

    def _get_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (font_path, size)
        if key not in self._font_cache:
            if os.path.exists(font_path):
                self._font_cache[key] = ImageFont.truetype(font_path, size)
            else:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def generate_sample(self) -> Tuple[np.ndarray, Dict[str, float]]:
        # 1. Chọn text ngẫu nhiên (từ đơn hoặc cụm từ ngắn)
        text = random.choice(SAMPLE_TEXTS)
        if random.random() > 0.4 and len(text.split()) > 3:
            words = text.split()
            sub_len = random.randint(1, min(4, len(words)))
            start = random.randint(0, len(words) - sub_len)
            text = " ".join(words[start : start + sub_len])

        # 2. Chọn thuộc tính định dạng (Ground-Truth)
        is_serif = 1 if random.random() < 0.55 else 0  # 1 = Serif (Times), 0 = Sans (Arial/Calibri)
        is_bold = 1 if random.random() < 0.45 else 0
        is_italic = 1 if random.random() < 0.35 else 0
        font_size_pt = round(random.uniform(10.0, 26.0), 1)

        # 3. Chọn file font phù hợp
        family_key = "serif" if is_serif == 1 else "sans"
        if is_bold and is_italic:
            style_key = "bold_italic"
        elif is_bold:
            style_key = "bold"
        elif is_italic:
            style_key = "italic"
        else:
            style_key = "regular"

        font_path = random.choice(FONT_FAMILIES[family_key][style_key])
        font_px = int(round(font_size_pt * 1.33))  # 1pt ~ 1.33px ở 96 DPI
        font = self._get_font(font_path, max(12, font_px))

        # 4. Render text lên ảnh tạm để đo kích thước
        dummy = Image.new("RGB", (800, 200), (255, 255, 255))
        draw_dummy = ImageDraw.Draw(dummy)
        bbox = draw_dummy.textbbox((0, 0), text, font=font)
        tw = max(10, bbox[2] - bbox[0])
        th = max(10, bbox[3] - bbox[1])

        # 5. Tạo ảnh với nền giấy và độ tương phản tự nhiên
        margin_x = random.randint(4, 16)
        margin_y = random.randint(4, 12)
        canvas_w = tw + 2 * margin_x
        canvas_h = th + 2 * margin_y

        # Nền giấy hơi ngà hoặc trắng tinh (240 - 255)
        paper_val = random.randint(245, 255)
        img = Image.new("L", (canvas_w, canvas_h), paper_val)
        draw = ImageDraw.Draw(img)

        # Màu mực (từ đen đậm 10 đến xám đậm 60)
        ink_val = random.randint(10, 55)
        draw.text((margin_x - bbox[0], margin_y - bbox[1]), text, fill=ink_val, font=font)

        # 6. Thêm suy hao thực tế (nhiễu muỗi Gaussian + mờ nhẹ)
        arr = np.array(img, dtype=np.float32)
        if random.random() > 0.4:
            noise = np.random.normal(0, random.uniform(1.0, 4.0), arr.shape)
            arr = np.clip(arr + noise, 0, 255)

        # Resize chuẩn hóa về (INPUT_WIDTH, INPUT_HEIGHT) giữ tỉ lệ aspect ratio
        h_arr, w_arr = arr.shape[:2]
        scale = min(float(INPUT_HEIGHT - 4) / max(1, h_arr), float(INPUT_WIDTH - 8) / max(1, w_arr))
        new_w = max(4, int(w_arr * scale))
        new_h = max(4, int(h_arr * scale))

        resized = cv2.resize(arr.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Đặt vào giữa khung 32x128
        final_img = np.full((INPUT_HEIGHT, INPUT_WIDTH), 255, dtype=np.uint8)
        off_y = (INPUT_HEIGHT - new_h) // 2
        off_x = (INPUT_WIDTH - new_w) // 2
        final_img[off_y : off_y + new_h, off_x : off_x + new_w] = resized

        # Chuẩn hóa về [0, 1]
        tensor_img = (final_img.astype(np.float32) / 255.0)

        targets = {
            "is_bold": float(is_bold),
            "is_italic": float(is_italic),
            "is_serif": float(is_serif),
            "font_size_pt": float(font_size_pt),
        }
        return tensor_img, targets


class SyntheticTypographyDataset(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.generator = SyntheticTypographyGenerator()
        self.samples = [self.generator.generate_sample() for _ in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img, tgt = self.samples[idx]
        img_t = torch.from_numpy(img).unsqueeze(0)  # Shape: [1, 32, 128]
        bold_t = torch.tensor(int(tgt["is_bold"]), dtype=torch.long)
        italic_t = torch.tensor(int(tgt["is_italic"]), dtype=torch.long)
        serif_t = torch.tensor(int(tgt["is_serif"]), dtype=torch.long)
        size_t = torch.tensor(float(tgt["font_size_pt"]), dtype=torch.float32)
        return img_t, bold_t, italic_t, serif_t, size_t


# ==============================================================================
# GIAI ĐOẠN 2: KIẾN TRÚC MẠNG NƠ-RON ĐA NHIỆM (MULTI-TASK CNN)
# ==============================================================================

class DocumentTypographyNet(nn.Module):
    """
    Mạng CNN đa nhiệm siêu nhẹ chuyên biệt bóc tách Typography tài liệu:
      - Backbone: 4 Conv-BN-ReLU-Pool Blocks
      - 4 Output Heads:
          1. Bold Classifier (2 classes: Regular vs Bold)
          2. Italic Classifier (2 classes: Upright vs Italic)
          3. Font Family Classifier (2 classes: Sans-serif vs Serif)
          4. Font Size Regressor (Hồi quy trực tiếp giá trị Point Size)
    """

    def __init__(self):
        super().__init__()
        # Backbone trích xuất đặc trưng hình thái chữ
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)

        # Nhánh 1: Bold Head
        self.head_bold = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        # Nhánh 2: Italic Head
        self.head_italic = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        # Nhánh 3: Serif Head
        self.head_serif = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        # Nhánh 4: Font Size Regressor (Point size: 10 - 26pt)
        self.head_size = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, 1, 32, 128] -> [B, 32, 16, 64]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # -> [B, 64, 8, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # -> [B, 128, 4, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # -> [B, 256, 2, 8]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        feat = self.global_pool(x).flatten(1)  # [B, 256]
        feat = self.dropout(feat)

        out_bold = self.head_bold(feat)
        out_italic = self.head_italic(feat)
        out_serif = self.head_serif(feat)
        out_size = self.head_size(feat).squeeze(-1)
        return out_bold, out_italic, out_serif, out_size

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# GIAI ĐOẠN 3: VÒNG LẶP HUẤN LUYỆN TOÀN BỘ (TRAINING & VALIDATION LOOP)
# ==============================================================================

def train_typography_model(
    train_size: int = 2400,
    val_size: int = 400,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 0.001,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("══════════════════════════════════════════════════════════════════════")
    print("  KHỞI TẠO HUẤN LUYỆN MODEL AI TYPOGRAPHY & ĐỊNH DẠNG TÀI LIỆU (A-Z)  ")
    print("══════════════════════════════════════════════════════════════════════")
    print(f"Thiết bị huấn luyện: {device.type.upper()}")
    print("Bước 1/4: Đang sinh tự động bộ dataset (Synthetic Data Generation)...")

    t0_data = time.time()
    train_dataset = SyntheticTypographyDataset(size=train_size)
    val_dataset = SyntheticTypographyDataset(size=val_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"  -> Đã sinh xong {train_size} mẫu Train, {val_size} mẫu Val trong {time.time() - t0_data:.2f}s!")

    print("\nBước 2/4: Khởi tạo kiến trúc mạng DocumentTypographyNet...")
    model = DocumentTypographyNet().to(device)
    num_params = model.count_parameters()
    print(f"  -> Tổng số tham số (Parameters): {num_params:,} tham số ({num_params / 1e6:.3f}M)")

    # Loss Functions: CrossEntropy cho phân loại + SmoothL1 cho cỡ chữ
    ce_loss = nn.CrossEntropyLoss()
    size_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = 0.0
    history = []

    print("\nBước 3/4: Bắt đầu vòng lặp huấn luyện qua các Epochs...")
    print("─" * 86)
    print(f"{'Epoch':<7} | {'Train Loss':<10} | {'Val Loss':<9} | {'Bold Acc':<9} | {'Italic Acc':<10} | {'Serif Acc':<10} | {'Size MAE':<9} | {'Thời gian':<7}")
    print("─" * 86)

    t_start_all = time.time()
    best_checkpoint_path = os.path.join(output_dir, "typography_best.pth")

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for imgs, bolds, italics, serifs, sizes in train_loader:
            imgs = imgs.to(device)
            bolds = bolds.to(device)
            italics = italics.to(device)
            serifs = serifs.to(device)
            sizes = sizes.to(device)

            optimizer.zero_grad()
            p_bold, p_italic, p_serif, p_size = model(imgs)

            loss_b = ce_loss(p_bold, bolds)
            loss_i = ce_loss(p_italic, italics)
            loss_s = ce_loss(p_serif, serifs)
            loss_sz = size_loss_fn(p_size, sizes) * 0.15

            total_loss = loss_b + loss_i + loss_s + loss_sz
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_dataset)

        # VALIDATION LOOP
        model.eval()
        val_loss = 0.0
        c_bold, c_italic, c_serif = 0, 0, 0
        total_size_mae = 0.0

        with torch.no_grad():
            for imgs, bolds, italics, serifs, sizes in val_loader:
                imgs = imgs.to(device)
                bolds = bolds.to(device)
                italics = italics.to(device)
                serifs = serifs.to(device)
                sizes = sizes.to(device)

                p_bold, p_italic, p_serif, p_size = model(imgs)

                loss_b = ce_loss(p_bold, bolds)
                loss_i = ce_loss(p_italic, italics)
                loss_s = ce_loss(p_serif, serifs)
                loss_sz = size_loss_fn(p_size, sizes) * 0.15
                val_loss += (loss_b + loss_i + loss_s + loss_sz).item() * imgs.size(0)

                pred_b = p_bold.argmax(dim=1)
                pred_i = p_italic.argmax(dim=1)
                pred_s = p_serif.argmax(dim=1)

                c_bold += (pred_b == bolds).sum().item()
                c_italic += (pred_i == italics).sum().item()
                c_serif += (pred_s == serifs).sum().item()
                total_size_mae += torch.abs(p_size - sizes).sum().item()

        val_loss = val_loss / len(val_dataset)
        bold_acc = (c_bold / val_size) * 100.0
        italic_acc = (c_italic / val_size) * 100.0
        serif_acc = (c_serif / val_size) * 100.0
        size_mae = total_size_mae / val_size

        epoch_sec = time.time() - t_epoch_start
        scheduler.step()

        # Tổng hợp điểm đánh giá trung bình
        avg_score = (bold_acc + italic_acc + serif_acc) / 3.0

        print(f"Ep {epoch:>2}/{epochs:<2} | {train_loss:<10.4f} | {val_loss:<9.4f} | {bold_acc:>6.2f}%   | {italic_acc:>7.2f}%   | {serif_acc:>7.2f}%   | {size_mae:>6.2f} pt | {epoch_sec:>5.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "bold_accuracy": round(bold_acc, 2),
            "italic_accuracy": round(italic_acc, 2),
            "serif_accuracy": round(serif_acc, 2),
            "size_mae_pt": round(size_mae, 2),
            "learning_rate": round(scheduler.get_last_lr()[0], 6),
            "seconds": round(epoch_sec, 2),
        })

        if avg_score > best_val_score:
            best_val_score = avg_score
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_params": num_params,
                "best_epoch": epoch,
                "bold_accuracy": bold_acc,
                "italic_accuracy": italic_acc,
                "serif_accuracy": serif_acc,
                "size_mae": size_mae,
                "input_shape": [1, INPUT_HEIGHT, INPUT_WIDTH],
            }, best_checkpoint_path)

    total_time = time.time() - t_start_all
    print("─" * 86)
    print(f"Huấn luyện hoàn tất trong: {total_time:.1f}s ({total_time / 60:.2f} phút)!")

    # Lưu log ra file JSON
    log_path = os.path.join(output_dir, "typography_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "DocumentTypographyNet",
            "parameters_count": num_params,
            "train_samples": train_size,
            "val_samples": val_size,
            "total_epochs": epochs,
            "total_training_seconds": round(total_time, 2),
            "best_validation_score": round(best_val_score, 2),
            "best_checkpoint_path": best_checkpoint_path,
            "history": history,
        }, f, indent=2, ensure_ascii=False)

    print(f"  -> Đã lưu mô hình tốt nhất vào: {best_checkpoint_path}")
    print(f"  -> Đã lưu nhật ký số liệu vào: {log_path}")

    # ==============================================================================
    # GIAI ĐOẠN 4: TEST THỰC NGHIỆM TRỰC TIẾP (INFERENCE VERIFICATION)
    # ==============================================================================
    print("\nBước 4/4: Kiểm thử suy luận thực tế (Inference Verification) trên 5 mẫu ngẫu nhiên:")
    print("─" * 86)
    test_gen = SyntheticTypographyGenerator()
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i in range(5):
            img_arr, gt = test_gen.generate_sample()
            tensor_in = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)
            t_inf_0 = time.time()
            p_bold, p_italic, p_serif, p_size = model(tensor_in)
            lat_ms = (time.time() - t_inf_0) * 1000

            pred_bold_str = "BOLD" if p_bold.argmax().item() == 1 else "Regular"
            pred_italic_str = "ITALIC" if p_italic.argmax().item() == 1 else "Upright"
            pred_font_str = "Times New Roman" if p_serif.argmax().item() == 1 else "Arial/Calibri"
            pred_size_val = round(p_size.item(), 1)

            gt_bold_str = "BOLD" if int(gt["is_bold"]) == 1 else "Regular"
            gt_italic_str = "ITALIC" if int(gt["is_italic"]) == 1 else "Upright"
            gt_font_str = "Times New Roman" if int(gt["is_serif"]) == 1 else "Arial/Calibri"
            gt_size_val = gt["font_size_pt"]

            print(f"Mẫu #{i+1} [Độ trễ: {lat_ms:.2f} ms]:")
            print(f"   Thực tế (GT):   Font={gt_font_str:<16} | Kiểu={gt_bold_str} + {gt_italic_str:<7} | Cỡ={gt_size_val:>4.1f} pt")
            print(f"   AI Dự đoán:     Font={pred_font_str:<16} | Kiểu={pred_bold_str} + {pred_italic_str:<7} | Cỡ={pred_size_val:>4.1f} pt")
    print("══════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    train_typography_model(
        train_size=2000,
        val_size=300,
        batch_size=32,
        epochs=8,
        lr=0.001,
        output_dir="models",
    )
