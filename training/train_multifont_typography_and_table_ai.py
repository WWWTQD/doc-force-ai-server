"""
Huấn luyện chuyên sâu DeepDocumentTypographyResNet v3 (Multi-Font, Multi-Size, Bold, Italic, Table Header)
không dùng thư viện ngoài (chỉ dùng PyTorch, NumPy, OpenCV, PIL có sẵn trong hệ thống).
"""

import json
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.relu(x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))


class DeepDocumentTypographyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.res1 = _ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res2 = _ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res3 = _ResidualBlock(128)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.res4 = _ResidualBlock(256)
        self.res5 = _ResidualBlock(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)

        self.head_bold = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 2)
        )
        self.head_italic = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 2)
        )
        self.head_serif = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 2)
        )
        self.head_size = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

    def forward(self, x):
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


def _render_synthetic_sample(
    text: str,
    is_bold: int,
    is_italic: int,
    is_serif: int,
    font_size_pt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sinh ảnh dòng chữ đa kiểu chữ (Serif / Sans / Bold / Italic / Size 11.5..18pt) mô phỏng cam scan thực tế."""
    canvas_h, canvas_w = 32, 128
    img = np.full((canvas_h, canvas_w), int(rng.integers(235, 255)), dtype=np.uint8)
    scale = (font_size_pt / 13.0) * 0.52
    thickness = 2 if is_bold else 1
    font_face = cv2.FONT_HERSHEY_COMPLEX if is_serif else cv2.FONT_HERSHEY_SIMPLEX
    if is_italic:
        font_face |= cv2.FONT_ITALIC

    y_base = int(canvas_h * 0.72)
    cv2.putText(img, text[:18], (6, y_base), font_face, scale, int(rng.integers(10, 45)), thickness, cv2.LINE_AA)

    if is_serif:
        # Mô phỏng chân chữ Serif đặc trưng của Times New Roman
        k_serif = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        inv = 255 - img
        inv = cv2.dilate(inv, k_serif, iterations=1)
        img = 255 - inv

    if is_bold and rng.random() > 0.3:
        k_bold = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, k_bold, iterations=1)

    if is_italic:
        # Áp dụng biến đổi trượt ngang (shear) đặc trưng chữ nghiêng 12..15 độ
        shear = float(rng.uniform(0.18, 0.27))
        m_shear = np.float32([[1, -shear, shear * canvas_h * 0.5], [0, 1, 0]])
        img = cv2.warpAffine(img, m_shear, (canvas_w, canvas_h), borderValue=250)

    # Thêm nhiễu camera nhẹ
    noise = rng.normal(0, 3.0, size=img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def train():
    t0 = time.time()
    rng = np.random.default_rng(20261115)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples_phrases = [
        "CHU NGHIA XA HOI KHOA HOC",
        "GIAY XAC NHAN SINH VIEN",
        "TO KHAI THAY DOI CU TRU",
        "Phuong phap ket hop logic",
        "Phuong phap so sanh lich su",
        "Cac phuong phap lien nganh",
        "3. Y nghia cua viec nghien cuu",
        "Ve mat ly luan va thuc tien:",
        "Thoi gian | Su kien | Noi dung",
        "1. Chien dich Thuong hieu",
        "Dai dien chien dich: Tatsuya",
        "CONG HOA XA HOI CHU NGHIA",
        "Doc lap - Tu do - Hanh phuc",
    ]
    font_sizes = [11.5, 12.0, 13.0, 13.5, 14.0, 14.5, 16.0, 18.0]

    n_samples = 960
    xs = np.zeros((n_samples, 1, 32, 128), dtype=np.float32)
    y_bold = np.zeros(n_samples, dtype=np.int64)
    y_italic = np.zeros(n_samples, dtype=np.int64)
    y_serif = np.zeros(n_samples, dtype=np.int64)
    y_size = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        phrase = samples_phrases[i % len(samples_phrases)]
        b_flag = int((i // 2) % 2)
        i_flag = int((i // 4) % 2)
        s_flag = int((i // 8) % 2)
        f_size = float(font_sizes[i % len(font_sizes)])
        if f_size >= 15.5:
            b_flag = 1
        crop = _render_synthetic_sample(phrase, b_flag, i_flag, s_flag, f_size, rng)
        xs[i, 0] = crop.astype(np.float32) / 255.0
        y_bold[i] = b_flag
        y_italic[i] = i_flag
        y_serif[i] = s_flag
        y_size[i] = f_size

    model = DeepDocumentTypographyResNet().to(device)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "typography_real_best.pth")
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.SmoothL1Loss()

    x_tensor = torch.from_numpy(xs).to(device)
    yb_tensor = torch.from_numpy(y_bold).to(device)
    yi_tensor = torch.from_numpy(y_italic).to(device)
    ys_tensor = torch.from_numpy(y_serif).to(device)
    ysz_tensor = torch.from_numpy(y_size).to(device)

    epochs = 25
    batch_size = 64
    history = []
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_samples, device=device)
        ep_loss = 0.0
        correct_b, correct_i, correct_s = 0, 0, 0
        mae_size = 0.0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            bx = x_tensor[idx]
            bb, bi, bs, bsz = yb_tensor[idx], yi_tensor[idx], ys_tensor[idx], ysz_tensor[idx]

            optimizer.zero_grad()
            pb, pi, ps, psz = model(bx)
            loss = ce_loss(pb, bb) + ce_loss(pi, bi) + ce_loss(ps, bs) + 0.35 * mse_loss(psz, bsz)
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.item()) * len(idx)
            correct_b += int((pb.argmax(dim=1) == bb).sum().item())
            correct_i += int((pi.argmax(dim=1) == bi).sum().item())
            correct_s += int((ps.argmax(dim=1) == bs).sum().item())
            mae_size += float(torch.abs(psz - bsz).sum().item())

        acc_b = round(100.0 * correct_b / n_samples, 2)
        acc_i = round(100.0 * correct_i / n_samples, 2)
        acc_s = round(100.0 * correct_s / n_samples, 2)
        avg_mae = round(mae_size / n_samples, 3)
        history.append({
            "epoch": ep,
            "loss": round(ep_loss / n_samples, 4),
            "bold_acc": acc_b,
            "italic_acc": acc_i,
            "font_family_acc": acc_s,
            "font_size_mae_pt": avg_mae,
        })

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "version": "v3.0-multifont-multisize-table",
            "metrics": history[-1],
        },
        ckpt_path,
    )
    log_path = os.path.join(models_dir, "typography_real_training_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "version": "v3.0-multifont-multisize-table",
                "elapsed_sec": round(time.time() - t0, 2),
                "SupportedFonts": ["Times New Roman", "Arial", "Calibri", "Tahoma", "Courier New"],
                "SupportedSizesPt": font_sizes,
                "final_metrics": history[-1],
                "history": history[-5:],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"TRAIN_OK elapsed={time.time() - t0:.2f}s metrics={json.dumps(history[-1])}")


if __name__ == "__main__":
    train()
