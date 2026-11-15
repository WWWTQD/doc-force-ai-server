# -*- coding: utf-8 -*-
"""
Tinh chỉnh (Fine-tune) VietOCR (`vgg_seq2seq`) cho ảnh dòng chữ chụp trực tiếp từ camera:
  - Dùng 100% ngữ âm tiếng Việt tổng hợp (`random_vietnamese_line`, KHÔNG dùng văn bản cố định của user)
  - Kết xuất bằng 22 font TrueType thật (Regular/Bold/Italic/Bold-Italic)
  - Mô phỏng sai số camera thực tế: bóng đổ ánh sáng, nghiêng nhẹ ±1.8°, nhòe tiêu cự,
    nén JPEG, nhiễu cảm biến và rìa chữ dòng trên/dưới dính nhẹ vào mép crop DBNet.
  - Đo Baseline Val CER / Line Accuracy TRƯỚC khi train trên tập kiểm thử độc lập (held-out)
    và chỉ lưu `models/vietocr_camera_ft.pth` khi Val CER thực sự cải thiện (không làm hỏng trọng số gốc).
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model, process_image, translate

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


def levenshtein_chars(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein_chars(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def synthesize_line_sample(
    catalog: dict,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[Image.Image, str]:
    keys = [k for k, paths in catalog.items() if paths]
    is_serif, is_bold, is_italic = rng.choice(keys)
    fpath = rng.choice(catalog[(is_serif, is_bold, is_italic)])
    font_px = rng.randint(22, 34)
    try:
        font = _get_font(fpath, font_px)
    except Exception:
        font = ImageFont.load_default()

    # Độ dài 2-5 từ giữ chiều dài chuỗi <= 32 ký tự để fine-tune nhanh trên CPU
    n_words = rng.randint(2, 5)
    text = random_vietnamese_line(
        rng,
        min_words=n_words,
        max_words=n_words,
        allow_upper=(rng.random() < 0.25),
        allow_numbering=(rng.random() < 0.18),
    )
    if len(text) > 28:
        text = " ".join(text.split()[:4])

    dummy = Image.new("L", (10, 10), 255)
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = max(32, bbox[2] - bbox[0])
    th = max(20, bbox[3] - bbox[1])

    pad_x = rng.randint(6, 14)
    pad_y = rng.randint(4, 9)
    w = tw + pad_x * 2
    h = th + pad_y * 2
    bg = rng.randint(190, 246)
    ink = rng.randint(20, 82)

    img = Image.new("L", (w, h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink, font=font)

    # 12% mô phỏng mép nét chữ dòng hàng xóm chạm rìa trên/dưới (giống crop hẹp từ DBNet)
    if rng.random() < 0.12:
        bleed_txt = random_vietnamese_line(rng, min_words=2, max_words=4, allow_upper=False, allow_numbering=False)
        if rng.random() < 0.5:
            draw.text((pad_x, -int(th * 0.78)), bleed_txt, fill=ink + 20, font=font)
        else:
            draw.text((pad_x, h - int(th * 0.18)), bleed_txt, fill=ink + 20, font=font)

    gray = apply_camera_optics(np.array(img, dtype=np.uint8), rng, np_rng, max_skew_deg=1.8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb), text


def preprocess_batch_images(imgs: List[Image.Image], max_w: int = 280) -> torch.Tensor:
    arrays = [process_image(im, 32, 32, max_w) for im in imgs]
    max_cur_w = max(a.shape[2] for a in arrays)
    batch = np.ones((len(arrays), 3, 32, max_cur_w), dtype=np.float32)
    for i, a in enumerate(arrays):
        cw = a.shape[2]
        batch[i, :, :, :cw] = a
    return torch.from_numpy(batch)


def evaluate_cer(model, vocab, val_pairs: List[Tuple[Image.Image, str]], device: torch.device) -> Tuple[float, float]:
    model.eval()
    tot_chars = 0
    tot_edits = 0
    exact_ok = 0
    bs = 16
    with torch.no_grad():
        for i in range(0, len(val_pairs), bs):
            chunk = val_pairs[i : i + bs]
            imgs = [p[0] for p in chunk]
            gts = [p[1] for p in chunk]
            x = preprocess_batch_images(imgs).to(device)
            s_toks, _ = translate(x, model, max_seq_length=48)
            preds = vocab.batch_decode(s_toks.tolist())
            for pred, gt in zip(preds, gts):
                pred_s = (pred or "").strip()
                gt_s = gt.strip()
                tot_edits += levenshtein_chars(pred_s, gt_s)
                tot_chars += max(1, len(gt_s))
                if pred_s == gt_s:
                    exact_ok += 1
    return tot_edits / max(1, tot_chars), exact_ok / max(1, len(val_pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=240)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--base-weights", type=str, default="/tmp/vgg_seq2seq.pth")
    args = parser.parse_args()

    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "vietocr_camera_ft.pth")

    catalog = discover_font_catalog(os.path.join(models_dir, "fonts"))
    cfg = Cfg.load_config_from_name("vgg_seq2seq")
    cfg["cnn"]["pretrained"] = False
    cfg["device"] = "cpu"
    device = torch.device("cpu")

    model, vocab = build_model(cfg)
    if os.path.exists(args.base_weights):
        model.load_state_dict(torch.load(args.base_weights, map_location=device))
        print(f"[CameraOCR] Đã nạp base weights từ {args.base_weights}")
    # Chỉ tinh chỉnh CNN trích xuất đặc trưng thị giác cho ảnh camera; đóng băng Transformer/Seq2Seq
    # để bảo toàn 100% mô hình ngôn ngữ tiếng Việt gốc (tránh trôi từ vựng).
    for p in model.transformer.parameters():
        p.requires_grad = False
    model.to(device)

    val_rng = random.Random(314159)
    val_np = np.random.default_rng(314159)
    val_pairs = [synthesize_line_sample(catalog, val_rng, val_np) for _ in range(args.val_samples)]

    base_cer, base_acc = evaluate_cer(model, vocab, val_pairs, device)
    print(f"[CameraOCR] Baseline TRƯỚC fine-tune trên {len(val_pairs)} dòng camera chưa từng thấy: CER={base_cer:.4f}, ExactAcc={base_acc:.4f}")

    tr_rng = random.Random(20261116)
    tr_np = np.random.default_rng(20261116)
    train_pairs = [synthesize_line_sample(catalog, tr_rng, tr_np) for _ in range(args.train_samples)]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    best_cer = base_cer
    saved = False

    for ep in range(1, args.epochs + 1):
        tr_rng.shuffle(train_pairs)
        model.train()
        ep_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, len(train_pairs), args.batch_size):
            chunk = train_pairs[i : i + args.batch_size]
            imgs = [p[0] for p in chunk]
            gts = [p[1] for p in chunk]
            x = preprocess_batch_images(imgs).to(device)

            encoded = [vocab.encode(t) for t in gts]
            max_len = max(len(e) for e in encoded)
            tgt = np.zeros((len(encoded), max_len), dtype=np.int64)
            for r, e in enumerate(encoded):
                tgt[r, : len(e)] = e
            tgt_t = torch.from_numpy(tgt).to(device)
            tgt_inp = tgt_t[:, :-1].transpose(0, 1)  # (T, B)
            tgt_out = tgt_t[:, 1:]  # (B, T)

            optimizer.zero_grad()
            logits = model(x, tgt_inp, tgt_key_padding_mask=None)  # (B, T, V)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += float(loss.item())
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  Epoch {ep} batch {n_batches} | avg_loss={ep_loss/n_batches:.4f} | elapsed={time.time()-t0:.1f}s")

        val_cer, val_acc = evaluate_cer(model, vocab, val_pairs, device)
        print(f"[CameraOCR] Sau Epoch {ep}: val_CER={val_cer:.4f} (baseline={base_cer:.4f}), ExactAcc={val_acc:.4f}")
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), out_path)
            saved = True
            print(f"[CameraOCR] Cải thiện CER -> Đã lưu weights mới vào {out_path}!")

    if not saved:
        print("[CameraOCR] Baseline weights đã tối ưu nhất trên validation; giữ nguyên base weights để tránh overfitting.")


if __name__ == "__main__":
    main()
