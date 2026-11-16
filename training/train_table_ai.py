# -*- coding: utf-8 -*-
"""
Huấn luyện TableLineNet: mạng nhận diện đường kẻ NGANG/DỌC của bảng trong ảnh chụp camera.

Dữ liệu 100% tổng hợp (không dùng trang của người dùng):
  - Bảng ngẫu nhiên 2-9 hàng x 2-6 cột, độ rộng cột/chiều cao hàng khác nhau, 2 kiểu viền,
    nội dung ô là âm tiết tiếng Việt ngẫu nhiên / số, 22 font TrueType (tiêu đề dùng font đậm).
  - Nhiễu "gây nhầm": đoạn văn, gạch chân chữ, đường kẻ phân cách đơn (nhãn = 0, không phải bảng).
  - 20% mẫu không có bảng.
  - Quang học camera: nghiêng ±2.5°, phối cảnh, bóng đổ, nhòe, nhiễu cảm biến, JPEG.

Đánh giá trên tập kiểm thử độc lập: tỉ lệ dựng ĐÚNG số hàng x số cột bằng đúng hàm
`extract_tables` mà worker sử dụng, và tỉ lệ báo nhầm bảng ở trang không có bảng.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "worker"))

from camera_text_generator import apply_camera_optics, discover_font_catalog, random_vietnamese_line  # noqa: E402
from table_structure import NET_H, NET_W, TableLineNet, extract_tables  # noqa: E402

RW, RH = NET_W * 2, NET_H * 2  # độ phân giải render (2x) rồi thu nhỏ như pipeline thật

_CATALOG = None
_FONT_CACHE: dict = {}


def _catalog():
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = discover_font_catalog()
    return _CATALOG


def _font(path: str, size: int):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def _pick_font(rng: random.Random, bold: bool = False, italic: bool = False) -> str:
    cat = _catalog()
    serif = rng.choice([0, 1])
    paths = cat.get((serif, int(bold), int(italic))) or cat.get((1, int(bold), int(italic))) or cat.get((0, 0, 0))
    return rng.choice(paths)


def _fit_text(text: str, font, max_w: int) -> str:
    while text and font.getlength(text) > max_w:
        if " " in text:
            text = text.rsplit(" ", 1)[0]
        else:
            text = text[:-1]
    return text


def _draw_paragraph(draw, rng, y_lo: int, y_hi: int, ink: int) -> None:
    if y_hi - y_lo < 50:
        return
    px = rng.randint(20, 28)
    step = int(px * rng.uniform(1.35, 1.7))
    path = _pick_font(rng)
    font = _font(path, px)
    y = y_lo + rng.randint(0, 12)
    x0 = rng.randint(24, 70)
    while y + px < y_hi:
        text = _fit_text(random_vietnamese_line(rng, min_words=6, max_words=11, allow_upper=False,
                                                allow_numbering=False), font, RW - x0 - 24)
        draw.text((x0, y), text, fill=ink, font=font)
        if rng.random() < 0.10:  # gạch chân chữ (nhãn 0)
            tw = int(font.getlength(text))
            draw.line([(x0, y + px + 3), (x0 + tw, y + px + 3)], fill=ink, width=rng.randint(1, 3))
        y += step


def render_sample(seed: int):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    bg = rng.randint(205, 250)
    page = Image.new("L", (RW, RH), bg)
    d = ImageDraw.Draw(page)
    hm = np.zeros((RH, RW), np.uint8)
    vm = np.zeros((RH, RW), np.uint8)
    ink = rng.randint(15, 90)
    gt: Optional[Tuple[int, int]] = None

    has_table = rng.random() < 0.8
    if has_table:
        n_cols = rng.randint(2, 6)
        n_rows = rng.randint(2, 9)
        tw = rng.randint(int(RW * 0.55), int(RW * 0.92))
        row_h = [rng.randint(46, 110) for _ in range(n_rows)]
        th = sum(row_h)
        max_th = int(RH * 0.78)
        if th > max_th:
            row_h = [max(40, int(h * max_th / th)) for h in row_h]
            th = sum(row_h)
        weights = [rng.uniform(0.6, 1.8) for _ in range(n_cols)]
        col_w = [max(80, int(tw * w / sum(weights))) for w in weights]
        tw = sum(col_w)
        x0 = rng.randint(8, max(9, RW - tw - 8))
        y0 = rng.randint(24, max(25, RH - th - 24))
        xs = [x0]
        for w in col_w:
            xs.append(xs[-1] + w)
        ys = [y0]
        for h in row_h:
            ys.append(ys[-1] + h)

        # đoạn văn phía trên/dưới bảng
        if rng.random() < 0.6:
            _draw_paragraph(d, rng, 10, y0 - 14, ink)
        if rng.random() < 0.6:
            _draw_paragraph(d, rng, ys[-1] + 14, RH - 10, ink)

        t = rng.choice([2, 3, 3, 4, 5])
        line_ink = min(140, ink + rng.randint(0, 40))
        style = rng.choice(["full", "full", "full", "outer_thick"])

        def thick(is_outer: bool) -> int:
            return t + 2 if (style == "outer_thick" and is_outer) else t

        for ri, y in enumerate(ys):
            tt = thick(ri in (0, len(ys) - 1))
            box = [xs[0] - tt // 2, y - tt // 2, xs[-1] + tt // 2, y + tt // 2]
            d.rectangle(box, fill=line_ink)
            hm[max(0, box[1]):box[3] + 1, max(0, box[0]):box[2] + 1] = 1
        for ci, x in enumerate(xs):
            tt = thick(ci in (0, len(xs) - 1))
            box = [x - tt // 2, ys[0] - tt // 2, x + tt // 2, ys[-1] + tt // 2]
            d.rectangle(box, fill=line_ink)
            vm[max(0, box[1]):box[3] + 1, max(0, box[0]):box[2] + 1] = 1

        for r in range(n_rows):
            ch = ys[r + 1] - ys[r]
            px = int(np.clip(ch * 0.42, 14, 32))
            for c in range(n_cols):
                cw = xs[c + 1] - xs[c]
                header = r == 0
                bold = header and rng.random() < 0.7
                italic = (not header) and rng.random() < 0.06
                font = _font(_pick_font(rng, bold, italic), px)
                if rng.random() < 0.22:
                    text = str(rng.randint(1, 9999))
                else:
                    nw = rng.randint(1, 2 if cw < 150 else 4)
                    text = random_vietnamese_line(rng, min_words=nw, max_words=nw, allow_upper=header and rng.random() < 0.3,
                                                  allow_numbering=False)
                text = _fit_text(text, font, cw - 14)
                if not text:
                    continue
                tw_txt = int(font.getlength(text))
                tx = xs[c] + (cw - tw_txt) // 2 if rng.random() < 0.3 else xs[c] + 7
                d.text((tx, ys[r] + (ch - px) // 2 - 2), text, fill=ink, font=font)
        gt = (n_rows, n_cols)
    else:
        _draw_paragraph(d, rng, 10, RH - 10, ink)
        for _ in range(rng.randint(0, 2)):  # đường kẻ phân cách đơn (không phải bảng)
            yy = rng.randint(40, RH - 40)
            d.line([(rng.randint(10, 80), yy), (RW - rng.randint(10, 80), yy)], fill=ink, width=rng.randint(2, 4))

    img = np.array(page, dtype=np.uint8)

    # hình học: nghiêng + phối cảnh, áp dụng giống nhau cho ảnh và nhãn
    ang = rng.uniform(-2.5, 2.5)
    M = cv2.getRotationMatrix2D((RW / 2, RH / 2), ang, 1.0)
    M3 = np.vstack([M, [0, 0, 1]])
    j = rng.uniform(0, 16)
    src = np.float32([[0, 0], [RW, 0], [RW, RH], [0, RH]])
    dst = src + np_rng.uniform(-j, j, size=(4, 2)).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    G = P @ M3
    img = cv2.warpPerspective(img, G, (RW, RH), flags=cv2.INTER_LINEAR, borderValue=int(bg))
    hm = cv2.warpPerspective(hm, G, (RW, RH), flags=cv2.INTER_LINEAR, borderValue=0)
    vm = cv2.warpPerspective(vm, G, (RW, RH), flags=cv2.INTER_LINEAR, borderValue=0)

    img = apply_camera_optics(img, rng, np_rng, max_skew_deg=0.0)

    img = cv2.resize(img, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    hm = (cv2.resize(hm.astype(np.float32), (NET_W, NET_H), interpolation=cv2.INTER_AREA) > 0.3).astype(np.uint8)
    vm = (cv2.resize(vm.astype(np.float32), (NET_W, NET_H), interpolation=cv2.INTER_AREA) > 0.3).astype(np.uint8)
    return img, hm, vm, gt


def _gen(seed: int):
    return render_sample(seed)


def build_set(seeds):
    with mp.Pool(max(2, min(6, (os.cpu_count() or 4) - 1))) as pool:
        res = pool.map(_gen, seeds, chunksize=8)
    imgs = np.stack([r[0] for r in res])
    hms = np.stack([r[1] for r in res])
    vms = np.stack([r[2] for r in res])
    gts = [r[3] for r in res]
    return imgs, hms, vms, gts


def loss_fn(logits, hm, vm):
    target = torch.stack([hm, vm], dim=1).float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(6.0))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
    return bce + dice.mean()


def batch_tensor(imgs, idx):
    x = torch.from_numpy((imgs[idx].astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(1)
    return x


@torch.no_grad()
def evaluate(model, imgs, hms, vms, gts, bs=16):
    model.eval()
    tp = fp = fn = 0
    exact = n_tab = fp_neg = n_neg = detect_any = 0
    for s in range(0, len(imgs), bs):
        idx = np.arange(s, min(len(imgs), s + bs))
        prob = torch.sigmoid(model(batch_tensor(imgs, idx))).numpy()
        for k, i in enumerate(idx):
            ph, pv = prob[k, 0], prob[k, 1]
            pred = np.stack([ph > 0.5, pv > 0.5])
            tgt = np.stack([hms[i] > 0, vms[i] > 0])
            tp += int((pred & tgt).sum()); fp += int((pred & ~tgt).sum()); fn += int((~pred & tgt).sum())
            tables = extract_tables(ph, pv)
            if gts[i] is None:
                n_neg += 1
                fp_neg += int(len(tables) > 0)
            else:
                n_tab += 1
                if len(tables) > 0:
                    detect_any += 1
                if len(tables) == 1 and (tables[0].n_rows, tables[0].n_cols) == gts[i]:
                    exact += 1
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return f1, exact / max(1, n_tab), detect_any / max(1, n_tab), fp_neg / max(1, n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=700)
    ap.add_argument("--val", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    root = os.path.dirname(_HERE)
    out = args.out or os.path.join(root, "models", "table_line_net.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    tr = build_set([1_000_000 + i for i in range(args.train)])
    va = build_set([9_000_000 + i for i in range(args.val)])
    print(f"[TableAI] Sinh {args.train} train + {args.val} val mẫu trong {time.time() - t0:.1f}s", flush=True)

    model = TableLineNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((args.train + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(0)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(args.train)
        ep_loss = 0.0
        te = time.time()
        for s in range(0, args.train, args.bs):
            idx = order[s:s + args.bs]
            # mạng là fully-convolutional: huấn luyện trên crop ngẫu nhiên 256x192 để nhanh trên CPU
            cy = int(rng.integers(0, NET_H - 256 + 1))
            cx = int(rng.integers(0, NET_W - 192 + 1))
            x = batch_tensor(tr[0], idx)[:, :, cy:cy + 256, cx:cx + 192]
            x = x * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.12, 0.12))
            logits = model(x)
            loss = loss_fn(logits, torch.from_numpy(tr[1][idx][:, cy:cy + 256, cx:cx + 192]),
                           torch.from_numpy(tr[2][idx][:, cy:cy + 256, cx:cx + 192]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss.item()) * len(idx)
        f1, exact, det, fpn = evaluate(model, *va)
        print(f"Epoch {ep:02d}/{args.epochs} ({time.time() - te:.0f}s) loss={ep_loss / args.train:.4f} | "
              f"val pixelF1={f1:.4f} | bảng đúng hàng x cột={exact:.4f} | phát hiện bảng={det:.4f} | báo nhầm(không bảng)={fpn:.4f}",
              flush=True)
        score = exact + f1 - fpn
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "pixel_f1": f1, "grid_exact": exact,
                        "false_pos_no_table": fpn, "epoch": ep}, out)
    print(f"[TableAI] Đã lưu {out}", flush=True)


if __name__ == "__main__":
    main()
