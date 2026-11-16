# -*- coding: utf-8 -*-
"""
Nhận diện cấu trúc bảng bằng mạng nơ-ron tự huấn luyện (TableLineNet).

- `TableLineNet`: U-Net nhỏ (1 kênh xám -> 2 kênh: đường kẻ NGANG, đường kẻ DỌC của bảng).
  Huấn luyện trong `training/train_table_ai.py` bằng bảng tổng hợp + quang học camera
  (nghiêng, phối cảnh, bóng đổ, nhòe, nhiễu, JPEG) — không dùng mẫu của người dùng.
- `extract_tables`: dựng lưới hàng/cột từ bản đồ xác suất (chịu được đường kẻ hơi nghiêng).

Một nguồn mã duy nhất cho cả huấn luyện và worker.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TORCH_OK = False

NET_W = 384
NET_H = 512


if TORCH_OK:

    class _ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.body(x)

    class TableLineNet(nn.Module):
        """U-Net nhỏ: logits (B, 2, H, W) = [đường ngang, đường dọc]."""

        def __init__(self, ch: Tuple[int, int, int, int] = (8, 16, 32, 48)):
            super().__init__()
            c0, c1, c2, c3 = ch
            self.e1 = _ConvBlock(1, c0)
            self.e2 = _ConvBlock(c0, c1)
            self.e3 = _ConvBlock(c1, c2)
            self.mid = _ConvBlock(c2, c3)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d3 = _ConvBlock(c3 + c2, c2)
            self.d2 = _ConvBlock(c2 + c1, c1)
            self.d1 = _ConvBlock(c1 + c0, c0)
            self.head = nn.Conv2d(c0, 2, 1)

        def forward(self, x):
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            m = self.mid(self.pool(s3))
            u3 = self.d3(torch.cat([self.up(m), s3], dim=1))
            u2 = self.d2(torch.cat([self.up(u3), s2], dim=1))
            u1 = self.d1(torch.cat([self.up(u2), s1], dim=1))
            return self.head(u1)


def predict_line_maps(model, gray: np.ndarray, device=None) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (h_prob, v_prob) kích thước (NET_H, NET_W) trong [0, 1]."""
    small = cv2.resize(gray, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy((small.astype(np.float32) / 255.0 - 0.5) / 0.5)[None, None]
    if device is not None:
        x = x.to(device)
    with torch.inference_mode():
        prob = torch.sigmoid(model(x))[0].cpu().numpy()
    return prob[0], prob[1]


def _fit_segments(mask: np.ndarray, axis: str, min_len: float, ox: int, oy: int) -> List[dict]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs: List[dict] = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        ext = w if axis == "h" else h
        if ext < min_len:
            continue
        ys, xs = np.nonzero(lab == i)
        if axis == "h":
            if np.ptp(xs) < 2:
                continue
            a, b = np.polyfit(xs + ox, ys + oy, 1)
            lo, hi = x + ox, x + w - 1 + ox
        else:
            if np.ptp(ys) < 2:
                continue
            a, b = np.polyfit(ys + oy, xs + ox, 1)
            lo, hi = y + oy, y + h - 1 + oy
        if abs(a) > 0.2:
            continue
        segs.append({"a": float(a), "b": float(b), "lo": int(lo), "hi": int(hi)})
    return segs


def _merge_lines(
    segs: List[dict],
    mid: float,
    gap: float,
    min_total_span: float = 0.0,
) -> List[Tuple[float, float]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s["a"] * mid + s["b"])
    groups: List[List[dict]] = [[segs[0]]]
    for s in segs[1:]:
        prev = groups[-1][-1]
        if abs((s["a"] * mid + s["b"]) - (prev["a"] * mid + prev["b"])) <= gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    out = []
    for g in groups:
        # Tính tổng độ phủ dọc theo đường kẻ (hợp các đoạn collinear bị đứt nét)
        intervals = sorted((s["lo"], s["hi"]) for s in g)
        covered = 0
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi + 8:
                cur_hi = max(cur_hi, hi)
            else:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        covered += cur_hi - cur_lo
        if covered < min_total_span:
            continue
        # Ước lượng (a, b) trung bình có trọng số theo độ dài từng đoạn
        weights = [max(1, s["hi"] - s["lo"]) for s in g]
        w_sum = float(sum(weights))
        a_avg = sum(s["a"] * w for s, w in zip(g, weights)) / w_sum
        b_avg = sum(s["b"] * w for s, w in zip(g, weights)) / w_sum
        out.append((float(a_avg), float(b_avg)))
    return out


class TableGrid:
    def __init__(self, bbox: Tuple[int, int, int, int], h_lines, v_lines):
        self.bbox = bbox
        self.h_lines = h_lines
        self.v_lines = v_lines
        self.n_rows = max(0, len(h_lines) - 1)
        self.n_cols = max(0, len(v_lines) - 1)

    def cell_quad(self, r: int, c: int) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Trả về 4 góc (tl, tr, br, bl) của ô (r, c) theo đúng giao điểm 4 đường kẻ nghiêng."""
        ah0, bh0 = self.h_lines[r]
        ah1, bh1 = self.h_lines[r + 1]
        av0, bv0 = self.v_lines[c]
        av1, bv1 = self.v_lines[c + 1]

        def _intersect(ah: float, bh: float, av: float, bv: float) -> Tuple[float, float]:
            denom = 1.0 - ah * av
            if abs(denom) < 1e-6:
                x = av * bh + bv
                return float(x), float(ah * x + bh)
            y = (ah * bv + bh) / denom
            x = av * y + bv
            return float(x), float(y)

        return (
            _intersect(ah0, bh0, av0, bv0),
            _intersect(ah0, bh0, av1, bv1),
            _intersect(ah1, bh1, av1, bv1),
            _intersect(ah1, bh1, av0, bv0),
        )

    def locate(self, cx: float, cy: float) -> Tuple[int, int]:
        ys = [a * cx + b for a, b in self.h_lines]
        xs = [a * cy + b for a, b in self.v_lines]
        r = sum(1 for y in ys if y < cy) - 1
        c = sum(1 for x in xs if x < cx) - 1
        return max(0, min(self.n_rows - 1, r)), max(0, min(self.n_cols - 1, c))

    def contains(self, cx: float, cy: float, margin: float = 4.0) -> bool:
        x0, y0, x1, y1 = self.bbox
        return (x0 - margin) <= cx <= (x1 + margin) and (y0 - margin) <= cy <= (y1 + margin)


def extract_tables(
    h_prob: np.ndarray,
    v_prob: np.ndarray,
    thr: float = 0.32,
    min_w_frac: float = 0.28,
    min_h_frac: float = 0.06,
) -> List[TableGrid]:
    """
    Dựng lưới bảng từ bản đồ đường kẻ `TableLineNet` (toạ độ ở độ phân giải NET_W x NET_H).
    Gộp toàn cục các đoạn thẳng đồng tuyến (collinear) chịu được đường kẻ bị đứt quãng bởi ánh sáng/chữ,
    sau đó lọc các đường dọc thực sự giao cắt với đường ngang để loại nhiễu gáy sách/mép giấy.
    """
    H, W = h_prob.shape[:2]
    hm = (h_prob > thr).astype(np.uint8)
    vm = (v_prob > thr).astype(np.uint8)
    hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    vm = cv2.morphologyEx(vm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))

    # 1. Tìm toàn bộ đoạn thẳng dọc/ngang, loại bỏ mép giấy ngoài cùng
    v_segs_all = _fit_segments(vm, "v", min_len=max(18.0, H * 0.045), ox=0, oy=0)
    v_segs_all = [s for s in v_segs_all if 3 < (s["a"] * 0.5 * (s["lo"] + s["hi"]) + s["b"]) < (W - 3)]
    h_segs_all = _fit_segments(hm, "h", min_len=max(12.0, W * 0.035), ox=0, oy=0)
    if len(v_segs_all) < 2 or len(h_segs_all) < 2:
        return []

    # Gom cụm các đường dọc sơ bộ theo tâm Y chung của trang
    v_groups = _group_collinear(v_segs_all, mid=H * 0.5, gap=8.0, min_total_span=max(32.0, H * 0.07))
    h_groups = _group_collinear(h_segs_all, mid=W * 0.5, gap=7.0, min_total_span=max(36.0, W * 0.10))
    if len(v_groups) < 2 or len(h_groups) < 2:
        return []

    # 2. Xác định cụm đường dọc có chồng lấn theo trục Y và giao cắt với đường ngang
    tables: List[TableGrid] = []
    valid_v = []
    for g in v_groups:
        y_lo = min(s["lo"] for s in g)
        y_hi = max(s["hi"] for s in g)
        x_lo = min(s["a"] * s["lo"] + s["b"] for s in g)
        x_hi = max(s["a"] * s["hi"] + s["b"] for s in g)
        xc = 0.5 * (x_lo + x_hi)
        h_hits = 0
        for hg in h_groups:
            hx_lo = min(s["lo"] for s in hg)
            hx_hi = max(s["hi"] for s in hg)
            hy = _weighted_line(hg)
            ym_v = hy[0] * xc + hy[1]
            # Khống chế dung sai hẹp ±14px để không dính vệt gáy sách ngoài bảng
            if (y_lo - 10) <= ym_v <= (y_hi + 10) and (hx_lo - 14) <= xc <= (hx_hi + 14):
                h_hits += 1
        if h_hits >= 1:
            valid_v.append((_weighted_line(g), y_lo, y_hi, h_hits))

    if len(valid_v) < 2:
        return []

    valid_v.sort(key=lambda item: item[1])
    v_clusters: List[List[tuple]] = [[valid_v[0]]]
    for item in valid_v[1:]:
        cur_max_y = max(it[2] for it in v_clusters[-1])
        if item[1] <= cur_max_y + 16:
            v_clusters[-1].append(item)
        else:
            v_clusters.append([item])

    for vc in v_clusters:
        # Cần ít nhất 2 đường dọc có >= 2 giao điểm ngang; các đường biên còn lại cần span tương đương
        anchor_v = [it for it in vc if it[3] >= 2]
        if len(anchor_v) < 2:
            continue
        anc_y0 = min(it[1] for it in anchor_v)
        anc_y1 = max(it[2] for it in anchor_v)
        anc_h = max(1, anc_y1 - anc_y0)
        vc_kept = [
            it for it in vc
            if it[3] >= 2 or ((it[2] - it[1]) >= 0.65 * anc_h and abs(it[1] - anc_y0) <= 22)
        ]
        if len(vc_kept) < 2:
            continue
        ty0 = min(it[1] for it in vc_kept)
        ty1 = max(it[2] for it in vc_kept)
        ymid = 0.5 * (ty0 + ty1)
        vc_sorted = sorted(vc_kept, key=lambda it: it[0][0] * ymid + it[0][1])
        tx0 = int(round(vc_sorted[0][0][0] * ymid + vc_sorted[0][0][1]))
        tx1 = int(round(vc_sorted[-1][0][0] * ymid + vc_sorted[-1][0][1]))
        tw = tx1 - tx0
        th = ty1 - ty0
        if tw < W * min_w_frac or th < H * min_h_frac:
            continue

        # Chọn các nhóm đường ngang thuộc khoảng X và Y của bảng này
        h_in_tbl = []
        for hg in h_groups:
            line_h = _weighted_line(hg)
            hx_lo = min(s["lo"] for s in hg)
            hx_hi = max(s["hi"] for s in hg)
            hy_mid = line_h[0] * (0.5 * (tx0 + tx1)) + line_h[1]
            ov_x = min(hx_hi, tx1 + 12) - max(hx_lo, tx0 - 12)
            total_span = sum(s["hi"] - s["lo"] for s in hg)
            if (ty0 - 12) <= hy_mid <= (ty1 + 12) and ov_x >= 0.14 * tw and total_span >= max(32.0, 0.14 * tw):
                h_in_tbl.append(line_h)

        if len(h_in_tbl) < 2:
            continue

        h_in_tbl.sort(key=lambda ln: ln[0] * (0.5 * (tx0 + tx1)) + ln[1])
        # Gộp hai đường ngang quá sát (< 8px)
        merged_h = [h_in_tbl[0]]
        for ln in h_in_tbl[1:]:
            prev_y = merged_h[-1][0] * (0.5 * (tx0 + tx1)) + merged_h[-1][1]
            cur_y = ln[0] * (0.5 * (tx0 + tx1)) + ln[1]
            if cur_y - prev_y >= 8.0:
                merged_h.append(ln)

        # Nếu đáy thực tế của các đường dọc neo kéo sâu hơn đường ngang cuối cùng >= 34px, đóng đáy bảng
        last_y = merged_h[-1][0] * (0.5 * (tx0 + tx1)) + merged_h[-1][1]
        if anc_y1 - last_y >= 34.0:
            merged_h.append((0.0, float(anc_y1)))

        if len(merged_h) < 2:
            continue
        v_lines_tbl = [it[0] for it in vc_sorted]
        gy0 = int(round(merged_h[0][0] * (0.5 * (tx0 + tx1)) + merged_h[0][1]))
        gy1 = int(round(merged_h[-1][0] * (0.5 * (tx0 + tx1)) + merged_h[-1][1]))
        tables.append(TableGrid((tx0, gy0, tx1, gy1), merged_h, v_lines_tbl))

    tables.sort(key=lambda t: t.bbox[1])
    return tables


def _weighted_line(g: List[dict]) -> Tuple[float, float]:
    weights = [max(1, s["hi"] - s["lo"]) for s in g]
    w_sum = float(sum(weights))
    a_avg = sum(s["a"] * w for s, w in zip(g, weights)) / w_sum
    b_avg = sum(s["b"] * w for s, w in zip(g, weights)) / w_sum
    return float(a_avg), float(b_avg)


def _group_collinear(
    segs: List[dict],
    mid: float,
    gap: float,
    min_total_span: float = 0.0,
) -> List[List[dict]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s["a"] * mid + s["b"])
    groups: List[List[dict]] = [[segs[0]]]
    for s in segs[1:]:
        prev = groups[-1][-1]
        if abs((s["a"] * mid + s["b"]) - (prev["a"] * mid + prev["b"])) <= gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    out = []
    for g in groups:
        intervals = sorted((s["lo"], s["hi"]) for s in g)
        covered = 0
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi + 10:
                cur_hi = max(cur_hi, hi)
            else:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        covered += cur_hi - cur_lo
        if covered >= min_total_span:
            out.append(g)
    return out
