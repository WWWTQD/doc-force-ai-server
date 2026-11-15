# -*- coding: utf-8 -*-
"""
Trình sinh dữ liệu văn bản tiếng Việt & mô phỏng quang học máy ảnh (Domain-Agnostic Camera Simulator).

Chức năng:
  - Sinh ngẫu nhiên các từ, cụm từ, số hiệu văn bản và dòng văn bản tiếng Việt hợp lệ theo quy tắc
    ngữ âm tiếng Việt (âm đầu + vần + thanh điệu + vần cuối), kết hợp chữ hoa, chữ thường, số, dấu câu.
  - Tuyệt đối KHÔNG sử dụng câu mẫu hay dữ liệu cố định từ bất kỳ tài liệu thực tế nào của người dùng.
  - Kết xuất qua 22 biến thể font TrueType thực tế (Times New Roman, Georgia, Arial, Calibri, Tahoma, Verdana;
    đủ Regular, Bold, Italic, Bold-Italic; cỡ 11pt - 18pt).
  - Áp dụng mô phỏng camera chụp tài liệu thật: bóng đổ không đều (lighting gradient), nghiêng phối cảnh,
    cong nhẹ theo gáy sách, nhòe tiêu cự, nhiễu cảm biến và nén JPEG.
"""

from __future__ import annotations

import glob
import io
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Bảng âm tiết tiếng Việt chuẩn để tổng hợp từ ngẫu nhiên không phụ thuộc ngữ liệu cố định
ONSETS = [
    "", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh",
    "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s",
    "t", "th", "tr", "v", "x",
]

# Các nhóm vần mang dấu thanh hợp lệ trên nguyên âm chính
VOWEL_TONES = {
    "a": ["a", "à", "á", "ả", "ã", "ạ"],
    "ă": ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
    "â": ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
    "e": ["e", "è", "é", "ẻ", "ẽ", "ẹ"],
    "ê": ["ê", "ề", "ế", "ể", "ễ", "ệ"],
    "i": ["i", "ì", "í", "ỉ", "ĩ", "ị"],
    "o": ["o", "ò", "ó", "ỏ", "õ", "ọ"],
    "ô": ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
    "ơ": ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
    "u": ["u", "ù", "ú", "ủ", "ũ", "ụ"],
    "ư": ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
    "y": ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"],
}

OPEN_RHYMES = [
    ("a", ""), ("a", "i"), ("a", "o"), ("a", "u"), ("a", "y"), ("a", "m"), ("a", "n"), ("a", "ng"), ("a", "nh"),
    ("ă", "m"), ("ă", "n"), ("ă", "ng"),
    ("â", "m"), ("â", "n"), ("â", "ng"), ("â", "u"), ("â", "y"),
    ("e", ""), ("e", "m"), ("e", "n"), ("e", "o"),
    ("ê", ""), ("ê", "m"), ("ê", "n"), ("ê", "nh"), ("ê", "u"),
    ("i", ""), ("i", "a"), ("i", "m"), ("i", "n"), ("i", "nh"), ("i", "u"),
    ("o", ""), ("o", "a"), ("o", "i"), ("o", "m"), ("o", "n"), ("o", "ng"),
    ("ô", ""), ("ô", "i"), ("ô", "m"), ("ô", "n"), ("ô", "ng"),
    ("ơ", ""), ("ơ", "i"), ("ơ", "m"), ("ơ", "n"),
    ("u", ""), ("u", "a"), ("u", "i"), ("u", "m"), ("u", "n"), ("u", "ng"), ("u", "ô"), ("u", "y"),
    ("ư", ""), ("ư", "a"), ("ư", "i"), ("ư", "ng"), ("ư", "ơ"),
    ("y", "ê"),
]

# Các vần thanh sắc/nặng (âm tắc c/ch/p/t)
STOP_RHYMES = [
    ("a", "c"), ("a", "ch"), ("a", "p"), ("a", "t"),
    ("ă", "c"), ("ă", "p"), ("ă", "t"),
    ("â", "c"), ("â", "p"), ("â", "t"),
    ("e", "p"), ("e", "t"),
    ("ê", "ch"), ("ê", "p"), ("ê", "t"),
    ("i", "ch"), ("i", "p"), ("i", "t"),
    ("o", "c"), ("o", "p"), ("o", "t"),
    ("ô", "c"), ("ô", "p"), ("ô", "t"),
    ("ơ", "p"), ("ơ", "t"),
    ("u", "c"), ("u", "p"), ("u", "t"),
    ("ư", "c"), ("ư", "t"),
]


def discover_font_catalog(fonts_dir: str | None = None) -> Dict[Tuple[int, int, int], List[str]]:
    """
    Quét các font TrueType có sẵn trong `models/fonts` và ánh xạ theo khóa (is_serif, is_bold, is_italic).
    """
    if fonts_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fonts_dir = os.path.join(root_dir, "models", "fonts")

    catalog: Dict[Tuple[int, int, int], List[str]] = {}
    for k_s in (0, 1):
        for k_b in (0, 1):
            for k_i in (0, 1):
                catalog[(k_s, k_b, k_i)] = []

    def _register(path: str, is_serif: int, is_bold: int, is_italic: int):
        if os.path.exists(path):
            catalog[(is_serif, is_bold, is_italic)].append(path)

    # Serif: Times New Roman, Georgia
    serif_specs = [
        ("times.ttf", 0, 0), ("timesbd.ttf", 1, 0), ("timesi.ttf", 0, 1), ("timesbi.ttf", 1, 1),
        ("georgia.ttf", 0, 0), ("georgiab.ttf", 1, 0), ("georgiai.ttf", 0, 1), ("georgiaz.ttf", 1, 1),
    ]
    # Sans-serif: Arial, Calibri, Tahoma, Verdana
    sans_specs = [
        ("arial.ttf", 0, 0), ("arialbd.ttf", 1, 0), ("ariali.ttf", 0, 1), ("arialbi.ttf", 1, 1),
        ("calibri.ttf", 0, 0), ("calibrib.ttf", 1, 0), ("calibrii.ttf", 0, 1), ("calibriz.ttf", 1, 1),
        ("tahoma.ttf", 0, 0), ("tahomabd.ttf", 1, 0),
        ("verdana.ttf", 0, 0), ("verdanab.ttf", 1, 0), ("verdanai.ttf", 0, 1), ("verdanaz.ttf", 1, 1),
    ]

    for fname, b, it in serif_specs:
        _register(os.path.join(fonts_dir, fname), 1, b, it)
    for fname, b, it in sans_specs:
        _register(os.path.join(fonts_dir, fname), 0, b, it)

    # Nếu thiếu Sans Italic ở font nào đó, dùng chính Italic sẵn có trong họ sans
    if not catalog[(0, 1, 1)]:
        catalog[(0, 1, 1)] = catalog[(0, 0, 1)][:]
    if not catalog[(0, 0, 1)]:
        catalog[(0, 0, 1)] = catalog[(0, 0, 0)][:]

    return catalog


def random_vietnamese_syllable(rng: random.Random) -> str:
    """Tổng hợp ngẫu nhiên 1 âm tiết tiếng Việt có thanh điệu chuẩn."""
    onset = rng.choice(ONSETS)
    if rng.random() < 0.78:
        nuc, coda = rng.choice(OPEN_RHYMES)
        # Vần đôi uô/ươ khi ghép có âm cuối cần chuẩn chính tả
        if nuc == "u" and coda == "ô":
            v = rng.choice(["uô", "uồ", "uố", "uổ", "uỗ", "uộ"])
            coda = rng.choice(["c", "i", "m", "n", "ng", "p", "t"])
        elif nuc == "ư" and coda == "ơ":
            v = rng.choice(["ươ", "ườ", "ướ", "ưở", "ưỡ", "ượ"])
            coda = rng.choice(["c", "i", "m", "n", "ng", "p", "t"])
        elif nuc == "y" and coda == "ê":
            v = rng.choice(["yê", "yề", "yế", "yể", "yễ", "yệ"])
            coda = rng.choice(["n", "ng", "p", "t", "u"])
            onset = ""
        else:
            v = rng.choice(VOWEL_TONES[nuc])
    else:
        nuc, coda = rng.choice(STOP_RHYMES)
        tones = VOWEL_TONES[nuc]
        v = rng.choice([tones[2], tones[5]])  # thanh sắc hoặc nặng
    return f"{onset}{v}{coda}"


def random_vietnamese_line(
    rng: random.Random,
    min_words: int = 3,
    max_words: int = 12,
    allow_upper: bool = True,
    allow_numbering: bool = True,
) -> str:
    """Sinh một dòng văn bản tiếng Việt ngẫu nhiên (chữ thường, chữ hoa, dấu câu, đề mục)."""
    n_words = rng.randint(max(1, min_words), max(min_words, max_words))
    words = [random_vietnamese_syllable(rng) for _ in range(n_words)]

    mode = rng.random()
    if allow_numbering and mode < 0.14:
        # Dạng tiêu đề đề mục: 1. ..., 2.1. ..., 3.3.2. ...
        depth = rng.choice([1, 2, 3])
        nums = ".".join(str(rng.randint(1, 9)) for _ in range(depth))
        words[0] = words[0].capitalize()
        line = f"{nums}. " + " ".join(words)
    elif allow_upper and mode < 0.28:
        # Tiêu đề IN HOA toàn bộ
        k_up = rng.randint(min(3, len(words)), min(8, len(words)))
        line = " ".join(words[:k_up]).upper()
    else:
        words[0] = words[0].capitalize()
        # Chèn dấu câu tiếng Việt thực tế (, ; : . - "..." (...))
        out_tokens = []
        for idx, w in enumerate(words):
            r = rng.random()
            if idx < len(words) - 1:
                if r < 0.08:
                    out_tokens.append(f"{w},")
                elif r < 0.11:
                    out_tokens.append(f"{w};")
                elif r < 0.13:
                    out_tokens.append(f"{w} -")
                elif r < 0.14:
                    out_tokens.append(f"({w})")
                elif r < 0.17:
                    out_tokens.append(f'"{w}')
                elif r < 0.19:
                    out_tokens.append(str(rng.randint(1945, 2026)))
                else:
                    out_tokens.append(w)
            else:
                end_p = rng.choice(["", ".", ".", ".", ":", ";", "?"])
                out_tokens.append(f"{w}{end_p}")
        line = " ".join(out_tokens)
        if rng.random() < 0.12:
            line = rng.choice(["- ", "+ "]) + line
    return line


def apply_camera_optics(
    gray: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
    max_skew_deg: float = 2.0,
) -> np.ndarray:
    """
    Mô phỏng chân thực sai số quang học máy ảnh chụp trang sách/tài liệu:
      - Nghiêng nhẹ góc chụp ±max_skew_deg
      - Bóng đổ ánh sáng nghiêng (illumination gradient, tối dần về phía gáy sách)
      - Độ mờ ống kính (Gaussian / Slight defocus)
      - Nhiễu hạt cảm biến & nén JPEG
    """
    h, w = gray.shape[:2]
    arr = gray.astype(np.float32)

    # 1. Ánh sáng không đều (giống bóng gáy sách / ánh đèn phòng)
    if rng.random() < 0.88:
        x_grad = np.linspace(rng.uniform(-42.0, 12.0), rng.uniform(-12.0, 26.0), w, dtype=np.float32)[None, :]
        y_grad = np.linspace(rng.uniform(-18.0, 10.0), rng.uniform(-10.0, 18.0), h, dtype=np.float32)[:, None]
        arr = arr + x_grad + y_grad

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 2. Xoay nghiêng nhỏ ±max_skew_deg (giống trang sách đặt hơi lệch dưới camera)
    if max_skew_deg > 0 and rng.random() < 0.65:
        angle = rng.uniform(-max_skew_deg, max_skew_deg)
        bg_fill = int(np.percentile(arr, 90))
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg_fill)

    # 3. Nhòe ống kính camera (kernel 3x3 hoặc Gaussian nhẹ)
    r_blur = rng.random()
    if r_blur < 0.50:
        sigma = rng.uniform(0.35, 1.15)
        arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=sigma)
    elif r_blur < 0.65:
        # Motion blur nhẹ theo chiều ngang hoặc dọc 3px
        k = np.zeros((3, 3), dtype=np.float32)
        k[1, :] = 1.0 / 3.0
        arr = cv2.filter2D(arr, -1, k)

    # 4. Nhiễu cảm biến ISO + nén JPEG
    noise_sigma = rng.uniform(1.0, 5.5)
    noise = np_rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.55:
        q = rng.randint(68, 94)
        _, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        arr = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    return arr
