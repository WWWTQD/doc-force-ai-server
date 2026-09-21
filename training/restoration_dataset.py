import os
import csv
import math
from typing import Tuple, List, Optional, Dict
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def resize_with_padding(
    img: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Resize image keeping aspect ratio and pad with zeros to reach target_size.

    Returns:
        canvas: padded image of shape (target_h, target_w, 3)
        pad_info: (pad_top, pad_left, new_h, new_w) for later unpadding
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, (pad_top, pad_left, new_h, new_w)


def remove_padding(
    img: np.ndarray,
    pad_info: Tuple[int, int, int, int],
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Remove padding and resize back to original dimensions.

    Args:
        img: padded image of shape (target_h, target_w, 3)
        pad_info: (pad_top, pad_left, new_h, new_w) from resize_with_padding
        original_size: (orig_h, orig_w) to resize back to
    """
    pad_top, pad_left, new_h, new_w = pad_info
    cropped = img[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    orig_h, orig_w = original_size
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    return restored


class DocumentRestorationDataset(Dataset):
    """
    Dataset loader for Document Image Restoration model.
    Reads paired degraded/clean images from a CSV metadata file.

    CSV columns required: input, target
    Optional columns: source_id, degradation, blur_level, blur_type, kernel_size, sigma
    """

    def __init__(
        self,
        metadata_csv: str,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[T.Compose] = None,
        return_metadata: bool = False
    ):
        self.image_size = image_size
        self.return_metadata = return_metadata

        # Read CSV metadata
        self.pairs: List[Dict[str, str]] = []
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_path = row["input"].strip()
                target_path = row["target"].strip()
                # Normalize paths for current OS
                input_path = os.path.normpath(input_path)
                target_path = os.path.normpath(target_path)
                if os.path.exists(input_path) and os.path.exists(target_path):
                    self.pairs.append({
                        "input": input_path,
                        "target": target_path,
                        "source_id": row.get("source_id", ""),
                        "degradation": row.get("degradation", ""),
                    })

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),  # converts [H,W,C] uint8 to [C,H,W] float [0,1]
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        # Load images as numpy BGR, then convert to RGB
        deg_bgr = cv2.imread(pair["input"], cv2.IMREAD_COLOR)
        clean_bgr = cv2.imread(pair["target"], cv2.IMREAD_COLOR)

        if deg_bgr is None:
            # Fallback: try PIL for formats like .jfif/.webp
            deg_pil = Image.open(pair["input"]).convert("RGB")
            deg_bgr = cv2.cvtColor(np.array(deg_pil), cv2.COLOR_RGB2BGR)
        if clean_bgr is None:
            clean_pil = Image.open(pair["target"]).convert("RGB")
            clean_bgr = cv2.cvtColor(np.array(clean_pil), cv2.COLOR_RGB2BGR)

        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

        # Resize with aspect ratio preservation + padding
        deg_padded, _ = resize_with_padding(deg_rgb, self.image_size)
        clean_padded, _ = resize_with_padding(clean_rgb, self.image_size)

        # Apply transforms (ToTensor converts uint8 [0,255] to float [0,1])
        deg_tensor = self.transform(deg_padded)
        clean_tensor = self.transform(clean_padded)

        if self.return_metadata:
            return deg_tensor, clean_tensor, pair

        return deg_tensor, clean_tensor
