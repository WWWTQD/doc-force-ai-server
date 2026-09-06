from typing import Tuple
import numpy as np
from PIL import Image


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # Convert RGB to BGR if needed
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[..., ::-1]
    return arr


def cv2_to_pil(img_arr) -> Image.Image:
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        img_arr = img_arr[..., ::-1]
    return Image.fromarray(img_arr)

