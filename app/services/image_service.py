import io
import os
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
from app.core.config import settings
from app.utils.file_utils import save_bytes_to_file
from app.core.exceptions import (
    FileTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
    InvalidFileError,
)


ALLOWED_EXT = {"jpg", "jpeg", "png", "tiff", "bmp"}


class ImageService:
    """Handles file validation and basic IO for images.

    Validates size, basic MIME/format, and attempts to decode with OpenCV.
    Saves original bytes to `data/input/{scan_id}.{ext}` and returns path and bytes.
    """

    async def validate_and_save(self, scan_id: str, upload_file) -> Tuple[str, bytes]:
        # read bytes
        content = await upload_file.read()

        # size check
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise FileTooLargeError(f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB} MB")

        # basic PIL detection to get format
        try:
            pil_img = Image.open(io.BytesIO(content))
            pil_img.verify()
            format_name = (pil_img.format or "JPEG").lower()
        except UnidentifiedImageError:
            raise InvalidImageError("Unable to identify image format")
        except Exception:
            raise InvalidImageError("Unable to decode uploaded image")

        # extension/format check
        if format_name not in ALLOWED_EXT:
            raise UnsupportedFormatError(f"Format {format_name} is not supported")

        # try decode with OpenCV to ensure valid image bytes
        try:
            arr = np.frombuffer(content, np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise InvalidImageError("OpenCV failed to decode image")
        except InvalidImageError:
            raise
        except Exception:
            raise InvalidImageError("OpenCV failed to decode image")

        # save to data/input with UUID filename (avoid user filename)
        ext = format_name if format_name != "jpeg" else "jpg"
        filename = f"{scan_id}.{ext}"
        out_path = os.path.join("data", "input")
        os.makedirs(out_path, exist_ok=True)
        saved_path = save_bytes_to_file(os.path.join(out_path, filename), content)
        return saved_path, content

