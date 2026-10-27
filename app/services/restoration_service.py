import io
import os
import time
import logging
from typing import Optional, Union
import numpy as np
import cv2
from PIL import Image

# Đặt số luồng OpenMP ở mức module/tiến trình
os.environ["OMP_NUM_THREADS"] = "4"

try:
    import torch
    try:
        torch.set_num_threads(4)
    except Exception:
        pass
except ImportError:
    torch = None

from app.core.config import settings

logger = logging.getLogger("doc-force-ai-server")


class RestorationService:
    """
    Singleton service managing the DocumentRestorationUNet deep learning model.
    Restores degraded document images (removes uneven lighting, shadows, noise,
    sharpens faded ink and text glyphs).
    """

    _instance: Optional["RestorationService"] = None
    _model = None
    _device: Optional["torch.device"] = None
    _model_path: Optional[str] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RestorationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Đảm bảo cấu hình luồng 1 lần duy nhất ở khởi tạo
        if not hasattr(self, "_threads_configured"):
            os.environ["OMP_NUM_THREADS"] = "4"
            if torch is not None:
                try:
                    torch.set_num_threads(4)
                except Exception:
                    pass
            self._threads_configured = True

    @classmethod
    def get_instance(cls) -> "RestorationService":
        if cls._instance is None:
            cls._instance = RestorationService()
        return cls._instance

    def _resolve_model_path(self) -> Optional[str]:
        candidates = [
            settings.RESTORATION_MODEL_PATH,
            os.path.join(os.path.dirname(__file__), "..", "..", settings.RESTORATION_MODEL_PATH),
            os.path.join(os.getcwd(), settings.RESTORATION_MODEL_PATH),
            "models/restoration_best.pth",
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "restoration_best.pth"),
        ]
        for path in candidates:
            norm = os.path.normpath(os.path.abspath(path))
            if os.path.exists(norm) and os.path.isfile(norm):
                return norm
        return None

    def initialize(self, model_path: Optional[str] = None, device_name: Optional[str] = None):
        """
        Initializes the model once. Safe to call multiple times.
        """
        if self._initialized and self._model is not None:
            return self._model

        if torch is None:
            logger.warning("[RestorationService] PyTorch is not installed. Restoration will be bypassed.")
            return None

        # Resolve device
        target_device = device_name or settings.AI_DEVICE
        if target_device in ("gpu", "cuda") and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        resolved_path = model_path or self._resolve_model_path()
        self._model_path = resolved_path

        if not resolved_path or not os.path.exists(resolved_path):
            logger.warning(
                "[RestorationService] Model checkpoint not found at '%s'. Falling back to pass-through.",
                resolved_path or settings.RESTORATION_MODEL_PATH,
            )
            self._model = None
            self._initialized = True
            return None

        try:
            logger.info(
                "[RestorationService] Loading DocumentRestorationUNet on %s from: %s",
                self._device,
                resolved_path,
            )
            t0 = time.perf_counter()

            from training.train_restoration import DocumentRestorationUNet

            model = DocumentRestorationUNet(in_channels=1, out_channels=1).to(self._device)
            checkpoint = torch.load(resolved_path, map_location=self._device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self._model = model
            self._initialized = True
            load_time = (time.perf_counter() - t0) * 1000.0
            logger.info("[RestorationService] Model loaded successfully in %.1fms.", load_time)
            return self._model
        except Exception as e:
            logger.warning("[RestorationService] Failed to load restoration model: %s. Will fallback to original.", e)
            self._model = None
            self._initialized = True
            return None

    @property
    def is_available(self) -> bool:
        if not self._initialized:
            self.initialize()
        return self._model is not None

    def restore_image_bgr(
        self,
        img_bgr: np.ndarray,
        tile_size: Optional[int] = None,
        overlap: Optional[int] = None,
        preserve_color: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Runs tiled inference through DocumentRestorationUNet on an image (BGR uint8).
        Falls back to returning original img_bgr if model is not loaded or on error.
        Optimized for CPU: max_side=1280 caps tile count (4-6 tiles) to finish in < 8s.
        """
        if not settings.ENABLE_IMAGE_RESTORATION:
            return img_bgr

        if not self._initialized:
            self.initialize()

        if self._model is None or torch is None:
            return img_bgr

        t_size = tile_size or settings.RESTORATION_TILE_SIZE
        ovlap = overlap or settings.RESTORATION_OVERLAP
        keep_color = settings.RESTORATION_PRESERVE_COLOR if preserve_color is None else preserve_color

        try:
            if img_bgr.ndim == 2:
                gray = img_bgr
                bgr_input = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            else:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                bgr_input = img_bgr

            from training.restoration_dataset import infer_single_image_tiled

            orig_h, orig_w = gray.shape[:2]
            t0 = time.perf_counter()

            # Giữ độ phân giải chuẩn cho VietOCR và tối ưu tốc độ CPU (< 8s):
            # Nếu max(orig_h, orig_w) > 1280, resize working copy về max_side = 1280 (chỉ mất 4-6 tiles thay vì 25+ tiles)
            max_side = max(orig_h, orig_w)
            if max_side > 1280:
                scale = 1280.0 / float(max_side)
                w_work = max(64, int(round(orig_w * scale)))
                h_work = max(64, int(round(orig_h * scale)))
                gray_work = cv2.resize(gray, (w_work, h_work), interpolation=cv2.INTER_AREA)
            else:
                gray_work = gray

            # Chạy thuần túy U-Net (Pure DocumentRestorationUNet) không gọi lại torch.set_num_threads để tránh xung đột
            with torch.no_grad():
                y_work = infer_single_image_tiled(
                    self._model,
                    gray_work,
                    self._device,
                    tile_size=t_size or 512,
                    overlap=ovlap or 64,
                    refine_glyphs=False,
                )

            # Đảm bảo kết quả trả về đúng kích thước gốc (orig_w, orig_h) 100%
            if y_work.shape[:2] != (orig_h, orig_w):
                y_clean = cv2.resize(y_work, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            else:
                y_clean = y_work

            dur_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("[RestorationService] Restored image shape %s (working %s) in %.1fms", (orig_h, orig_w), gray_work.shape, dur_ms)

            if not keep_color:
                return cv2.cvtColor(y_clean, cv2.COLOR_GRAY2BGR)

            # Preserve colorful ink (stamps, signatures) using YCrCb chroma channel
            ycrcb = cv2.cvtColor(bgr_input, cv2.COLOR_BGR2YCrCb)
            _, cr_chan, cb_chan = cv2.split(ycrcb)
            chroma_sat = np.hypot(cr_chan.astype(np.float32) - 128.0, cb_chan.astype(np.float32) - 128.0)
            color_fg_mask = (y_clean < 245) & (chroma_sat > 18.0)
            cr_clean = np.where(color_fg_mask, cr_chan, 128).astype(np.uint8)
            cb_clean = np.where(color_fg_mask, cb_chan, 128).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([y_clean, cr_clean, cb_clean]), cv2.COLOR_YCrCb2BGR)

        except Exception as ex:
            logger.warning("[RestorationService] Restoration inference exception: %s. Using original.", ex)
            return img_bgr

    def restore_image_bytes(
        self,
        image_bytes: bytes,
        tile_size: Optional[int] = None,
        overlap: Optional[int] = None,
        preserve_color: Optional[bool] = None,
    ) -> bytes:
        """
        Receives raw image bytes, restores the image, and encodes back to JPEG bytes.
        Returns original bytes safely on any error.
        """
        if not settings.ENABLE_IMAGE_RESTORATION:
            return image_bytes

        if not self._initialized:
            self.initialize()

        if self._model is None:
            return image_bytes

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return image_bytes

            restored_bgr = self.restore_image_bgr(
                img_bgr,
                tile_size=tile_size,
                overlap=overlap,
                preserve_color=preserve_color,
            )

            success, encoded_buf = cv2.imencode(".jpg", restored_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                return encoded_buf.tobytes()
            return image_bytes
        except Exception as e:
            logger.warning("[RestorationService] Exception in restore_image_bytes: %s. Returning raw bytes.", e)
            return image_bytes
