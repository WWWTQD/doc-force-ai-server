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
    _preprocessor = None

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
            checkpoint = torch.load(resolved_path, map_location=self._device, weights_only=True)

            state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

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

    def _get_preprocessor(self):
        if self._preprocessor is not None:
            return self._preprocessor
        import sys
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        worker_dir = os.path.join(root_dir, "worker")
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        if worker_dir not in sys.path:
            sys.path.insert(0, worker_dir)
        try:
            from worker.preprocessor import ImagePreprocessor
        except ImportError:
            from preprocessor import ImagePreprocessor
        self._preprocessor = ImagePreprocessor()
        return self._preprocessor

    def _infer_single_pass(self, gray_work: np.ndarray) -> np.ndarray:
        """Run a single padded forward pass through U-Net using torch.inference_mode()."""
        h_w, w_w = gray_work.shape[:2]
        pad_h = (8 - (h_w % 8)) % 8
        pad_w = (8 - (w_w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(gray_work, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            img_pad = gray_work

        tensor_in = (
            torch.from_numpy(img_pad).unsqueeze(0).unsqueeze(0).float().to(self._device) / 255.0
        ).clamp(0.0, 1.0)
        with torch.inference_mode():
            pred = self._model(tensor_in)
            pred = torch.nan_to_num(pred.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        out = (pred.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
        return out[:h_w, :w_w]

    @staticmethod
    def _stroke_preserving_guided_fusion(
        y_unet: np.ndarray,
        clean_bgr: np.ndarray,
        prep,
        precomputed_soft_norm: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        U-Net v5.1 True U-Net-Dominant + High-Frequency Native Pixel Preservation:
        - `y_unet` (DocumentRestorationUNet, resized via INTER_LANCZOS4) decides the clean white
          background map and regions where stains, bleed-through, shadows, and watermarks are erased.
        - Multi-scale `soft_normalize_luma` neutralizes multi-line highlighter bands and back-side
          bleed-through ghosts so they become 100% pure white (255).
        - At all text strokes and stroke transition boundaries, preserves 1:1 native pixel sharpness
          directly from the original image (zero U-Net bottleneck blur) with subtle high-frequency
          edge enhancement so stroke edge sharpness is >= 100% of the original image.
        """
        gray_orig = (
            prep.neutralize_highlighter_luma(clean_bgr)
            if clean_bgr.ndim == 3
            else clean_bgr
        )
        h, w = gray_orig.shape[:2]
        if y_unet.shape[:2] != (h, w):
            y_unet = cv2.resize(y_unet, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if precomputed_soft_norm is not None:
            soft_norm, bg_safe = precomputed_soft_norm
        else:
            soft_norm, bg_safe = prep.soft_normalize_luma(clean_bgr, mode="dilate")
        gray_f = gray_orig.astype(np.float32)
        raw_norm = np.clip((gray_f / bg_safe) * 255.0, 0.0, 255.0)
        bleed_severity = prep.estimate_page_bleed_severity(raw_norm)
        stroke_conf, local_drop, _ = prep.compute_stroke_protection(
            soft_norm, h, w, bleed_severity=bleed_severity
        )

        u_f = y_unet.astype(np.float32) / 255.0
        s_f = soft_norm / 255.0

        # Prevent U-Net from re-darkening multi-line highlighter bands or suppressed bleed-through ghosts
        # where soft_norm is already bright (>= 200) and edge-aware stroke signal is absent.
        u_f = np.where(
            (soft_norm >= 200.0) & (local_drop < 14.0) & (stroke_conf < 0.05),
            np.maximum(u_f, s_f),
            u_f,
        )

        # 1. U-Net dominant background restoration (82% U-Net + 18% native soft-normalized reflectance).
        bg_mix = 0.82 * u_f + 0.18 * s_f
        soft_paper_floor = 0.40 * u_f + 0.60 * s_f
        non_stroke_w = np.clip((0.28 - stroke_conf) / 0.28, 0.0, 1.0)
        bg_evidence = bg_mix + non_stroke_w * np.maximum(0.0, soft_paper_floor - bg_mix)

        # Smooth quadratic lift to pure white (gentler threshold on clean pages: Do No Harm)
        lift_lo = 0.84 - 0.08 * bleed_severity
        lift_span = 0.11 + 0.04 * bleed_severity
        lift_t = np.clip((bg_evidence - lift_lo) / lift_span, 0.0, 1.0)
        y_unet_bg_cleaned = bg_evidence + (lift_t * lift_t) * (1.0 - bg_evidence)
        y_unet_bg_cleaned = np.where(
            (u_f >= 0.93) & (stroke_conf < 0.05) & (local_drop < 12.0),
            np.maximum(y_unet_bg_cleaned, 0.99),
            y_unet_bg_cleaned,
        )

        # 2. High-Frequency Native Pixel Preservation at all strokes & stroke-edge transition boundaries
        grad_x = cv2.Sobel(soft_norm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(soft_norm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(grad_x, grad_y)
        # Edge transition band around character boundaries; crisp light-gray strokes on clean pages
        # are preserved even when downscaled U-Net misses 1-2px thin glyphs (e.g. '/', ':', '-').
        unet_bg_veto = np.clip((0.97 - u_f) / 0.12, 0.0, 1.0)
        crisp_override = (
            (1.0 - 0.85 * bleed_severity)
            * np.clip((grad_mag - 45.0) / 55.0, 0.0, 1.0)
            * np.clip((local_drop - 14.0) / 10.0, 0.0, 1.0)
        )
        edge_gate = np.maximum(unet_bg_veto, crisp_override)
        upper_soft = 234.0 - 16.0 * bleed_severity
        edge_band = (
            np.clip((grad_mag - 24.0) / 50.0, 0.0, 1.0)
            * np.clip((local_drop - 8.0) / 14.0, 0.0, 1.0)
            * edge_gate
            * np.clip((upper_soft - soft_norm) / 28.0, 0.0, 1.0)
        )

        if max(h, w) < 900:
            faint_lowres = np.clip((local_drop - 11.0) / 16.0, 0.0, 1.0) * np.clip((240.0 - soft_norm) / 22.0, 0.0, 1.0)
            stroke_conf = np.maximum(stroke_conf, faint_lowres)
        unet_guided = (
            np.clip((0.78 - u_f) / 0.20, 0.0, 1.0)
            * np.clip((local_drop - 20.0) / 18.0, 0.0, 1.0)
            * np.clip((195.0 - soft_norm) / 25.0, 0.0, 1.0)
        )
        stroke_mask = np.clip(np.maximum(stroke_conf, np.maximum(unet_guided, edge_band)), 0.0, 1.0)

        # Extract 1:1 native high-frequency edge detail directly from original & soft-normalized image
        hf_orig = gray_f - cv2.GaussianBlur(gray_f, (0, 0), sigmaX=0.9)
        hf_soft = soft_norm - cv2.GaussianBlur(soft_norm, (0, 0), sigmaX=0.9)
        hf_detail = 0.60 * hf_soft + 0.40 * hf_orig

        s_dark = np.power(np.clip(s_f / 0.96, 0.0, 1.0), 1.80) * 255.0
        drop_pull = np.where(stroke_conf >= 0.35, 0.72, 0.55)
        s_dark = np.minimum(s_dark, np.clip(soft_norm - drop_pull * local_drop, 0.0, 255.0))
        # Only pull down toward raw gray_f when local background is normal bright paper (bg_safe >= 190),
        # never inside a darker highlighter band where gray_f has a constant dye offset.
        s_dark = np.where(
            (local_drop >= 20.0) & (gray_f < 195.0) & (bg_safe >= 190.0),
            np.minimum(s_dark, gray_f),
            s_dark,
        )
        s_sharp = np.clip(s_dark + 0.55 * hf_detail, 0.0, 255.0)

        stroke_envelope = (1.0 - stroke_mask) * 255.0 + stroke_mask * s_sharp

        # 3. Minimum envelope preserves 100% stroke darkness & native edge sharpness on U-Net white background
        bg_clean_255 = y_unet_bg_cleaned * 255.0
        y_fused = np.minimum(bg_clean_255, stroke_envelope)
        edge_ring = np.clip(edge_band * (1.0 - 0.4 * stroke_conf), 0.0, 1.0)
        y_out = np.clip(y_fused + 0.35 * edge_ring * hf_detail, 0.0, 255.0)
        return y_out.astype(np.uint8)

    def restore_image_bgr(
        self,
        img_bgr: np.ndarray,
        tile_size: Optional[int] = None,
        overlap: Optional[int] = None,
        preserve_color: Optional[bool] = None,
        max_side: Optional[int] = None,
        pre_cleaned: bool = False,
        force: bool = False,
    ) -> np.ndarray:
        """
        Runs U-Net v5.1 (U-Net-Dominant + Non-Destructive Stroke Envelope) restoration on an image (BGR uint8):
        1. Pre-cleans only outer book gutter (< 4% width) via `clean_handwriting_and_book_gutter(remove_colored_pen=False)`.
        2. Upscales small images (`max(h, w) < 900`) by 1.5x-2.0x (`INTER_LANCZOS4`) before DocumentRestorationUNet
           to prevent erasing thin strokes, or uses `working_max_side = 768` on CPU for large scans (> 1000px)
           for fast ~10-13s U-Net inference while preserving 1:1 native sharpness at full `(orig_h, orig_w)`.
        3. Applies U-Net-dominant background cleaning + Non-Destructive Stroke Envelope fusion.
        Falls back to returning original img_bgr if model is not loaded or on error.
        """
        if not force and not settings.ENABLE_IMAGE_RESTORATION:
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
                bgr_input = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            else:
                bgr_input = img_bgr

            t0 = time.perf_counter()
            prep = self._get_preprocessor()

            # Step 1: Only clean extreme outer book gutter (< 4% width); preserve red stamps & signatures
            if not pre_cleaned and bgr_input.ndim == 3:
                clean_bgr = prep.clean_handwriting_and_book_gutter(bgr_input, remove_colored_pen=False)
            else:
                clean_bgr = bgr_input

            gray = prep.neutralize_highlighter_luma(clean_bgr)
            orig_h, orig_w = gray.shape[:2]
            soft_norm_full, bg_safe_full = prep.soft_normalize_luma(clean_bgr, mode="dilate")

            # On medium/large scans, feed soft_norm_full (with highlighter bands & bleed-through ghosts neutralized)
            # into U-Net so U-Net never mistakes a highlighter band or bleed-through smudge for foreground ink.
            cur_max_side = max(orig_h, orig_w)
            if cur_max_side >= 900:
                unet_input_gray = np.clip(soft_norm_full, 0.0, 255.0).astype(np.uint8)
            else:
                unet_input_gray = gray

            # Step 2: Adaptive U-Net working resolution
            # Small images (max(h, w) < 900) are upscaled 1.5x-2.0x via INTER_LANCZOS4 so thin strokes are never eroded;
            # large images (> 1000px) use working_max_side = 640 on CPU (~10-13s inference) while preserving 1:1 native detail.
            if cur_max_side < 900:
                up_scale = min(2.0, max(1.5, 960.0 / float(max(1, cur_max_side))))
                if self._device.type == "cpu" and cur_max_side * up_scale > 1152:
                    up_scale = max(1.4, 1152.0 / float(cur_max_side))
                w_work = max(64, int(round(orig_w * up_scale)))
                h_work = max(64, int(round(orig_h * up_scale)))
                gray_work = cv2.resize(unet_input_gray, (w_work, h_work), interpolation=cv2.INTER_LANCZOS4)
            else:
                work_max_side = max_side or (640 if self._device.type == "cpu" else 1536)
                if cur_max_side > work_max_side:
                    scale = float(work_max_side) / float(cur_max_side)
                    w_work = max(64, int(round(orig_w * scale)))
                    h_work = max(64, int(round(orig_h * scale)))
                    gray_work = cv2.resize(unet_input_gray, (w_work, h_work), interpolation=cv2.INTER_AREA)
                else:
                    h_work, w_work = orig_h, orig_w
                    gray_work = unet_input_gray

            if self._device.type == "cpu":
                try:
                    torch.set_num_threads(4)
                except Exception:
                    pass

            if max(h_work, w_work) <= 1200:
                y_work = self._infer_single_pass(gray_work)
            else:
                from training.restoration_dataset import infer_single_image_tiled
                with torch.inference_mode():
                    y_work = infer_single_image_tiled(
                        self._model,
                        gray_work,
                        self._device,
                        tile_size=t_size or 512,
                        overlap=ovlap or 64,
                        refine_glyphs=False,
                    )

            if y_work.shape[:2] != (orig_h, orig_w):
                y_unet = cv2.resize(y_work, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                y_unet = y_work

            # Step 3: U-Net-dominant background cleaning + High-Frequency Native Pixel Preservation
            y_clean = self._stroke_preserving_guided_fusion(
                y_unet,
                clean_bgr,
                prep,
                precomputed_soft_norm=(soft_norm_full, bg_safe_full),
            )
            if y_clean.shape[:2] != (orig_h, orig_w):
                y_clean = cv2.resize(y_clean, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            dur_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "[RestorationService v5.1] Restored shape %s (UNet working %s) in %.1fms",
                (orig_h, orig_w),
                gray_work.shape,
                dur_ms,
            )

            if not keep_color:
                return cv2.cvtColor(y_clean, cv2.COLOR_GRAY2BGR)

            # Preserve genuine vivid colored ink (red stamps, blue signatures) without re-introducing yellow paper tint
            ycrcb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2YCrCb)
            _, cr_chan, cb_chan = cv2.split(ycrcb)
            cr_f = cr_chan.astype(np.float32)
            cb_f = cb_chan.astype(np.float32)
            # Local paper chroma baseline so aged yellow/brown paper itself stays neutral white
            cr_bg = cv2.GaussianBlur(cr_f, (0, 0), sigmaX=max(11.0, min(orig_h, orig_w) * 0.03))
            cb_bg = cv2.GaussianBlur(cb_f, (0, 0), sigmaX=max(11.0, min(orig_h, orig_w) * 0.03))
            chroma_rel = np.hypot(cr_f - cr_bg, cb_f - cb_bg)
            chroma_sat = np.hypot(cr_f - 128.0, cb_f - 128.0)
            is_vivid_color = (chroma_sat > 24.0) & (chroma_rel > 18.0)
            is_blue_pen_color = (cb_f - cr_f > 6.0) & (chroma_sat > 10.5) & (chroma_rel > 10.5) & (y_clean < 215)
            color_fg_mask = (y_clean < 228) & (is_vivid_color | is_blue_pen_color)
            # Keep natural luminance on colored stamps/signatures so red stays vivid crimson while
            # overlapping neutral black text (e.g. "TS. Huỳnh Hữu Hưng") stays deep black.
            y_color_nat = np.clip(soft_norm_full * 0.90, 85.0, 212.0)
            y_out_final = np.where(
                color_fg_mask,
                np.maximum(y_clean.astype(np.float32), y_color_nat),
                y_clean.astype(np.float32),
            )
            y_final_u8 = np.clip(y_out_final, 0.0, 255.0).astype(np.uint8)
            cr_clean = np.where(color_fg_mask, cr_chan, 128).astype(np.uint8)
            cb_clean = np.where(color_fg_mask, cb_chan, 128).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([y_final_u8, cr_clean, cb_clean]), cv2.COLOR_YCrCb2BGR)

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

            success, encoded_buf = cv2.imencode(".jpg", restored_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            if success:
                return encoded_buf.tobytes()
            return image_bytes
        except Exception as e:
            logger.warning("[RestorationService] Exception in restore_image_bytes: %s. Returning raw bytes.", e)
            return image_bytes
