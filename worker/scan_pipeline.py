"""
Scan Pipeline cho Doc Forge AI Worker.

Ghép toàn bộ chuỗi xử lý một trang ảnh chụp từ camera:
  ảnh gốc
    -> DocumentScanner (YOLO document_detector.pt: bắt tờ giấy, nắn phẳng, xoay đúng chiều)
    -> ImagePreprocessor (deskew, denoise, Magic Color khử bóng / làm trắng nền)
    -> DocumentRestorationUNet (RestorationService: khôi phục chất lượng ảnh, làm trắng nền & rõ nét chữ)
    -> OCREngine (YOLO layout_best_backup.pt + PaddleOCR + VietOCR tiếng Việt)
    -> rawLayoutJson

Dùng chung cho real_worker.py và scripts/evaluate_scanner.py.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

# Tắt MKLDNN & PIR API cho PaddlePaddle trên CPU Windows để không bị lỗi xung đột OneDnnContext
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import numpy as np

# Đảm bảo có thể import app.services.restoration_service từ thư mục gốc dự án
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKER_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKER_DIR not in sys.path:
    sys.path.insert(0, WORKER_DIR)

from document_scanner import DocumentScanner, ScanGeometry, map_layout_to_original, rotate_image, rotation_matrix
from job_progress import parse_job_options, parse_scan_options
from llm_postprocessor import crop_line, process_page
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor

LOG = logging.getLogger("scan_pipeline")


@dataclass
class PageScanResult:
    layout: dict  # toạ độ theo ảnh đã xử lý (restored_image)
    restored_image: np.ndarray  # ảnh đã nắn phẳng + khôi phục bằng U-Net, chính là ảnh đưa vào OCR
    geometry: ScanGeometry
    transform: np.ndarray  # 3x3: ảnh gốc -> restored_image

    def layout_in_original_coords(self) -> dict:
        """Bản sao layout với bbox ánh xạ về ảnh gốc (khi frontend chỉ có ảnh gốc)."""
        import copy
        return map_layout_to_original(copy.deepcopy(self.layout), self.transform, self.geometry.original_size)


class ScanPipeline:
    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        preprocessor: ImagePreprocessor | None = None,
        scanner: DocumentScanner | None = None,
        restoration_service: Any | None = None,
        enable_document_scan: bool = True,
        enable_restoration: bool = True,
    ):
        self.ocr_engine = ocr_engine or OCREngine(lang="vi")
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.enable_document_scan = enable_document_scan
        self.enable_restoration = enable_restoration
        self.restoration_service = restoration_service

        if self.enable_restoration and self.restoration_service is None:
            try:
                from app.services.restoration_service import RestorationService
                self.restoration_service = RestorationService.get_instance()
            except Exception as exc:
                LOG.warning("Không nạp được RestorationService trong ScanPipeline: %s", exc)

        scorer = self.ocr_engine.score_orientation if self.ocr_engine.can_score_orientation else None
        gate = getattr(self.ocr_engine, "verify_orientation_gate", None) if self.ocr_engine.can_score_orientation else None
        axis_det = getattr(self.ocr_engine, "detect_text_axis", None) if self.ocr_engine.can_score_orientation else None
        self.scanner = scanner or DocumentScanner(
            orientation_scorer=scorer,
            orientation_gate=gate,
            axis_detector=axis_det,
        )

    def _ensure_restoration_service(self) -> Any | None:
        if self.restoration_service is None:
            try:
                from app.services.restoration_service import RestorationService
                self.restoration_service = RestorationService.get_instance()
            except Exception as exc:
                LOG.warning("Không nạp được RestorationService theo yêu cầu: %s", exc)
        return self.restoration_service

    @staticmethod
    def _build_stroke_preserved_ocr_input(
        pre_unet_bgr: np.ndarray,
        unet_restored_bgr: np.ndarray,
        preprocessor: ImagePreprocessor,
    ) -> np.ndarray:
        """
        Tạo ảnh đầu vào tối ưu cho OCREngine (PaddleOCR + VietOCR) khi bật U-Net:
        - Dùng phiên bản khử nền mềm tự nhiên (Soft-Normalized luma, không cắt ngưỡng gắt,
          không gamma bóp nét) để giữ trọn 100% chi tiết nét thanh/dấu tiếng Việt cho VietOCR.
        - Chỉ mượn tín hiệu nền trắng từ U-Net tại vùng nền thực sự không chứa nét chữ
          (nhằm triệt tiêu watermark/thấu mực nhẹ mà không làm gãy hay bết chữ).
        """
        import cv2

        u_gray = cv2.cvtColor(unet_restored_bgr, cv2.COLOR_BGR2GRAY) if unet_restored_bgr.ndim == 3 else unet_restored_bgr
        h, w = u_gray.shape[:2]

        gray_in = preprocessor.neutralize_highlighter_luma(pre_unet_bgr) if pre_unet_bgr.ndim == 3 else pre_unet_bgr
        soft_norm, bg_safe = preprocessor.soft_normalize_luma(pre_unet_bgr, mode="dilate")
        raw_norm = np.clip((gray_in.astype(np.float32) / bg_safe) * 255.0, 0.0, 255.0)
        bleed_severity = preprocessor.estimate_page_bleed_severity(raw_norm)
        stroke_conf, local_drop, _ = preprocessor.compute_stroke_protection(
            soft_norm, h, w, bleed_severity=bleed_severity
        )

        u_f = u_gray.astype(np.float32)
        drop_ceiling = 8.0 + 14.0 * bleed_severity
        bg_only = (
            np.clip((0.04 - stroke_conf) / 0.04, 0.0, 1.0)
            * np.clip((drop_ceiling - local_drop) / max(8.0, drop_ceiling), 0.0, 1.0)
            * np.clip((soft_norm - 175.0) / 25.0, 0.0, 1.0)
            * np.clip((232.0 - soft_norm) / 18.0, 0.0, 1.0)
            * np.clip((u_f - 242.0) / 12.0, 0.0, 1.0)
        )
        blend_w = 0.45 + 0.50 * bleed_severity
        soft_target = np.clip(soft_norm + blend_w * np.maximum(0.0, u_f - soft_norm), 0.0, 255.0)
        ocr_gray = np.clip((1.0 - bg_only) * soft_norm + bg_only * soft_target, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2BGR)

    def process_image(
        self,
        img: np.ndarray,
        page_number: int,
        options: dict | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> PageScanResult:
        def _notify(stage: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(stage)
                except Exception as cb_exc:
                    LOG.debug("progress_cb(%s) lỗi: %s", stage, cb_exc)

        _notify("PREPROCESSING")
        h, w = img.shape[:2]
        if self.enable_document_scan:
            # Giữ nguyên 100% khung ảnh gốc (không cắt lẹm lề/đầu/cuối trang và bỏ qua YOLO detect_corners chậm trên CPU)
            # Chỉ thực hiện xoay tự động 0°/90°/180°/270° về đúng chiều đọc tiếng Việt (< 0.4s)
            warped_img = img
            warp_transform = np.eye(3, dtype=np.float64)
            method = "original"
            conf = 1.0
            corners_list = None

            if hasattr(self.scanner, "orient_with_verification"):
                rotation, orientation_verified = self.scanner.orient_with_verification(warped_img)
            else:
                rotation = self.scanner.orient(warped_img)
                orientation_verified = False
            if rotation:
                rh, rw = warped_img.shape[:2]
                warp_transform = rotation_matrix(rotation, rw, rh) @ warp_transform
                warped_img = rotate_image(warped_img, rotation)

            try:
                warped_img = self.preprocessor.clean_handwriting_and_book_gutter(
                    warped_img, remove_colored_pen=False
                )
            except Exception as exc:
                LOG.warning("Lỗi khi lọc rãnh gáy sách: %s", exc)

            LOG.info(
                "  [ScanPipeline] method=%s conf=%.2f rotation=%d° verified=%s (%dx%d -> %dx%d)",
                method, conf, rotation, orientation_verified, w, h, warped_img.shape[1], warped_img.shape[0],
            )
            geometry = ScanGeometry(
                image=warped_img,
                transform=warp_transform,
                original_size=(w, h),
                method=method,
                detection_confidence=conf,
                corners=corners_list,
                rotation=rotation,
                orientation_verified=orientation_verified,
            )
        else:
            geometry = ScanGeometry(image=img, transform=np.eye(3), original_size=(w, h), method="disabled")

        _notify("RESTORING")
        opts_dict = options if isinstance(options, dict) else {}
        if "useUnet" in opts_dict:
            want_unet = bool(opts_dict.get("useUnet"))
        elif str(opts_dict.get("restoreMode") or "").strip().lower() in ("unet", "unet_v5_1", "unet_hybrid"):
            want_unet = True
        else:
            want_unet = bool(self.enable_restoration)

        if want_unet and self.restoration_service is None:
            self._ensure_restoration_service()

        use_unet = (
            want_unet
            and self.restoration_service is not None
            and getattr(self.restoration_service, "is_available", False)
        )
        actual_restoration_mode = "opencv_magic"

        if use_unet:
            # Khi có DocumentRestorationUNet v5.1: chỉ xoay thẳng hình học (deskew),
            # giữ nguyên phân bố pixel tự nhiên đưa vào U-Net v5.1
            if getattr(self.preprocessor, "enable_deskew", True):
                try:
                    restored, deskew_matrix = self.preprocessor.deskew_with_matrix(geometry.image)
                except Exception as exc:
                    LOG.warning("Lỗi khi deskew ảnh: %s", exc)
                    restored, deskew_matrix = geometry.image, np.eye(3, dtype=np.float64)
            else:
                restored, deskew_matrix = geometry.image, np.eye(3, dtype=np.float64)
            transform = deskew_matrix @ geometry.transform
            pre_unet_table_img = restored

            try:
                LOG.info("  -> Khôi phục ảnh bằng DocumentRestorationUNet v5.1 (U-Net-Dominant + Non-Destructive Stroke Envelope)...")
                restored_h, restored_w = restored.shape[:2]
                unet_restored = self.restoration_service.restore_image_bgr(
                    restored,
                    tile_size=512,
                    overlap=64,
                    preserve_color=True,
                    max_side=None,
                    pre_cleaned=self.enable_document_scan,
                    force=True,
                )
                if unet_restored.shape[:2] != (restored_h, restored_w):
                    import cv2
                    unet_restored = cv2.resize(
                        unet_restored,
                        (restored_w, restored_h),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                restored = unet_restored
                ocr_input_img = self._build_stroke_preserved_ocr_input(
                    pre_unet_table_img, restored, self.preprocessor
                )
                actual_restoration_mode = "unet_v5_1"
            except Exception as exc:
                LOG.warning("  Cảnh báo: Khôi phục ảnh U-Net v5.1 gặp lỗi (%s), giữ nguyên ảnh camera tự nhiên cho OCR.", exc)
                restored, deskew_matrix = self.preprocessor.enhance(geometry.image)
                transform = deskew_matrix @ geometry.transform
                if max(pre_unet_table_img.shape[:2]) >= 900:
                    ocr_input_img = self._build_stroke_preserved_ocr_input(
                        pre_unet_table_img, restored, self.preprocessor
                    )
                else:
                    ocr_input_img = pre_unet_table_img
        else:
            # Tắt bilateralFilter + cắt ngưỡng cứng cho OCR: giữ trọn nét dấu thanh tiếng Việt sắc nét từ ảnh camera gốc
            import cv2
            restored, deskew_matrix = self.preprocessor.enhance(geometry.image)
            transform = deskew_matrix @ geometry.transform
            pre_unet_table_img = cv2.warpAffine(
                geometry.image,
                deskew_matrix[:2],
                (restored.shape[1], restored.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if max(pre_unet_table_img.shape[:2]) >= 900:
                ocr_input_img = self._build_stroke_preserved_ocr_input(
                    pre_unet_table_img, restored, self.preprocessor
                )
            else:
                ocr_input_img = pre_unet_table_img

        _notify("OCR")
        layout = self.ocr_engine.process_image(ocr_input_img, page_number, table_ref_img=pre_unet_table_img)
        scan_meta = geometry.to_metadata()
        scan_meta["restorationMode"] = actual_restoration_mode
        layout["scan"] = scan_meta

        post_opts = parse_scan_options(options, default_llm=True, default_unet=self.enable_restoration)
        if post_opts.get("llmPostprocess"):
            _notify("LLM_POSTPROCESS")
            LOG.info("  -> Hậu xử lý sửa lỗi chính tả tiếng Việt bằng Gemini (LLM Post-OCR)...")

            def _crop_provider(blk: dict) -> bytes | None:
                import cv2

                crop = crop_line(ocr_input_img, blk.get("bbox"))
                if crop is None:
                    return None
                ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
                return enc.tobytes() if ok else None

            def _reocr_provider(reocr_blocks: list[dict]) -> list[str]:
                if not hasattr(self.ocr_engine, "recognize_crop"):
                    return ["" for _ in reocr_blocks]
                outputs: list[str] = []
                for blk in reocr_blocks:
                    crop = crop_line(restored, blk.get("bbox"))
                    if crop is None:
                        outputs.append("")
                        continue
                    try:
                        txt, _ = self.ocr_engine.recognize_crop(crop)
                        outputs.append(txt or "")
                    except Exception as exc:
                        LOG.debug("Re-OCR crop lỗi: %s", exc)
                        outputs.append("")
                return outputs

            # Bảo toàn 100% style / tableData / bbox của từng block, chỉ cập nhật trường text đã sửa lỗi chính tả
            layout = process_page(
                layout,
                options={**post_opts, "autoApply": True},
                crop_provider=_crop_provider,
                reocr_provider=_reocr_provider,
            )
            pp_meta = layout.get("postprocessMeta") or {}
            LOG.info(
                "  ✓ Gemini Post-OCR: status=%s model=%s corrected_blocks=%s latency=%sms",
                pp_meta.get("status"),
                pp_meta.get("llmModel"),
                pp_meta.get("suggestionsCreated"),
                pp_meta.get("latencyMs"),
            )

        return PageScanResult(layout=layout, restored_image=restored, geometry=geometry, transform=transform)

    def process_bytes(
        self,
        image_bytes: bytes,
        page_number: int,
        options: dict | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> PageScanResult:
        return self.process_image(
            self.preprocessor.decode_image(image_bytes),
            page_number,
            options=options,
            progress_cb=progress_cb,
        )
