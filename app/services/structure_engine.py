import os
import time
import uuid
import logging
from typing import List, Dict, Any, Union
from PIL import Image
import cv2
import numpy as np

# PRE-IMPORT TORCH TO PREVENT WINDOWS DLL LOAD ERRORS (shm.dll) IN PADDLEX
try:
    import torch
except Exception:
    pass

logger = logging.getLogger("doc-force-ai-server")


class StructureEngine:
    """Singleton engine managing PP-StructureV3 pipeline lifecycle and inference."""

    _instance = None

    def __init__(self):
        self.pipeline = None
        self.device = "cpu"
        self.initialized = False

    @classmethod
    def get_instance(cls) -> "StructureEngine":
        if cls._instance is None:
            cls._instance = StructureEngine()
        return cls._instance

    def initialize(
        self,
        device: str = "cpu",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        use_seal_recognition: bool = False,
        use_table_recognition: bool = True,
        use_formula_recognition: bool = False,
    ):
        """Initializes PP-StructureV3 pipeline once during app lifespan."""
        if self.initialized and self.pipeline is not None:
            logger.info("[StructureEngine] Model already initialized.")
            return

        start_t = time.time()
        logger.info(f"[StructureEngine] Initializing PP-StructureV3 on device={device}...")

        self.device = device.lower()

        # Patch C++ static runner configuration to prevent PIR attribute crash on Windows CPU
        try:
            import paddle
            import paddle.inference as paddle_infer
            if hasattr(paddle_infer, "Config") and hasattr(paddle_infer.Config, "enable_mkldnn"):
                paddle_infer.Config.enable_mkldnn = lambda self, *args, **kwargs: None

            import paddlex.inference.models.runners.paddle_static.runner as paddle_runner
            if hasattr(paddle_runner, "resolve_paddle_static_engine_config"):
                orig_resolve = paddle_runner.resolve_paddle_static_engine_config

                def safe_resolve(model_name, config):
                    cfg = orig_resolve(model_name, config)
                    if cfg.get("device_type") == "cpu":
                        cfg["run_mode"] = "paddle"
                    return cfg

                paddle_runner.resolve_paddle_static_engine_config = safe_resolve
        except Exception as e:
            logger.warning(f"[StructureEngine] PaddleX CPU patch warning: {e}")

        try:
            from paddleocr import PPStructureV3

            kwargs = {
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "use_textline_orientation": use_textline_orientation,
                "use_seal_recognition": use_seal_recognition,
                "use_table_recognition": use_table_recognition,
                "use_formula_recognition": use_formula_recognition,
            }

            if self.device in ("gpu", "cuda"):
                kwargs["device"] = "gpu"

            self.pipeline = PPStructureV3(**kwargs)
            self.initialized = True
            elapsed = time.time() - start_t
            logger.info(f"[StructureEngine] PP-StructureV3 initialized successfully in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"[StructureEngine] Failed to load PP-StructureV3: {e}")
            raise e

    def predict(self, input_data: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """Runs PP-StructureV3 prediction on a single image file path, numpy ndarray, or PIL Image."""
        if not self.initialized or self.pipeline is None:
            raise RuntimeError("PP-StructureV3 model is not initialized.")

        logger.info(f"[AI Engine] Running PP-StructureV3 prediction on input type: {type(input_data).__name__}")
        results = list(self.pipeline.predict(input_data))

        parsed_results = []
        for res in results:
            if hasattr(res, "json") and res.json:
                val = res.json if isinstance(res.json, dict) else res.json
                parsed_results.append(val)
            elif hasattr(res, "to_dict"):
                parsed_results.append(res.to_dict())
            elif isinstance(res, dict):
                parsed_results.append(res)
            else:
                parsed_results.append(dict(res))

        return parsed_results

    def predict_raw(self, input_data: Union[str, np.ndarray, Image.Image]):
        """Runs PP-StructureV3 prediction and returns raw result objects."""
        if not self.initialized or self.pipeline is None:
            raise RuntimeError("PP-StructureV3 model is not initialized.")

        logger.info(f"[AI Engine] Running raw PP-StructureV3 prediction...")
        return list(self.pipeline.predict(input_data))

    def predict_image_path(self, image_path: str) -> List[Dict[str, Any]]:
        """Runs PP-StructureV3 prediction on a single image file path (backwards compatible)."""
        return self.predict(image_path)

    def predict_pdf_path(self, pdf_path: str, temp_dir: str = "data/input") -> List[Dict[str, Any]]:
        """Renders PDF pages into images and runs PP-StructureV3 prediction page-by-page."""
        if not self.initialized or self.pipeline is None:
            raise RuntimeError("PP-StructureV3 model is not initialized.")

        logger.info(f"[AI Engine] Rendering and predicting PDF: {pdf_path}")
        page_results = []

        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            logger.info(f"[AI Engine] PDF loaded with {total_pages} page(s)")

            for p_idx in range(total_pages):
                page = pdf[p_idx]
                image = page.render(scale=2.0).to_pil()

                temp_page_name = f"pdf_page_{uuid.uuid4().hex}_{p_idx+1}.png"
                temp_page_path = os.path.join(temp_dir, temp_page_name)
                image.save(temp_page_path, format="PNG")

                try:
                    res_list = self.predict_image_path(temp_page_path)
                    for res in res_list:
                        res["page_index"] = p_idx + 1
                        res["page_count"] = total_pages
                        res["width"] = image.width
                        res["height"] = image.height
                        page_results.append(res)
                finally:
                    if os.path.exists(temp_page_path):
                        os.remove(temp_page_path)

        except Exception as e:
            logger.error(f"[AI Engine] PDF processing error with pypdfium2: {e}")
            raise e

        return page_results
