import os
import json
import logging
from typing import List

from app.schemas.scan import (
    ScanResult,
    QualityModel,
    DocumentModel,
    OCRModel,
    FilesModel,
)
from app.schemas.defect import DefectModel

from app.services.image_service import ImageService
from app.services.ocr_service import OCRService
from app.services.quality_service import QualityService
from app.services.document_service import DocumentService
from app.services.defect_service import DefectService

from app.core.logging import Timer


logger = logging.getLogger(__name__)


class ScanService:
    """
    Orchestrates the document scanning pipeline.

    Pipeline:
        1. Validate and save uploaded file
        2. Detect document
        3. Analyze image quality
        4. Detect defects
        5. Extract / preprocess document
        6. OCR
        7. Save processed image
        8. Build and save scan result
    """

    def __init__(self):
        self.image_service = ImageService()
        self.ocr_service = OCRService()
        self.quality_service = QualityService()
        self.document_service = DocumentService()
        self.defect_service = DefectService()

    async def process_scan(self, scan_id: str, upload_file) -> ScanResult:
        timer = Timer()

        # ============================================================
        # 1. Validate file and save original image
        # ============================================================
        input_path, content = await self.image_service.validate_and_save(
            scan_id,
            upload_file
        )

        # ============================================================
        # 2. Detect document
        # ============================================================
        doc_info = self.document_service.detect_document_from_bytes(
            content
        )

        # ============================================================
        # 3. Analyze image quality
        # ============================================================
        quality = self.quality_service.analyze_bytes(
            content,
            doc_info
        )

        # ============================================================
        # 5. Extract / preprocess document
        # ============================================================
        processed_bytes = self.document_service.extract_document_bytes(
            content,
            doc_info
        )

        # ============================================================
        # 6. OCR
        # ============================================================
        ocr_res = self.ocr_service.extract_text_from_bytes(
            processed_bytes
        )

        # ============================================================
        # 7. Detect defects (include OCR results)
        # ============================================================
        defects = self.defect_service.detect_defects(
            content,
            doc_info,
            quality,
            ocr_res
        )

        # ============================================================
        # 7. Save processed image
        # ============================================================
        processed_filename = f"{scan_id}_processed.jpg"

        out_dir = os.path.join(
            "data",
            "output"
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        processed_path = os.path.join(
            out_dir,
            processed_filename
        )

        with open(
            processed_path,
            "wb"
        ) as f:
            f.write(processed_bytes)

        # ============================================================
        # 8. Build response models
        # ============================================================

        # Quality
        overall_quality = QualityModel(
            score=quality.get(
                "score",
                0.0
            ),
            level=quality.get(
                "level",
                "VERY_POOR"
            )
        )

        # Document
        document_model = DocumentModel(
            detected=doc_info.get(
                "detected",
                False
            ),
            confidence=doc_info.get(
                "confidence",
                0.0
            ),
            rotation=doc_info.get(
                "rotation",
                0
            ),
            skew_angle=doc_info.get(
                "skew_angle",
                0.0
            )
        )

        # Defects
        defects_models: List[DefectModel] = [
            DefectModel(**defect)
            for defect in defects
        ]

        # OCR
        ocr_model = OCRModel(
            text=ocr_res.get(
                "text",
                ""
            ),
            confidence=ocr_res.get(
                "confidence",
                0.0
            ),
            language=ocr_res.get(
                "language"
            )
        )

        # Files
        files = FilesModel(
            original=input_path,
            processed=processed_path
        )

        # Final result
        result = ScanResult(
            scan_id=scan_id,
            status="completed",
            quality=overall_quality,
            document=document_model,
            defects=defects_models,
            ocr=ocr_model,
            files=files
        )

        # ============================================================
        # 9. Persist scan result as JSON
        # ============================================================
        try:
            summary_path = os.path.join(
                out_dir,
                f"{scan_id}.json"
            )

            with open(
                summary_path,
                "w",
                encoding="utf-8"
            ) as fh:
                json.dump(
                    result.model_dump(),
                    fh,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception:
            logger.exception(
                "Failed to write summary for %s",
                scan_id
            )

        # ============================================================
        # 10. Logging
        # ============================================================
        logger.info(
            "Processed scan %s in %.2fms",
            scan_id,
            timer.elapsed_ms()
        )

        return result