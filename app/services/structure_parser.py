import uuid
from typing import List, Dict, Any, Union
from app.schemas.structure import DocumentElement, DocumentPage, DocumentContent, AnalyzeResponse


class StructureParser:
    """Parses raw PP-StructureV3 output into PBL4 standardized document schema."""

    @staticmethod
    def normalize_label(label: str) -> tuple[str, str]:
        """Normalizes PaddleOCR category label to PBL4 standard element type."""
        if not label:
            return "unknown", "none"

        raw_label = str(label).strip().lower()

        mapping = {
            "title": "title",
            "doc_title": "title",
            "header": "header",
            "footer": "footer",
            "table": "table",
            "figure": "image",
            "image": "image",
            "equation": "formula",
            "formula": "formula",
            "chart": "chart",
            "seal": "seal",
            "text": "text",
            "normal_text": "text",
            "doc_text": "text",
            "paragraph": "paragraph",
            "reference": "text",
            "abstract": "text",
        }

        normalized = mapping.get(raw_label, raw_label if raw_label else "unknown")
        return normalized, raw_label

    @classmethod
    def parse(
        cls,
        raw_results: Union[List[Dict[str, Any]], Dict[str, Any]],
        file_id: str,
        filename: str,
        processing_time_ms: float,
    ) -> AnalyzeResponse:
        """Converts raw list of page dictionary results into an AnalyzeResponse."""
        if isinstance(raw_results, dict):
            raw_results = [raw_results]

        pages: List[DocumentPage] = []

        for p_idx, page_dict in enumerate(raw_results):
            page_num_raw = page_dict.get("page_index")
            if isinstance(page_num_raw, int) and page_num_raw > 0:
                page_num = page_num_raw
            else:
                page_num = p_idx + 1

            width = page_dict.get("width", 0)
            height = page_dict.get("height", 0)

            elements: List[DocumentElement] = []
            parsing_res_list = page_dict.get("parsing_res_list") or page_dict.get("res") or []

            if not isinstance(parsing_res_list, list):
                parsing_res_list = []

            element_counter = 1

            for item in parsing_res_list:
                if not isinstance(item, dict):
                    continue

                # Category / Label
                raw_cat = item.get("label") or item.get("category") or item.get("type") or ""
                norm_type, raw_type = cls.normalize_label(raw_cat)

                # Bounding box [xmin, ymin, xmax, ymax]
                bbox_raw = item.get("bbox") or item.get("box") or []
                bbox = []
                if isinstance(bbox_raw, (list, tuple)):
                    bbox = [float(coord) for coord in bbox_raw[:4]]

                # Confidence score
                score = float(item.get("score") or item.get("confidence") or 0.0)

                # Text content
                text = ""
                if "text" in item and item["text"] is not None:
                    text = str(item["text"])
                elif "content" in item and item["content"] is not None:
                    text = str(item["content"])
                elif "rec_res" in item and isinstance(item["rec_res"], list):
                    # Combine text line recognitions
                    text_parts = []
                    for rec_line in item["rec_res"]:
                        if isinstance(rec_line, (list, tuple)) and len(rec_line) > 0:
                            text_parts.append(str(rec_line[0]))
                        elif isinstance(rec_line, dict) and "text" in rec_line:
                            text_parts.append(str(rec_line["text"]))
                    text = "\n".join(text_parts)

                # Table HTML representation
                html = item.get("html") or item.get("table_html")

                # Table structured cell data if present
                table_data = item.get("cell_box_list") or item.get("structure")

                elem_id = f"element-{element_counter}"
                element_counter += 1

                elements.append(
                    DocumentElement(
                        id=elem_id,
                        type=norm_type,
                        raw_type=raw_type,
                        text=text.strip(),
                        bbox=bbox,
                        confidence=score,
                        html=html if isinstance(html, str) else None,
                        table_data={"cells": table_data} if table_data else None,
                        attributes={"raw_label": raw_cat},
                    )
                )

            # Fallback if parsing_res_list was empty but overall_ocr_res exists
            if not elements and "overall_ocr_res" in page_dict:
                ocr_res = page_dict.get("overall_ocr_res") or {}
                dt_polys = ocr_res.get("dt_polys") or ocr_res.get("boxes") or []
                rec_text = ocr_res.get("rec_text") or ocr_res.get("texts") or []
                rec_score = ocr_res.get("rec_score") or ocr_res.get("scores") or []

                for idx in range(min(len(dt_polys), len(rec_text))):
                    box_poly = dt_polys[idx]
                    poly_text = rec_text[idx]
                    poly_score = float(rec_score[idx]) if idx < len(rec_score) else 0.0

                    # Convert polygon to bounding box
                    if isinstance(box_poly, (list, tuple)) and len(box_poly) >= 4:
                        if isinstance(box_poly[0], (list, tuple)):
                            xs = [pt[0] for pt in box_poly]
                            ys = [pt[1] for pt in box_poly]
                            poly_bbox = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            poly_bbox = [float(c) for c in box_poly[:4]]
                    else:
                        poly_bbox = []

                    elements.append(
                        DocumentElement(
                            id=f"element-{element_counter}",
                            type="text",
                            raw_type="ocr_line",
                            text=str(poly_text).strip(),
                            bbox=poly_bbox,
                            confidence=poly_score,
                        )
                    )
                    element_counter += 1

            pages.append(
                DocumentPage(
                    page_number=page_num,
                    width=int(width),
                    height=int(height),
                    elements=elements,
                )
            )

        doc_content = DocumentContent(
            file_id=file_id,
            filename=filename,
            pages=pages,
        )

        return AnalyzeResponse(
            success=True,
            file_id=file_id,
            filename=filename,
            pages=pages,
            document=doc_content,
            processing_time_ms=round(processing_time_ms, 2),
            model="PP-StructureV3",
        )
