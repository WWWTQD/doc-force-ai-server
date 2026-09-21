from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    SEAL = "seal"
    CHART = "chart"
    UNKNOWN = "unknown"


class BlockNode(BaseModel):
    id: str
    block_type: BlockType
    content: str
    bbox: Optional[List[float]] = None
    score: float = 1.0
    table_html: Optional[str] = None
    table_cells: Optional[List[Dict[str, Any]]] = None
    image_bytes: Optional[bytes] = None


class PageNode(BaseModel):
    page_index: int
    width: float
    height: float
    blocks: List[BlockNode] = []


class DocumentDOM(BaseModel):
    title: str = "Document"
    pages: List[PageNode] = []


class DocumentReconstructionEngine:
    """
    Converts PP-StructureV3 prediction output into a unified intermediate Document DOM representation.
    """

    @staticmethod
    def build_dom(pp_results: List[Dict[str, Any]], title: str = "Reconstructed Document") -> DocumentDOM:
        pages: List[PageNode] = []

        for p_idx, page_data in enumerate(pp_results, start=1):
            width = float(page_data.get("width", 800))
            height = float(page_data.get("height", 1100))
            blocks: List[BlockNode] = []

            parsing_list = page_data.get("parsing_res_list", [])
            if not parsing_list:
                # fallback to overall_ocr_res
                ocr_info = page_data.get("overall_ocr_res", {})
                rec_texts = ocr_info.get("rec_texts", [])
                rec_boxes = ocr_info.get("rec_boxes", [])
                rec_scores = ocr_info.get("rec_scores", [])
                for idx, text in enumerate(rec_texts):
                    bbox = rec_boxes[idx] if idx < len(rec_boxes) else None
                    score = rec_scores[idx] if idx < len(rec_scores) else 1.0
                    blocks.append(BlockNode(
                        id=f"b_{p_idx}_{idx}",
                        block_type=BlockType.PARAGRAPH,
                        content=text,
                        bbox=list(bbox) if bbox is not None else None,
                        score=float(score)
                    ))
            else:
                for idx, item in enumerate(parsing_list):
                    label = str(item.get("block_label", item.get("label", ""))).lower()
                    content = item.get("block_content", item.get("content", ""))
                    bbox = item.get("block_bbox", item.get("bbox", None))

                    b_type = BlockType.PARAGRAPH
                    if "title" in label or "header" in label:
                        b_type = BlockType.TITLE if "doc_title" in label else BlockType.HEADING
                    elif "table" in label:
                        b_type = BlockType.TABLE
                    elif "image" in label or "figure" in label:
                        b_type = BlockType.IMAGE
                    elif "formula" in label or "equation" in label:
                        b_type = BlockType.FORMULA
                    elif "seal" in label:
                        b_type = BlockType.SEAL

                    blocks.append(BlockNode(
                        id=f"b_{p_idx}_{idx}",
                        block_type=b_type,
                        content=content,
                        bbox=list(bbox) if bbox is not None else None,
                        table_html=item.get("table_html")
                    ))

            pages.append(PageNode(
                page_index=p_idx,
                width=width,
                height=height,
                blocks=blocks
            ))

        return DocumentDOM(title=title, pages=pages)
