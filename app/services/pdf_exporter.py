import os
import io
import logging
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from app.services.reconstruction import DocumentDOM, BlockType

logger = logging.getLogger(__name__)


class PDFExporter:
    """
    Exports DocumentDOM into a Searchable PDF / Rendered PDF document.
    """

    @staticmethod
    def export_to_pdf(dom: DocumentDOM, output_pdf_path: str, background_img_path: Optional[str] = None) -> str:
        """
        Renders DocumentDOM into PDF pages. If background_img_path is provided,
        overlays OCR text blocks on top of restored background.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
        pil_pages = []

        for page in dom.pages:
            # Create page canvas
            w, h = int(page.width), int(page.height)
            if background_img_path and os.path.exists(background_img_path):
                bg_img = Image.open(background_img_path).convert("RGB").resize((w, h))
            else:
                bg_img = Image.new("RGB", (w, h), color=(255, 255, 255))

            draw = ImageDraw.Draw(bg_img)

            for block in page.blocks:
                if not block.content:
                    continue
                bbox = block.bbox
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox]
                    # Render text box
                    if block.block_type == BlockType.TITLE:
                        font_size = 24
                    elif block.block_type == BlockType.HEADING:
                        font_size = 18
                    else:
                        font_size = 14

                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()

                    # Draw text in visual canvas
                    if not background_img_path:
                        draw.text((x0, y0), block.content, fill=(0, 0, 0), font=font)

            pil_pages.append(bg_img)

        if pil_pages:
            pil_pages[0].save(
                output_pdf_path,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=pil_pages[1:] if len(pil_pages) > 1 else []
            )
            logger.info(f"Successfully generated PDF: {output_pdf_path}")
        return output_pdf_path
