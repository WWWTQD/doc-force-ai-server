import io
from typing import Dict
from PIL import Image
import pytesseract


class OCRService:
    """Simple OCR wrapper around pytesseract."""

    def extract_text_from_bytes(self, data: bytes, lang: str = None) -> Dict:
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            lang = lang or "eng"
            # image_to_data returns box info and confidence per word
            try:
                data_out = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            except pytesseract.TesseractError as e:
                # language data might be missing
                return {"text": "", "confidence": 0.0, "language": lang, "error": str(e)}

            texts = data_out.get("text", [])
            confs = data_out.get("conf", [])
            words = [t for t in texts if t.strip()]
            valid_confs = [float(c) for c in confs if c.strip() and c != "-1"]
            avg_conf = float(sum(valid_confs) / len(valid_confs)) / 100.0 if valid_confs else 0.0
            full_text = " ".join(words)
            return {"text": full_text, "confidence": avg_conf, "language": lang}
        except Exception:
            return {"text": "", "confidence": 0.0, "language": lang}

