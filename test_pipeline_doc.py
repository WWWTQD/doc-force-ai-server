import os
import sys
import cv2
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
except Exception:
    pass

sys.path.append(r"c:\Users\imdrx\IdeaProjects\doc-force-ai-server\worker")
from scan_pipeline import ScanPipeline
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor

img_path = r"c:\Users\imdrx\IdeaProjects\doc-force-ai-server\sample_doc.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Sample Doc size: {w}x{h}")

prep = ImagePreprocessor(enable_magic=True)
engine = OCREngine(lang="vi")
pipeline = ScanPipeline(ocr_engine=engine, preprocessor=prep, enable_document_scan=True)

res = pipeline.process_image(img, page_number=1)
print(f"Geometry: method={res.geometry.method}, conf={res.geometry.detection_confidence}, rotation={res.geometry.rotation}")
print(f"Restored image size: {res.restored_image.shape[1]}x{res.restored_image.shape[0]}")
blocks = res.layout.get("blocks", [])
print(f"Total blocks detected: {len(blocks)}")
for i, b in enumerate(blocks[:15]):
    print(f"[{i+1}] ({b['type']}) conf={b['confidence']:.2f}: {b['text']}")

cv2.imwrite(r"c:\Users\imdrx\IdeaProjects\doc-force-ai-server\test_restored.jpg", res.restored_image)
with open(r"c:\Users\imdrx\IdeaProjects\doc-force-ai-server\test_layout.json", "w", encoding="utf-8") as f:
    json.dump(res.layout, f, ensure_ascii=False, indent=2)
