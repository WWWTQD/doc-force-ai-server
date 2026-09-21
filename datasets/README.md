# PBL4 Dataset Directory

This directory is designated for storing dataset samples, bounding box annotations, and dataset manifest files for fine-tuning PaddleOCR / PP-StructureV3 on PBL4 specific document templates.

## Recommended Dataset Format

For PP-StructureV3 / PP-DocLayout fine-tuning:
```text
datasets/
├── images/
│   ├── doc_001.jpg
│   └── doc_002.jpg
└── annotations/
    ├── train.json  # COCO or PaddleOCR PP-Structure JSON format
    └── val.json
```

## Annotation Schema

Annotations should include bounding boxes and categories:
- `title`
- `text`
- `paragraph`
- `table`
- `header`
- `footer`
- `seal`
- `formula`
