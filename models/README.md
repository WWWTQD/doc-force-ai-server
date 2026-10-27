# PBL4 Models Directory

This directory is designated for storing fine-tuned model checkpoints and model configuration files for the PBL4 Document AI system.

## Pretrained vs Fine-tuned Pipeline

```text
Pretrained PP-StructureV3 (Default)
        ↓
Collect PBL4 Domain Dataset (Invoices, Reports, Form Documents)
        ↓
Fine-tune PP-DocLayout / OCR / Table Recognition Models
        ↓
Replace/Configure Model Checkpoints here
```

## Setup Instructions for Fine-Tuned Weights

1. Place your fine-tuned model directories here:
   ```text
   models/
   ├── pbl4_doc_layout/
   │   ├── inference.pdmodel
   │   └── inference.pdiparams
   └── pbl4_table_structure/
       ├── inference.pdmodel
       └── inference.pdiparams
   ```
2. Update `.env` configuration to point to custom model paths:
   ```env
   AI_MODEL_NAME=PBL4-Structure-V1
   # AI_LAYOUT_MODEL_DIR=models/pbl4_doc_layout
   ```

> [!WARNING]
> Do NOT commit large `.pdparams`, `.pdmodel`, or `.bin` weight files directly into Git! Use Git LFS or external cloud storage.
