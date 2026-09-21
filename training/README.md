# PBL4 Training & Evaluation Workflow

This directory contains fine-tuning scripts and evaluation benchmarks for replacing pretrained PP-StructureV3 with custom-trained PBL4 models.

## Phase Overview

```text
Phase 1: Deploy Pretrained PP-StructureV3 (CURRENT)
        ↓
Phase 2: Collect & Annotate PBL4 Document Dataset
        ↓
Phase 3: Benchmark Pretrained Performance on PBL4 Dataset
        ↓
Phase 4: Fine-tune PP-DocLayout / PaddleOCR / SLANet
        ↓
Phase 5: Evaluate Precision & Replace Model
```

## Running Fine-Tuning with PaddleOCR/PaddleX

Example PaddleX CLI command for layout detection fine-tuning:
```bash
paddlex --train --config training/layout_det_config.yaml
```
