import os
import sys
import json

# PRE-IMPORT TORCH AND APPLY PADDLE CPU STATIC RUNNER PATCH FOR WINDOWS
import torch
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

from paddleocr import PaddleOCR

def main():
    ocr = PaddleOCR(use_doc_unwarping=False, use_doc_orientation_classify=False, lang="vi")
    img_path = "outputs/restoration/test/Anh1_medium/restored.png"
    results = list(ocr.predict(img_path))
    res = results[0]
    print("KEYS IN res.json:", res.json.keys())
    print("res.json['res'] KEYS:", res.json["res"].keys())
    print("rec_text:", res.json["res"].get("rec_text"))
    print("rec_score:", res.json["res"].get("rec_score"))

if __name__ == "__main__":
    main()
