import os
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
    for i, res in enumerate(results):
        print(f"Result {i} type: {type(res)}")
        print(f"Attributes: {dir(res)}")
        if hasattr(res, "json"):
            print("JSON dict keys:", res.json.keys())
            if "res" in res.json:
                print("res.json['res'] keys:", res.json["res"].keys())
                for k, v in res.json["res"].items():
                    if isinstance(v, list):
                        print(f"  key '{k}': list of length {len(v)}")
                        if len(v) > 0:
                            print(f"    first element type: {type(v[0])}, preview: {repr(v[0])[:100]}")
                    else:
                        print(f"  key '{k}': type {type(v)}")

if __name__ == "__main__":
    main()
