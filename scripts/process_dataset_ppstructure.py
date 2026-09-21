import os
import sys
import glob
import json
import time
import logging
import pathlib
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ppstructure_dataset_processor")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def process_dataset(
    dataset_dir: str = "datasets/images",
    output_dir: str = "outputs/pp_structure",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Recursively scans dataset_dir for images, runs PP-StructureV3 inference,
    saves structured raw JSON, visualization, Markdown, and Word output.
    """
    start_time = time.time()
    
    # 1. Initialize PP-StructureV3 Engine
    from app.services.structure_engine import StructureEngine
    engine = StructureEngine.get_instance()
    if not engine.initialized:
        logger.info("Initializing PP-StructureV3 model for dataset batch inference...")
        engine.initialize(device=device)

    # 2. Recursively find all supported images
    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                all_files.append(os.path.join(root, f))

    logger.info(f"Discovered total {len(all_files)} document images in '{dataset_dir}'")

    stats = {
        "total_images": len(all_files),
        "success": 0,
        "failed": 0,
        "categories": {},
        "file_details": []
    }

    base_path = pathlib.Path(dataset_dir)

    for idx, img_path in enumerate(all_files, start=1):
        rel_path = pathlib.Path(img_path).relative_to(base_path)
        parts = rel_path.parts
        
        # Category is the immediate parent directory or 'root' if directly under dataset_dir
        category = parts[0] if len(parts) > 1 else "root"
        filename = os.path.basename(img_path)

        if category not in stats["categories"]:
            stats["categories"][category] = {"total": 0, "success": 0, "failed": 0}
        stats["categories"][category]["total"] += 1

        logger.info(f"[{idx}/{len(all_files)}] Processing [{category}] -> {filename}")

        # Prepare category output directory
        cat_out_dir = os.path.join(output_dir, category)
        os.makedirs(cat_out_dir, exist_ok=True)

        img_stem = os.path.splitext(filename)[0]
        json_out_path = os.path.join(cat_out_dir, f"{img_stem}.json")

        try:
            # Predict raw result objects
            raw_objs = engine.predict_raw(img_path)
            
            # Convert raw results to serializable structure
            parsed_results = []
            for res in raw_objs:
                if hasattr(res, "json") and res.json:
                    val = res.json if isinstance(res.json, dict) else res.json
                    parsed_results.append(val)
                elif hasattr(res, "to_dict"):
                    parsed_results.append(res.to_dict())
                elif isinstance(res, dict):
                    parsed_results.append(res)
                else:
                    parsed_results.append(dict(res))

            # Build standardized document prediction record
            doc_record = {
                "filename": filename,
                "category": category,
                "source_path": img_path.replace("\\", "/"),
                "label_type": "prediction",
                "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "results": parsed_results
            }

            # Save raw JSON
            with open(json_out_path, "w", encoding="utf-8") as f:
                json.dump(doc_record, f, ensure_ascii=False, indent=2, default=str)

            # Save visualizations and document exports if supported by PaddleX result object
            vis_save_dir = os.path.join(cat_out_dir, "visualizations")
            doc_export_dir = os.path.join(cat_out_dir, "exports")
            os.makedirs(vis_save_dir, exist_ok=True)
            os.makedirs(doc_export_dir, exist_ok=True)

            for item_idx, res_obj in enumerate(raw_objs):
                # 1. Visualization
                if hasattr(res_obj, "save_to_img"):
                    try:
                        res_obj.save_to_img(save_path=vis_save_dir)
                    except Exception as e_vis:
                        logger.debug(f"save_to_img note for {filename}: {e_vis}")

                # 2. Markdown export
                if hasattr(res_obj, "save_to_markdown"):
                    try:
                        res_obj.save_to_markdown(save_path=doc_export_dir)
                    except Exception as e_md:
                        logger.debug(f"save_to_markdown note for {filename}: {e_md}")

                # 3. Word DOCX export
                if hasattr(res_obj, "save_to_word"):
                    try:
                        res_obj.save_to_word(save_path=doc_export_dir)
                    except Exception as e_doc:
                        logger.debug(f"save_to_word note for {filename}: {e_doc}")

            stats["success"] += 1
            stats["categories"][category]["success"] += 1
            stats["file_details"].append({
                "filename": filename,
                "category": category,
                "status": "success",
                "output_json": json_out_path
            })

        except Exception as e:
            logger.error(f"Failed processing {img_path}: {e}", exc_info=True)
            stats["failed"] += 1
            stats["categories"][category]["failed"] += 1
            stats["file_details"].append({
                "filename": filename,
                "category": category,
                "status": "failed",
                "error": str(e)
            })

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)

    # Save summary manifest
    manifest_path = os.path.join(output_dir, "batch_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("==================================================")
    logger.info(f"Dataset PP-StructureV3 processing complete in {elapsed:.2f}s!")
    logger.info(f"Total: {stats['total_images']} | Success: {stats['success']} | Failed: {stats['failed']}")
    for cat, cat_stat in stats["categories"].items():
        logger.info(f"  - Category [{cat}]: total={cat_stat['total']}, success={cat_stat['success']}, failed={cat_stat['failed']}")
    logger.info(f"Summary manifest saved to: {manifest_path}")
    logger.info("==================================================")

    return stats


if __name__ == "__main__":
    process_dataset()
