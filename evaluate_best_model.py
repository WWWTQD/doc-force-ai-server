import os
import json
import csv
import cv2
from PIL import Image
from ultralytics import YOLO

def main():
    output_dir = "eval_results"
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    model_path = "worker/Model/best.pt"
    input_dir = "data/input"

    print("=== STEP 1: Loading model ===")
    try:
        model = YOLO(model_path)
        load_success = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        load_success = False
        return

    print("\n=== STEP 2: Model Architecture & Details ===")
    task = getattr(model, "task", "unknown")
    names = getattr(model, "names", {})
    num_classes = len(names)
    
    # Layer and parameter count
    info = model.info()
    layers = info[0] if info else "N/A"
    params = info[1] if info else "N/A"
    gflops = info[3] if info and len(info) > 3 else "N/A"

    print(f"Task: {task}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {names}")
    print(f"Layers: {layers}, Parameters: {params}, GFLOPs: {gflops}")

    print("\n=== STEP 3: Single Image Inference Test ===")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print(f"No test images found in {input_dir}")
        return

    test_single_img = os.path.join(input_dir, image_files[0])
    print(f"Running inference on single test image: {test_single_img}")
    single_results = model.predict(test_single_img, save=False, verbose=False)
    print(f"Single image inference output boxes count: {len(single_results[0].boxes)}")

    print("\n=== STEP 4 & 5 & 6: Batch Inference on all test images ===")
    summary_data = []

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(img_path, conf=0.25, save=False, verbose=False)[0]
        
        # Save plotted prediction image
        annotated_img = results.plot()
        out_img_path = os.path.join(pred_dir, f"pred_{img_name}")
        cv2.imwrite(out_img_path, annotated_img)

        boxes_info = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())
            xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
            boxes_info.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": xyxy
            })

        summary_data.append({
            "image_name": img_name,
            "width": w,
            "height": h,
            "num_detections": len(boxes_info),
            "detections": boxes_info
        })

    # Save JSON summary
    metadata = {
        "model_path": model_path,
        "load_success": load_success,
        "task": task,
        "num_classes": num_classes,
        "class_names": names,
        "layers": layers,
        "params": params,
        "total_test_images": len(image_files),
        "results": summary_data
    }

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save CSV summary
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Resolution", "Detections Count", "Detected Classes", "Confidences", "BBoxes (xyxy)"])
        for item in summary_data:
            classes_str = ", ".join([d["class_name"] for d in item["detections"]]) if item["detections"] else "None"
            confs_str = ", ".join([str(d["confidence"]) for d in item["detections"]]) if item["detections"] else "N/A"
            boxes_str = "; ".join([str(d["bbox_xyxy"]) for d in item["detections"]]) if item["detections"] else "N/A"
            writer.writerow([
                item["image_name"],
                f"{item['width']}x{item['height']}",
                item["num_detections"],
                classes_str,
                confs_str,
                boxes_str
            ])

    print(f"\nSaved evaluation summary JSON to {os.path.join(output_dir, 'evaluation_summary.json')}")
    print(f"Saved evaluation summary CSV to {csv_path}")
    print(f"Saved {len(summary_data)} prediction images to {pred_dir}")

    print("\n=== STEP 7 & 8: Checking model.val() dataset ===")
    yaml_exists = os.path.exists("data.yaml")
    if yaml_exists:
        print("data.yaml found! Running model.val()...")
        try:
            val_metrics = model.val(data="data.yaml")
            print("Validation completed successfully.")
            print("mp:", val_metrics.box.mp)
            print("mr:", val_metrics.box.mr)
            print("map50:", val_metrics.box.map50)
            print("map:", val_metrics.box.map)
        except Exception as e:
            print(f"Validation error: {e}")
    else:
        print("NOTICE: 'data.yaml' file not found in current directory workspace.")
        print("Without ground truth annotations (labels & data.yaml), model.val() ground-truth metrics cannot be computed locally.")

if __name__ == "__main__":
    main()
