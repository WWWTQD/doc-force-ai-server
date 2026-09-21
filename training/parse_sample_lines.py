import os
import json

def main():
    sample_file = "outputs/ppstructure_anh1_results/sample.json"
    if not os.path.exists(sample_file):
        print("sample.json not found")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    res_inner = data.get("res", {})
    rec_texts = res_inner.get("rec_text", [])
    rec_scores = res_inner.get("rec_score", [])

    lines = []
    for idx, (text, score) in enumerate(zip(rec_texts, rec_scores), 1):
        lines.append({
            "line_no": idx,
            "text": text,
            "confidence": round(float(score), 4)
        })

    out_file = "outputs/ppstructure_anh1_results/anh1_medium_restored_ocr_lines.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "image": "outputs/restoration/test/Anh1_medium/restored.png",
            "total_lines": len(lines),
            "lines": lines
        }, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"PP-STRUCTUREV3 PARSED {len(lines)} TEXT LINES FROM ANH1 RESTORED:")
    print("=" * 60)
    for l in lines:
        print(f"  Line {l['line_no']:2d} [{l['confidence']:.2f}]: {l['text']}")

if __name__ == "__main__":
    main()
