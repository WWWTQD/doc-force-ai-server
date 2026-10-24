import os
import sys
import json

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample_file = "outputs/ppstructure_anh1_results/sample.json"
    if not os.path.exists(sample_file):
        print("sample.json not found")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    res_inner = data.get("res", {})
    rec_texts = res_inner.get("rec_texts", [])
    rec_scores = res_inner.get("rec_scores", [])

    lines = []
    for idx, (text, score) in enumerate(zip(rec_texts, rec_scores), 1):
        lines.append({
            "line_no": idx,
            "text": text,
            "confidence": round(float(score), 4)
        })

    out_file = "outputs/ppstructure_anh1_results/anh1_medium_parsed_lines.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "image": "outputs/restoration/test/Anh1_medium/restored.png",
            "total_lines": len(lines),
            "lines": lines
        }, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"EXTRACTED {len(lines)} TEXT LINES FROM ANH1_MEDIUM RESTORED IMAGE:")
    print("=" * 70)
    for l in lines:
        print(f"Line {l['line_no']:2d} [{l['confidence']:.2f}]: {l['text']}")

if __name__ == "__main__":
    main()
