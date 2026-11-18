"""Read-only measurement of Vietnamese syllable validator false positive rate on datasets/raw_docs/*.docx.

NOTE (Decision B5): Reported numbers are in-sample only and are NOT generalization accuracy.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worker.vietnamese_syllable_validator import invalid_syllables


def extract_docx_lines(docx_path: Path) -> list[str]:
    """Extract paragraph text lines from a .docx archive using stdlib zipfile + ElementTree."""
    lines: list[str] = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        if "word/document.xml" not in zf.namelist():
            return lines
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns_w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for p in root.iter(f"{ns_w}p"):
        texts = [t.text for t in p.iter(f"{ns_w}t") if t.text]
        joined = "".join(texts).strip()
        if joined:
            lines.append(joined)
    return lines


def count_word_tokens(lines: list[str]) -> int:
    total = 0
    for line in lines:
        total += len(re.findall(r"\S+", line))
    return total


def main() -> int:
    raw_docs_dir = ROOT / "datasets" / "raw_docs"
    docx_files = sorted(raw_docs_dir.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found under {raw_docs_dir}")
        return 1

    print("=== IN-SAMPLE MEASUREMENT ONLY (datasets/raw_docs/*.docx, khong phai do chinh xac) ===")
    print(f"{'File':<62} {'Tokens':>8} {'Flag(noLW)':>10} {'FP%(noLW)':>10} {'Flag(LW)':>9} {'FP%(LW)':>9}")

    tot_tokens = 0
    tot_flag_no_lw = 0
    tot_flag_lw = 0
    sample_fp: Counter[str] = Counter()

    for path in docx_files:
        lines = extract_docx_lines(path)
        n_tok = count_word_tokens(lines)
        f_no = 0
        f_lw = 0
        for ln in lines:
            f_no += len(invalid_syllables(ln, allow_loanwords=False))
            spans_lw = invalid_syllables(ln, allow_loanwords=True)
            f_lw += len(spans_lw)
            for sp in spans_lw:
                sample_fp[sp.token.lower()] += 1
        tot_tokens += n_tok
        tot_flag_no_lw += f_no
        tot_flag_lw += f_lw
        r_no = (f_no / n_tok * 100.0) if n_tok else 0.0
        r_lw = (f_lw / n_tok * 100.0) if n_tok else 0.0
        print(f"{path.name:<62} {n_tok:>8} {f_no:>10} {r_no:>9.2f}% {f_lw:>9} {r_lw:>8.2f}%")

    tot_r_no = (tot_flag_no_lw / tot_tokens * 100.0) if tot_tokens else 0.0
    tot_r_lw = (tot_flag_lw / tot_tokens * 100.0) if tot_tokens else 0.0
    print("-" * 112)
    print(
        f"{'TOTAL (in-sample, khong phai do chinh xac)':<62} "
        f"{tot_tokens:>8} {tot_flag_no_lw:>10} {tot_r_no:>9.2f}% {tot_flag_lw:>9} {tot_r_lw:>8.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
