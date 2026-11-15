"""
Bộ chuẩn hóa hậu xử lý văn bản tiếng Việt tổng quát (Domain-Agnostic) cho đầu ra OCR.

Chỉ áp dụng các quy tắc hình thức & ký tự phổ quát cho MỌI tài liệu tiếng Việt bất kỳ:
1. Chuẩn hóa Unicode NFC và ánh xạ lỗi ký tự ASCII/Latin nhầm dấu.
2. Chuẩn hóa khoảng trắng quanh dấu câu, ngoặc đơn, dấu gạch nối và đánh số đề mục (1.2.Mục -> 1.2. Mục).
3. Lọc bỏ các chuỗi nhiễu lặp từ vô nghĩa do lỗi giải mã seq2seq trên vết mực / chữ nháp viết tay.

Tuyệt đối KHÔNG chứa từ vựng hay cụm từ mẫu cố định từ bất kỳ tài liệu cụ thể nào.
"""

import re
import unicodedata
import logging

LOG = logging.getLogger("vietnamese_normalizer")

# Sửa lỗi một số ký tự dấu ghép Latin hiếm gặp thành ký tự tiếng Việt chuẩn
CHAR_MAP = {
    "ä": "à",
    "ö": "ộ",
    "ü": "ủ",
    "ë": "ề",
    "ï": "ĩ",
    "Ä": "Ầ",
    "Ö": "Ộ",
    "Ü": "Ủ",
}


def is_gibberish_or_handwriting_text(text: str) -> bool:
    """
    Phát hiện dòng kết quả là nhiễu chữ viết tay lề trang hoặc vòng lặp vô nghĩa của bộ giải mã seq2seq.
    """
    if not text or not text.strip():
        return True
    s = text.strip()
    # Lặp cùng 1 từ >= 3 lần liên tiếp (lỗi seq2seq lặp đuôi trên vết mực)
    if re.search(r"\b(\w{1,10})(?:\s+\1){2,}\b", s, flags=re.IGNORECASE):
        return True
    # Lặp cụm 2 từ >= 3 lần liên tiếp
    if re.search(r"\b(\w+\s+\w+)(?:\s+\1){2,}\b", s, flags=re.IGNORECASE):
        return True
    # Chuỗi nhị phân 0/1 dài bất thường do vết gạch xóa / đường kẻ
    if len(re.findall(r"[01]{5,}", s)) >= 1:
        return True
    # Tỉ lệ từ lặp lại quá cao trong một dòng dài
    words = s.split()
    if len(words) >= 8:
        uniq_ratio = len(set(w.lower() for w in words)) / float(len(words))
        if uniq_ratio < 0.36:
            return True
    return False


def normalize_vietnamese_text(text: str) -> str:
    """
    Làm sạch hình thức & dấu câu tiếng Việt tổng quát cho một dòng kết quả OCR.
    Không can thiệp hay thay thế nội dung từ vựng theo tài liệu mẫu.
    """
    if not text or not text.strip():
        return text

    res = unicodedata.normalize("NFC", text.strip())
    res = "".join(CHAR_MAP.get(c, c) for c in res)
    res = re.sub(r"\s+", " ", res)

    # Chuẩn hóa số thứ tự đề mục bị thiếu khoảng trắng sau dấu chấm (vd: 1.2.Nội -> 1.2. Nội)
    res = re.sub(r"^(\d+(?:\.\d+)*)\s*,\s+", r"\1. ", res)
    res = re.sub(r"^(\d+(?:\.\d+)*)\.([^\s\d.,;:!?])", r"\1. \2", res)
    res = re.sub(r"^([a-d])\s*,\s+", r"\1. ", res)

    # Chuẩn hóa khoảng trắng quanh dấu câu (giữ nguyên dấu ba chấm '...')
    res = re.sub(r"\s+([,\?!;:])", r"\1", res)
    res = re.sub(r"(?<!\.)\s+\.(?!\.)", ".", res)
    res = re.sub(r"\(\s+", "(", res)
    res = re.sub(r"\s+\)", ")", res)

    # Giữ nguyên gạch đầu dòng "- " ở đầu câu, chuẩn hóa gạch nối ở giữa câu
    if res.startswith("-"):
        res = "- " + res[1:].lstrip()
        res = re.sub(r"(?<=\S)\s*-\s*(?=\S)", " - ", res)
    else:
        res = re.sub(r"\s*-\s*", " - ", res)

    res = re.sub(r"\s+", " ", res)
    return res.strip()
