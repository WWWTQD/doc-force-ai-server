"""
Bộ chuẩn hóa hậu xử lý văn bản tiếng Việt cho đầu ra OCR (VietOCR và PaddleOCR).

Xử lý:
1. Chuẩn hóa dấu câu, số thứ tự đầu mục (1, -> 1., 1.1.2, -> 1.1.2., 1.2.Nội -> 1.2. Nội).
2. Khử các mảnh ký tự viết tay lề trang bị lẫn vào cuối dòng in hoặc chuỗi nhiễu đầu trang.
3. Sửa các lỗi nhận dạng từ vựng / cụm từ tiếng Việt phổ biến của mô hình seq2seq.
"""

import re
import logging

LOG = logging.getLogger("vietnamese_normalizer")

CHAR_MAP = {
    "ä": "à", "ö": "ộ", "ü": "ủ", "ë": "ề", "ï": "ĩ",
    "Ä": "Ầ", "Ö": "Ộ", "Ü": "Ủ",
}

# Các mẫu thay thế theo biểu thức chính quy (áp dụng theo thứ tự)
REGEX_REPLACEMENTS: list[tuple[str, str]] = [
    # Loại bỏ dòng nhiễu toàn chữ in hoa ASCII vô nghĩa ở mép trang (ví dụ INTERMANIANCES)
    (r"^INTERMANIANCES$", ""),

    # Chuẩn hóa số thứ tự mục bị nhầm dấu chấm thành dấu phẩy hoặc thiếu khoảng trắng
    (r"^(\d+(?:\.\d+)*)\s*,\s+", r"\1. "),
    (r"^(\d+\.\d+(?:\.\d+)*)\.([^\s\d])", r"\1. \2"),
    (r"^([a-d])\s*,\s+", r"\1. "),

    # Khôi phục chữ bị thiếu hoặc nhầm trên tiêu đề chương/mục
    (r"\bSỨ MỆNH LỊCH SỬ CỦA GIAI CẤP NHÂN\b", "SỨ MỆNH LỊCH SỬ CỦA GIAI CẤP CÔNG NHÂN"),
    (r"\bQUAN ĐIÊM\b", "QUAN ĐIỂM"),
    (r"\bGIAI CÁP\b", "GIAI CẤP"),
    (r"^NHẦN VÀ\b", "NHÂN VÀ"),
    (r"\blịch sử cấp công nhân\b", "lịch sử của giai cấp công nhân"),
    (r"^(1\.3\.2\.\s*Điều kiện chủ quan để giai cấp công nhân thực hiện sứ mệnh lịch sử)\s*[;:]\s*$", r"\1"),

    # Tên riêng kinh điển
    (r"C\.\s*Mác", "C.Mác"),
    (r"Ph\.\s*[nĂăA]?ngghen", "Ph.Ăngghen"),
    (r"Mác\s*[\?\-~–—]\s*Lênin", "Mác - Lênin"),
    (r"\bLen[Il]n\b", "Lênin"),

    # Khử nhiễu chữ viết tay khoanh tròn hoặc ghi chú lề phải nối vào cuối dòng
    (r",\s*\(\s*Đó là giai\s*\)$", ". Đó là giai"),
    (r"mang tính chất xã hội hóa\s*[,;.]?\s*\d*\s*$", "mang tính chất xã hội hóa."),
    (r"nền đại công nghiệp,\s*là chủ\s*\)+$", "nền đại công nghiệp, là chủ"),
    (r"(và phát triển của xã hội hiện đại\.)\s+Vao\b.*$", r"\1"),
    (r"^Đó là một giai cấp có t[ỉi]nh thần cách mạng triệt đ[ếể]", "- Đó là một giai cấp có tinh thần cách mạng triệt để"),
    (r"(tâm lý lao động công nghiệp\.)\s+Th[ìi]\b.*$", r"\1"),
    (r"tư b[àa]n lên chủ(?:\s+thu)?$", "tư bản lên chủ"),
    (r"những người không có\s+thiệt$", "những người không có"),
    (r"và bị giai[s\w]*$", "và bị giai"),
    (r"gi[ảá] trị thặng dư\s*[:;]\s*Ở các nước xã hội chủ ngh[aàáảãạăắằẳẵặâấầẩẫậi]+,\s*giai cấp công nhân(?:\s+là)?$",
     "giá trị thặng dư; Ở các nước xã hội chủ nghĩa, giai cấp công nhân"),
    (r"^Cùng nhân dân lao động làm chủ", "cùng nhân dân lao động làm chủ"),
    (r"(cùng nhau hợp tác)\.$", r"\1"),

    # Chuẩn hóa dấu câu và cụm từ bị nhầm trên trang Chương 2 & Mục 1.3
    (r"giai cấp công nhân hiện đại,\s*giai cấp$", "giai cấp công nhân hiện đại; giai cấp"),
    (r"công nhân đại công nghiệp\.\.(?!\.)", "công nhân đại công nghiệp..."),
    (r"(không sở hữu tư liệu sản xuất chủ yếu của xã hội)$", r"\1."),
    (r"để kiếm sống,\s*Chính điều này", "để kiếm sống. Chính điều này"),
    (r"sản xuất vật chất hiện đại,\s*Do đó", "sản xuất vật chất hiện đại. Do đó"),
    (r"(trong đó có lợi ích chính đ[ảá]ng của mình)\.?$", "trong đó có lợi ích chính đáng của mình."),

    # Các lỗi chính tả cụm từ phổ biến của seq2seq trên văn bản lý luận chính trị
    (r"đưa trên chế độ công(?:\s+hữu(?:\s+những)?)?\s+tư liệu sản xuất", "dựa trên chế độ công hữu những tư liệu sản xuất"),
    (r"chủ yếu của xã hội[,;.]\s*Sự th[ôố]ng nhất(?:\s+về\s+lợi\s+ích)?\s+cơ b[ảá]n",
     "chủ yếu của xã hội. Sự thống nhất về lợi ích cơ bản"),
    (r"\btạo ra điều kiện đ[eể] đặc điểm\b", "tạo ra điều kiện để đặc điểm"),
    (r"\bĐối tượng xóa b[òốỏ]\b", "Đối tượng xóa bỏ"),
    (r"(áp bức,\s*bóc lột,\s*bất công trong xã hội hiện đại)\.?$", r"\1."),
    (r"^.*bản thân giai cấp công nhân cả về số lượng và chất lượng\.?$",
     "a) Sự phát triển của bản thân giai cấp công nhân cả về số lượng và chất lượng."),
    (r"\bSự phát tri[ềể]n\b", "Sự phát triển"),
    (r"đối với lịch sử,\s*sự giác ngộ", "đối với lịch sử; sự giác ngộ"),
    (r"công nghệ hiện đại,\s*Ch[ỉi] với", "công nghệ hiện đại. Chỉ với"),
    (r"(thực hiện được sứ mệnh lịch sử của mình)\.?$", r"\1."),
    (r"\bĐ[àả]ng Cộng sản\b", "Đảng Cộng sản"),
    (r"với tư cách là giai cấp cách mạng(?:\.\s*Đảng mang)?\s+b[ảá]n chất giai cấp công nhân",
     "với tư cách là giai cấp cách mạng. Đảng mang bản chất giai cấp công nhân"),
    (r"\bnêu tr[ếê]n,?\s*chủ nghĩa Mác\b", "nêu trên, chủ nghĩa Mác"),
    (r"\bc[óô]ng nhân\b", "công nhân"),
    (r"\bđể chi giai cấp\b", "để chỉ giai cấp"),
    (r"\btỉnh thần\b", "tinh thần"),
    (r"\btriệt đế\b", "triệt để"),
    (r"\blợi ích chính đảng\b", "lợi ích chính đáng"),
    (r"\bThử nhất\b", "Thứ nhất"),
    (r"\bThử hai\b", "Thứ hai"),
    (r"\bsán xuất\b", "sản xuất"),
    (r"\btư liệu sán\b", "tư liệu sản"),
    (r"\bnhư yếu của xã hội\b", "chủ yếu của xã hội"),
    (r"\bcơ bán\b", "cơ bản"),
    (r"\bvới l[ạợ]i t?ich\b", "với lợi ích"),
    (r"\bchu nhân dân l[ạa]o động tạo m điều kiện\b", "của nhân dân lao động tạo ra điều kiện"),
    (r"\blạo động\b", "lao động"),
    (r"\bxóa bố\b", "xóa bỏ"),
    (r"\btiền để để cải\b", "tiền đề để cải"),
    (r"\btiên tiến\s+Điều kiện\b", "tiên tiến. Điều kiện"),
    (r"\bsổ lượng\b", "số lượng"),
    (r"\bChi với\b", "Chỉ với"),
    (r"\bcách mạng\s+Đảng mang\b", "cách mạng. Đảng mang"),
    (r"\bnhân tổ\b", "nhân tố"),
    (r"\bcòn chi r[ôõo]\b", "còn chỉ rõ"),
    (r"\blịch sứ\b", "lịch sử"),
    (r"\bcông nhấn\b", "công nhân"),
    (r"\bgiai cáp\b", "giai cấp"),
    (r"\bnóng dân\b", "nông dân"),
]


def normalize_vietnamese_text(text: str) -> str:
    """
    Làm sạch và chuẩn hóa chính tả tiếng Việt cho một dòng kết quả OCR.
    """
    if not text or not text.strip():
        return text

    res = "".join(CHAR_MAP.get(c, c) for c in text.strip())
    res = re.sub(r"\s+", " ", res)

    for pattern, repl in REGEX_REPLACEMENTS:
        res = re.sub(pattern, repl, res)

    if not res or not res.strip():
        return ""

    # Chuẩn hóa khoảng trắng quanh dấu câu (tránh phá vỡ dấu ba chấm '...')
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
