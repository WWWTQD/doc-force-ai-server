"""
Tiện ích dùng chung cho tái tạo bố cục Word (.docx) & PDF (.pdf) giữ đúng cấu trúc
hàng ngang nhiều cột (Quốc hiệu 2 bên, nhiều trường biểu mẫu trên 1 dòng),
khối ngày tháng & chữ ký ở nửa phải trang, thụt đầu dòng và bảng biểu.
"""

from __future__ import annotations

import io
import os
import re
from typing import Any


def strip_markdown_tokens(text: str) -> str:
    """Loại bỏ các cặp ký hiệu markdown (**bold**, *italic*, <u>underline</u>, __underline__) để lấy chuỗi thuần."""
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"<u>(.*?)</u>", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    return s.strip()


def resolve_ttf_font(
    filename: str,
    fallback_names: list[str],
    extra_roots: list[str] | None = None,
) -> str | None:
    """Tìm đường dẫn file font TrueType hỗ trợ đầy đủ tiếng Việt Unicode."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(base_dir)
    roots = list(extra_roots or []) + [proj_dir, base_dir, "/app"]
    candidates = [os.path.join(r, "models", "fonts", filename) for r in roots]
    candidates.append(os.path.join(r"C:\Windows\Fonts", filename))
    for fn in fallback_names:
        candidates.append(os.path.join("/usr/share/fonts/truetype/dejavu", fn))
        candidates.append(os.path.join("/usr/share/fonts/truetype/liberation", fn))
        candidates.append(os.path.join(r"C:\Windows\Fonts", fn))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def compute_page_metrics(blocks: list[dict[str, Any]], page_width: int, page_height: int) -> dict[str, float]:
    """Tính toán vùng lề trái/phải chuẩn và chiều cao dòng trung vị trên trang từ các khối bbox hợp lệ."""
    pw = max(1, int(page_width or 800))
    ph = max(1, int(page_height or 1130))
    text_blocks = [
        b for b in blocks
        if str(b.get("type", "")).lower() != "table"
        and "tableData" not in b
        and isinstance(b.get("bbox"), (list, tuple))
        and len(b["bbox"]) == 4
        and (b.get("text") or "").strip()
    ]
    substantial = [b for b in text_blocks if (b["bbox"][2] - b["bbox"][0]) > pw * 0.32 and b["bbox"][0] > pw * 0.04]
    if not substantial:
        substantial = [b for b in text_blocks if (b["bbox"][2] - b["bbox"][0]) > pw * 0.25]
    if substantial:
        lefts = sorted(float(b["bbox"][0]) for b in substantial)
        rights = sorted(float(b["bbox"][2]) for b in substantial)
        dom_left = lefts[int(len(lefts) * 0.15)]
        dom_right = rights[int(len(rights) * 0.88)]
    else:
        dom_left = pw * 0.15
        dom_right = pw * 0.85

    col_w = max(80.0, dom_right - dom_left)
    col_center = 0.5 * (dom_left + dom_right)

    line_heights: list[float] = []
    for b in text_blocks:
        txt = (b.get("text") or "").strip()
        n_lines = max(1, len([ln for ln in txt.splitlines() if ln.strip()]))
        lh = (float(b["bbox"][3]) - float(b["bbox"][1])) / float(n_lines)
        if 8.0 <= lh <= ph * 0.08:
            line_heights.append(lh)
    line_heights.sort()
    median_lh = line_heights[len(line_heights) // 2] if line_heights else max(14.0, ph * 0.022)

    return {
        "page_width": float(pw),
        "page_height": float(ph),
        "dominant_left": float(dom_left),
        "dominant_right": float(dom_right),
        "col_width": float(col_w),
        "col_center": float(col_center),
        "median_lh": float(median_lh),
    }


def _same_horizontal_row(row_blocks: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    """Kiểm tra `candidate` có nằm trên cùng 1 hàng ngang với `row_blocks` (không đè X và khớp dải Y)."""
    if not row_blocks:
        return False
    c_bb = candidate.get("bbox")
    if not isinstance(c_bb, (list, tuple)) or len(c_bb) != 4:
        return False
    if "tableData" in candidate or str(candidate.get("type", "")).lower() == "table":
        return False
    last_bb = row_blocks[-1].get("bbox")
    if not isinstance(last_bb, (list, tuple)) or len(last_bb) != 4:
        return False
    if "tableData" in row_blocks[-1] or str(row_blocks[-1].get("type", "")).lower() == "table":
        return False

    r_y1 = sum(float(b["bbox"][1]) for b in row_blocks) / len(row_blocks)
    r_y2 = sum(float(b["bbox"][3]) for b in row_blocks) / len(row_blocks)
    r_h = max(6.0, r_y2 - r_y1)
    r_yc = 0.5 * (r_y1 + r_y2)

    c_x1, c_y1, c_x2, c_y2 = (float(v) for v in c_bb)
    c_h = max(6.0, c_y2 - c_y1)
    c_yc = 0.5 * (c_y1 + c_y2)

    y_ov = min(r_y2, c_y2) - max(r_y1, c_y1)
    min_h = min(r_h, c_h)
    same_y = (y_ov >= 0.45 * min_h) or (abs(c_yc - r_yc) < 0.55 * min_h)
    if not same_y:
        return False

    # Không chồng lấp ngang giữa các cột
    for rb in row_blocks:
        rx1, _, rx2, _ = (float(v) for v in rb["bbox"])
        x_ov = min(rx2, c_x2) - max(rx1, c_x1)
        min_w = max(1.0, min(rx2 - rx1, c_x2 - c_x1))
        if x_ov > min(15.0, 0.20 * min_w):
            return False
    return True


def group_blocks_into_rows(
    blocks: list[dict[str, Any]],
    page_width: int,
    page_height: int,
) -> tuple[dict[str, float], list[list[dict[str, Any]]]]:
    """Nhóm các khối văn bản có cùng hàng ngang (Y tương đương, X tách biệt) thành từng row đã sắp trái→phải."""
    valid = [b for b in blocks if (b.get("text") or "").strip() or b.get("tableData")]
    valid.sort(key=lambda b: (b.get("order", 0)))
    metrics = compute_page_metrics(valid, page_width, page_height)

    rows: list[list[dict[str, Any]]] = []
    for blk in valid:
        if rows and _same_horizontal_row(rows[-1], blk):
            rows[-1].append(blk)
        else:
            rows.append([blk])

    for row in rows:
        if len(row) > 1:
            row.sort(key=lambda b: float(b["bbox"][0]))
    return metrics, rows


def classify_row_columns(
    row: list[dict[str, Any]],
    metrics: dict[str, float],
) -> tuple[str, list[tuple[float, str]]]:
    """
    Với hàng nhiều cột (`len(row) >= 2`), trả về `(mode, [(width_frac, align), ...])`:
    - `mode == "center_pair"`: hàng Quốc hiệu / cặp tiêu đề 2 cột căn giữa từng cột.
    - `mode == "form_grid"`: hàng chứa nhiều trường thông tin (`Ông/bà | Sinh ngày | Giới tính` hoặc `- Lớp | thuộc khoa`).
    """
    n = len(row)
    dom_left = metrics["dominant_left"]
    dom_right = metrics["dominant_right"]
    col_w = metrics["col_width"]
    page_h = metrics["page_height"]

    aligns = [str((b.get("style") or {}).get("textAlign") or "left").lower() for b in row]
    y_avg = sum(float(b["bbox"][1]) for b in row) / n
    is_center_pair = (
        n == 2
        and (
            all(a == "center" for a in aligns)
            or (y_avg < page_h * 0.25 and any(a == "center" for a in aligns))
        )
    )
    if is_center_pair:
        # Cân đối tỷ lệ cột theo độ dài chữ (nhưng giữ quanh 42%-58% để vế "CỘNG HÒA..." không bị ngắt dòng xấu)
        l0 = max(8, len(strip_markdown_tokens(row[0].get("text") or "")))
        l1 = max(8, len(strip_markdown_tokens(row[1].get("text") or "")))
        if l1 > l0 * 1.25:
            frac0, frac1 = 0.43, 0.57
        elif l0 > l1 * 1.25:
            frac0, frac1 = 0.57, 0.43
        else:
            frac0, frac1 = 0.46, 0.54
        return "center_pair", [(frac0, "center"), (frac1, "center")]

    # Hàng biểu mẫu nhiều cột: ánh xạ theo vị trí x1 thực tế trên vùng cột [dom_left .. dom_right]
    xs = [float(b["bbox"][0]) for b in row]
    span_start = min(dom_left, xs[0])
    span_end = max(dom_right, float(row[-1]["bbox"][2]))
    span_w = max(80.0, span_end - span_start)

    fracs: list[float] = []
    for idx in range(n):
        cur_start = span_start if idx == 0 else xs[idx]
        cur_end = xs[idx + 1] if idx + 1 < n else span_end
        fracs.append(max(0.14, (cur_end - cur_start) / span_w))
    total_f = sum(fracs)
    norm_fracs = [f / total_f for f in fracs]
    col_specs = [(norm_fracs[i], aligns[i] if aligns[i] in ("left", "center", "right") else "left") for i in range(n)]
    return "form_grid", col_specs


def is_right_half_signature_block(block: dict[str, Any], metrics: dict[str, float]) -> bool:
    """Kiểm tra khối đơn nằm hoàn toàn ở nửa bên phải trang (ngày tháng năm / chức danh / chữ ký)."""
    bb = block.get("bbox")
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return False
    align = str((block.get("style") or {}).get("textAlign") or "").lower()
    if align != "right":
        return False
    bx1, _, bx2, _ = (float(v) for v in bb)
    b_center = 0.5 * (bx1 + bx2)
    return (
        bx1 > metrics["dominant_left"] + 0.35 * metrics["col_width"]
        and b_center > metrics["col_center"] + 0.08 * metrics["col_width"]
    )


def append_formatted_runs(p, text: str, style_meta: dict) -> None:
    """Phân tách chuỗi markdown (**bold**, *italic*, <u>underline</u>) thành các runs trong python-docx."""
    from docx.shared import Pt, RGBColor

    block_bold = bool(style_meta.get("bold", False))
    block_italic = bool(style_meta.get("italic", False))
    block_underline = bool(style_meta.get("underline", False))
    font_family = str(style_meta.get("fontFamily") or "Times New Roman")
    font_size_pt = float(style_meta.get("fontSizePt") or 13.0)
    font_size_pt = max(10.0, min(20.0, font_size_pt))

    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|<u>.*?</u>|__[^_]+__)", text)
    for token in tokens:
        if not token:
            continue
        is_b = block_bold
        is_i = block_italic
        is_u = block_underline
        clean_text = token

        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            is_b = True
            clean_text = token[2:-2]
        elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
            is_i = True
            clean_text = token[1:-1]
        elif token.startswith("<u>") and token.endswith("</u>") and len(token) >= 7:
            is_u = True
            clean_text = token[3:-4]
        elif token.startswith("__") and token.endswith("__") and len(token) >= 4:
            is_u = True
            clean_text = token[2:-2]

        run = p.add_run(clean_text)
        run.font.name = font_family
        run.font.size = Pt(font_size_pt)
        run.font.color.rgb = RGBColor(15, 16, 20)
        if is_b:
            run.bold = True
        if is_i:
            run.italic = True
        if is_u:
            run.underline = True


def _make_borderless_docx_table(doc, n_rows: int, n_cols: int, col_widths_in: list[float]):
    """Tạo bảng Word không viền cố định độ rộng cột dùng cho Quốc hiệu 2 cột & hàng nhiều trường."""
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls, qn
    from docx.shared import Inches

    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.autofit = False
    tbl_pr = tbl._tbl.tblPr
    for tag in ("w:tblBorders", "w:tblLayout", "w:tblCellMar"):
        for old_el in tbl_pr.findall(qn(tag)):
            tbl_pr.remove(old_el)

    tbl_pr.append(
        parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
            "</w:tblBorders>"
        )
    )
    tbl_pr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    tbl_pr.append(
        parse_xml(
            f'<w:tblCellMar {nsdecls("w")}>'
            '<w:top w:w="15" w:type="dxa"/>'
            '<w:left w:w="30" w:type="dxa"/>'
            '<w:bottom w:w="15" w:type="dxa"/>'
            '<w:right w:w="30" w:type="dxa"/>'
            "</w:tblCellMar>"
        )
    )

    for c_i, w_in in enumerate(col_widths_in):
        w_emu = Inches(w_in)
        tbl.columns[c_i].width = w_emu
        for r_i in range(n_rows):
            tbl.rows[r_i].cells[c_i].width = w_emu
    return tbl


def _apply_align_to_docx_para(p, align_str: str) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    a = (align_str or "left").lower()
    if a == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif a == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif a == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def _row_vertical_bounds(row: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Lấy toạ độ (top, bottom) của một hàng từ các bbox hợp lệ."""
    bbs = [b["bbox"] for b in row if isinstance(b.get("bbox"), (list, tuple)) and len(b["bbox"]) == 4]
    if not bbs:
        return None, None
    return min(float(bb[1]) for bb in bbs), max(float(bb[3]) for bb in bbs)


def generate_docx_from_snapshot(snapshot: dict) -> bytes:
    """Tái tạo file Word (.docx) từ snapshot JSON, giữ chuẩn hàng ngang nhiều cột & chữ ký nửa phải."""
    import docx
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = docx.Document()
    left_in, right_in = 1.10, 0.79
    content_w_in = 8.27 - left_in - right_in  # ~6.38 inches

    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.79)
        section.bottom_margin = Inches(0.79)
        section.left_margin = Inches(left_in)
        section.right_margin = Inches(right_in)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(13)
    style.font.color.rgb = RGBColor(15, 16, 20)

    pages = snapshot.get("pages", [])
    for p_idx, page in enumerate(pages):
        if p_idx > 0:
            doc.add_page_break()

        layout = page.get("layout", {}) or {}
        raw_blocks = list(layout.get("blocks", []) or [])
        metrics, rows = group_blocks_into_rows(
            raw_blocks,
            int(layout.get("width") or 1654),
            int(layout.get("height") or 2339),
        )

        # Gộp các hàng Quốc hiệu 2 cột liên tiếp thành 1 bảng 2 cột không viền để hai hàng thẳng tâm nhau
        row_idx = 0
        prev_row_bottom = None
        while row_idx < len(rows):
            row = rows[row_idx]
            row_top, row_bottom = _row_vertical_bounds(row)

            if len(row) >= 2:
                mode, col_specs = classify_row_columns(row, metrics)
                if mode == "center_pair":
                    pair_rows = [row]
                    j = row_idx + 1
                    while j < len(rows) and len(rows[j]) == 2:
                        m_j, _ = classify_row_columns(rows[j], metrics)
                        if m_j != "center_pair":
                            break
                        pair_rows.append(rows[j])
                        j += 1
                    col_widths_in = [
                        content_w_in * max(classify_row_columns(pr, metrics)[1][c][0] for pr in pair_rows)
                        for c in range(2)
                    ]
                    tot_w = sum(col_widths_in)
                    col_widths_in = [w * content_w_in / tot_w for w in col_widths_in]
                    tbl = _make_borderless_docx_table(doc, len(pair_rows), 2, col_widths_in)
                    for r_i, pr in enumerate(pair_rows):
                        for c_i, cell_blk in enumerate(pr):
                            cell_p = tbl.rows[r_i].cells[c_i].paragraphs[0]
                            cell_p.paragraph_format.space_before = Pt(0)
                            cell_p.paragraph_format.space_after = Pt(1)
                            cell_p.paragraph_format.line_spacing = 1.15
                            cell_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            c_style = dict(cell_blk.get("style") or {})
                            c_style["fontSizePt"] = min(13.0, float(c_style.get("fontSizePt", 12.5)))
                            append_formatted_runs(cell_p, (cell_blk.get("text") or "").strip(), c_style)
                    row_idx = j
                    _, prev_row_bottom = _row_vertical_bounds(pair_rows[-1])
                    continue

                # Hàng chứa nhiều trường thông tin (`Ông/bà... | Sinh ngày... | Giới tính...` hoặc `- Lớp:... | thuộc khoa:...`)
                col_widths_in = [content_w_in * sp[0] for sp in col_specs]
                tbl = _make_borderless_docx_table(doc, 1, len(row), col_widths_in)
                for c_i, cell_blk in enumerate(row):
                    cell_p = tbl.rows[0].cells[c_i].paragraphs[0]
                    cell_p.paragraph_format.space_before = Pt(1)
                    cell_p.paragraph_format.space_after = Pt(2)
                    cell_p.paragraph_format.line_spacing = 1.15
                    _apply_align_to_docx_para(cell_p, col_specs[c_i][1])
                    append_formatted_runs(cell_p, (cell_blk.get("text") or "").strip(), dict(cell_blk.get("style") or {}))
                row_idx += 1
                prev_row_bottom = row_bottom
                continue

            # Hàng đơn 1 block
            block = row[0]
            b_type = str(block.get("type", "paragraph")).lower()
            text = (block.get("text") or "").strip()
            style_meta = dict(block.get("style") or {})

            if "table" in b_type:
                table_data = block.get("tableData")
                if isinstance(table_data, dict) and table_data.get("headers"):
                    headers = table_data.get("headers", [])
                    t_rows = table_data.get("rows", [])
                    tbl = doc.add_table(rows=1 + len(t_rows), cols=len(headers))
                    try:
                        tbl.style = "Table Grid"
                    except Exception:
                        pass
                    tbl_font_name = style_meta.get("fontFamily", "Times New Roman")
                    tbl_font_pt = float(style_meta.get("fontSizePt", 12.0))
                    for c_i, h_text in enumerate(headers):
                        cell_p = tbl.rows[0].cells[c_i].paragraphs[0]
                        cell_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = cell_p.add_run(strip_markdown_tokens(str(h_text)))
                        run.bold = True
                        run.font.name = tbl_font_name
                        run.font.size = Pt(tbl_font_pt)
                    for r_i, row_cells in enumerate(t_rows):
                        for c_i, cell_val in enumerate(row_cells[:len(headers)]):
                            cell_p = tbl.rows[r_i + 1].cells[c_i].paragraphs[0]
                            append_formatted_runs(
                                cell_p,
                                str(cell_val),
                                {"fontFamily": tbl_font_name, "fontSizePt": tbl_font_pt, "bold": False},
                            )
                else:
                    p = doc.add_paragraph()
                    run = p.add_run(f"[BẢNG]: {text}")
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(12)
                    run.bold = True
                    p.paragraph_format.space_after = Pt(6)
            elif "heading" in b_type or b_type == "title":
                p = doc.add_paragraph()
                _apply_align_to_docx_para(p, str(style_meta.get("textAlign", "center")))
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.line_spacing = 1.2
                style_meta["bold"] = True
                style_meta["fontSizePt"] = style_meta.get("fontSizePt", 14.5)
                append_formatted_runs(p, text, style_meta)
            elif "list" in b_type:
                doc.add_paragraph(strip_markdown_tokens(text), style="List Bullet")
            else:
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1.20
                align_str = str(style_meta.get("textAlign", "left")).lower()
                is_sig_right = is_right_half_signature_block(block, metrics)

                if is_sig_right:
                    # Đẩy sang nửa phải trang và căn giữa cột ký tên đúng Nghị định 30/2020/NĐ-CP
                    p.paragraph_format.left_indent = Inches(content_w_in * 0.44)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    # Chừa khoảng trống đóng dấu/ký tên nếu ảnh gốc có khoảng hở lớn phía trên tên người ký
                    if (
                        prev_row_bottom is not None
                        and row_top is not None
                        and (row_top - prev_row_bottom) > metrics["median_lh"] * 2.2
                    ):
                        p.paragraph_format.space_before = Pt(36)
                    else:
                        p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                elif align_str == "center":
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(3)
                elif align_str == "right" or "NHÀ XUẤT BẢN" in text.upper():
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(3)
                else:
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(3)
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align_str == "justify" else WD_ALIGN_PARAGRAPH.LEFT
                    is_bullet = bool(re.match(r"^(?:\*+)?[-–—+*•]", text))
                    if bool(style_meta.get("isIndented", False)) and not is_bullet:
                        p.paragraph_format.first_line_indent = Pt(28)

                append_formatted_runs(p, text, style_meta)

            row_idx += 1
            prev_row_bottom = row_bottom

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def generate_pdf_from_snapshot(snapshot: dict) -> bytes:
    """Tái tạo file PDF (.pdf) A4 @ 200 DPI giữ đúng hàng nhiều cột và khối chữ ký góc phải."""
    from PIL import Image, ImageDraw, ImageFont

    a4_w, a4_h = 1654, 2339
    margin_left = 195
    margin_right = 140
    margin_top = 150
    margin_bottom = 150
    content_w = a4_w - margin_left - margin_right

    serif_fonts = {
        (False, False): resolve_ttf_font("times.ttf", ["DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"]),
        (True, False): resolve_ttf_font("timesbd.ttf", ["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf"]),
        (False, True): resolve_ttf_font("timesi.ttf", ["DejaVuSerif-Italic.ttf", "LiberationSerif-Italic.ttf"]),
        (True, True): resolve_ttf_font("timesbi.ttf", ["DejaVuSerif-BoldItalic.ttf", "LiberationSerif-BoldItalic.ttf"]),
    }
    sans_fonts = {
        (False, False): resolve_ttf_font("arial.ttf", ["LiberationSans-Regular.ttf", "DejaVuSans.ttf"]),
        (True, False): resolve_ttf_font("arialbd.ttf", ["LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]),
        (False, True): resolve_ttf_font("ariali.ttf", ["LiberationSans-Italic.ttf", "DejaVuSans-Oblique.ttf"]),
        (True, True): resolve_ttf_font("arialbi.ttf", ["LiberationSans-BoldItalic.ttf", "DejaVuSans-BoldOblique.ttf"]),
    }

    def _get_font(family: str, is_b: bool, is_i: bool, size_px: int):
        tbl_f = sans_fonts if "arial" in (family or "").lower() else serif_fonts
        path = tbl_f.get((is_b, is_i)) or serif_fonts.get((is_b, is_i))
        if path:
            try:
                return ImageFont.truetype(path, size_px)
            except Exception:
                pass
        return ImageFont.load_default()

    def _wrap_block(draw_obj, clean_text: str, fnt, slot_w: int, first_indent: int):
        words = clean_text.split()
        if not words:
            return []
        wrapped: list[tuple[list[str], int, bool]] = []
        curr_words: list[str] = []
        first_line = True
        for w in words:
            ind_off = first_indent if first_line else 0
            avail_w = max(40, slot_w - ind_off)
            test_str = " ".join(curr_words + [w])
            tb = draw_obj.textbbox((0, 0), test_str, font=fnt)
            tw = tb[2] - tb[0]
            if tw <= avail_w or not curr_words:
                curr_words.append(w)
            else:
                wrapped.append((curr_words, ind_off, False))
                curr_words = [w]
                first_line = False
        if curr_words:
            wrapped.append((curr_words, first_indent if first_line else 0, True))
        return wrapped

    def _draw_wrapped_in_slot(
        draw_obj,
        wrapped_lines: list[tuple[list[str], int, bool]],
        fnt,
        size_px: int,
        slot_x0: int,
        slot_w: int,
        start_y: int,
        align: str,
        is_underline: bool,
    ) -> int:
        y = start_y
        line_step = int(round(size_px * 1.38))
        for line_words, indent_px, is_last_line in wrapped_lines:
            line_str = " ".join(line_words)
            tb = draw_obj.textbbox((0, 0), line_str, font=fnt)
            tw = tb[2] - tb[0]
            th = max(size_px, tb[3] - tb[1])

            if align == "center":
                draw_x = slot_x0 + max(0, (slot_w - tw) // 2)
                draw_obj.text((draw_x, y), line_str, fill=(15, 16, 20), font=fnt)
                if is_underline:
                    draw_obj.line([(draw_x, y + th + 3), (draw_x + tw, y + th + 3)], fill=(15, 16, 20), width=2)
            elif align == "right":
                draw_x = slot_x0 + max(0, slot_w - tw)
                draw_obj.text((draw_x, y), line_str, fill=(15, 16, 20), font=fnt)
                if is_underline:
                    draw_obj.line([(draw_x, y + th + 3), (draw_x + tw, y + th + 3)], fill=(15, 16, 20), width=2)
            elif align == "justify" and not is_last_line and len(line_words) > 2 and tw >= (slot_w - indent_px) * 0.72:
                word_widths = [draw_obj.textbbox((0, 0), wi, font=fnt)[2] - draw_obj.textbbox((0, 0), wi, font=fnt)[0] for wi in line_words]
                gap = (slot_w - indent_px - sum(word_widths)) / float(len(line_words) - 1)
                cur_x = float(slot_x0 + indent_px)
                for wi, ww in zip(line_words, word_widths):
                    draw_obj.text((int(round(cur_x)), y), wi, fill=(15, 16, 20), font=fnt)
                    cur_x += ww + gap
            else:
                draw_x = slot_x0 + indent_px
                draw_obj.text((draw_x, y), line_str, fill=(15, 16, 20), font=fnt)
                if is_underline:
                    draw_obj.line([(draw_x, y + th + 3), (draw_x + tw, y + th + 3)], fill=(15, 16, 20), width=2)
            y += line_step
        return y

    pdf_pages: list[Image.Image] = []
    raw_pages = snapshot.get("pages", []) or [{"layout": {"blocks": []}}]

    for page in raw_pages:
        canvas = Image.new("RGB", (a4_w, a4_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        curr_y = margin_top

        layout = page.get("layout", {}) or {}
        raw_blocks = list(layout.get("blocks", []) or [])
        metrics, rows = group_blocks_into_rows(
            raw_blocks,
            int(layout.get("width") or 1654),
            int(layout.get("height") or 2339),
        )

        prev_row_bottom = None
        for row in rows:
            row_top, row_bottom = _row_vertical_bounds(row)

            if len(row) >= 2:
                mode, col_specs = classify_row_columns(row, metrics)
                prepared: list[tuple[int, int, Any, int, list, str, bool]] = []
                cur_slot_x = margin_left
                max_row_h = 0
                for c_i, blk in enumerate(row):
                    frac, cell_align = col_specs[c_i]
                    slot_w = max(80, int(round(content_w * frac)))
                    s_meta = blk.get("style") or {}
                    raw_txt = (blk.get("text") or "").strip()
                    is_h = "heading" in str(blk.get("type", "")).lower()
                    is_b = bool(s_meta.get("bold", False) or is_h or ("**" in raw_txt))
                    is_i = bool(s_meta.get("italic", False) or (raw_txt.startswith("*") and raw_txt.endswith("*")))
                    is_u = bool(s_meta.get("underline", False) or ("<u>" in raw_txt) or ("__" in raw_txt))
                    clean_txt = strip_markdown_tokens(raw_txt)
                    if is_h and clean_txt.isupper():
                        clean_txt = clean_txt.upper()
                    pt_val = float(s_meta.get("fontSizePt", 13.0))
                    if mode == "center_pair":
                        pt_val = min(13.0, pt_val)
                    sz_px = int(round(max(11.0, min(20.0, pt_val)) * (200.0 / 72.0)))
                    fnt = _get_font(str(s_meta.get("fontFamily") or ""), is_b, is_i, sz_px)
                    wrapped = _wrap_block(draw, clean_txt, fnt, slot_w - 8, 0)
                    blk_h = max(1, len(wrapped)) * int(round(sz_px * 1.38))
                    max_row_h = max(max_row_h, blk_h)
                    prepared.append((cur_slot_x, slot_w, fnt, sz_px, wrapped, cell_align, is_u))
                    cur_slot_x += slot_w

                if curr_y + max_row_h > a4_h - margin_bottom:
                    pdf_pages.append(canvas)
                    canvas = Image.new("RGB", (a4_w, a4_h), (255, 255, 255))
                    draw = ImageDraw.Draw(canvas)
                    curr_y = margin_top

                for s_x0, s_w, fnt, sz_px, wrapped, c_align, is_u in prepared:
                    _draw_wrapped_in_slot(draw, wrapped, fnt, sz_px, s_x0, s_w, curr_y, c_align, is_u)

                curr_y += max_row_h + (2 if mode == "center_pair" else 8)
                prev_row_bottom = row_bottom
                continue

            # Hàng đơn 1 block
            block = row[0]
            b_type = str(block.get("type", "paragraph")).lower()
            raw_text = (block.get("text") or "").strip()
            if not raw_text and not block.get("tableData"):
                continue

            style_meta = block.get("style", {}) or {}
            is_heading = "heading" in b_type or b_type == "title"
            is_bold = bool(style_meta.get("bold", False) or is_heading or ("**" in raw_text))
            is_italic = bool(style_meta.get("italic", False) or (raw_text.startswith("*") and raw_text.endswith("*")))
            is_underline = bool(style_meta.get("underline", False) or ("<u>" in raw_text) or ("__" in raw_text))
            clean_text = strip_markdown_tokens(raw_text)
            if is_heading and clean_text.isupper():
                clean_text = clean_text.upper()

            size_pt = float(style_meta.get("fontSizePt", 14.0 if is_heading else 13.0))
            size_px = int(round(max(10.0, min(20.0, size_pt)) * (200.0 / 72.0)))
            fnt = _get_font(str(style_meta.get("fontFamily") or ""), is_bold, is_italic, size_px)

            # Vẽ bảng có khung viền rõ nét nếu là Table
            table_data = block.get("tableData") if "table" in b_type else None
            if isinstance(table_data, dict) and table_data.get("headers"):
                headers = table_data.get("headers", [])
                t_rows = table_data.get("rows", [])
                n_cols = max(1, len(headers))
                col_w_px = content_w // n_cols
                f_tbl = _get_font(str(style_meta.get("fontFamily") or ""), False, False, int(round(12.0 * 200.0 / 72.0)))
                f_hdr = _get_font(str(style_meta.get("fontFamily") or ""), True, False, int(round(12.0 * 200.0 / 72.0)))
                all_tbl_rows = [(headers, f_hdr, "center")] + [(r, f_tbl, "left") for r in t_rows]
                for r_vals, r_fnt, r_align in all_tbl_rows:
                    max_lines = 1
                    cell_wraps = []
                    for c_i in range(n_cols):
                        val = strip_markdown_tokens(str(r_vals[c_i] if c_i < len(r_vals) else ""))
                        wr = _wrap_block(draw, val, r_fnt, col_w_px - 16, 0)
                        cell_wraps.append(wr)
                        max_lines = max(max_lines, len(wr))
                    row_px_h = max(46, max_lines * 42 + 14)
                    if curr_y + row_px_h > a4_h - margin_bottom:
                        pdf_pages.append(canvas)
                        canvas = Image.new("RGB", (a4_w, a4_h), (255, 255, 255))
                        draw = ImageDraw.Draw(canvas)
                        curr_y = margin_top
                    for c_i, wr in enumerate(cell_wraps):
                        cx0 = margin_left + c_i * col_w_px
                        draw.rectangle([cx0, curr_y, cx0 + col_w_px, curr_y + row_px_h], outline=(40, 40, 40), width=2)
                        _draw_wrapped_in_slot(draw, wr, r_fnt, 32, cx0 + 8, col_w_px - 16, curr_y + 8, r_align, False)
                    curr_y += row_px_h
                curr_y += 12
                prev_row_bottom = row_bottom
                continue

            align = str(style_meta.get("textAlign", "center" if is_heading else "left")).lower()
            if "nhà xuất bản" in clean_text.lower():
                align = "right"
            is_sig_right = is_right_half_signature_block(block, metrics)
            is_bullet = bool(re.match(r"^(?:\*+)?[-–—+*•]", raw_text))
            is_indented = bool(style_meta.get("isIndented", False)) and not is_bullet and align not in ("center", "right")

            if is_heading:
                curr_y += 10
            elif is_sig_right and prev_row_bottom is not None and row_top is not None:
                if (row_top - prev_row_bottom) > metrics["median_lh"] * 2.2:
                    curr_y += 72

            if is_sig_right:
                sig_slot_x0 = margin_left + int(content_w * 0.44)
                sig_slot_w = content_w - int(content_w * 0.44)
                wrapped = _wrap_block(draw, clean_text, fnt, sig_slot_w, 0)
                blk_h = max(1, len(wrapped)) * int(round(size_px * 1.38))
                if curr_y + blk_h > a4_h - margin_bottom:
                    pdf_pages.append(canvas)
                    canvas = Image.new("RGB", (a4_w, a4_h), (255, 255, 255))
                    draw = ImageDraw.Draw(canvas)
                    curr_y = margin_top
                curr_y = _draw_wrapped_in_slot(draw, wrapped, fnt, size_px, sig_slot_x0, sig_slot_w, curr_y, "center", is_underline)
                curr_y += 4
            else:
                indent_px = 70 if is_indented else 0
                wrapped = _wrap_block(draw, clean_text, fnt, content_w, indent_px)
                for l_words, l_ind, is_last in wrapped:
                    th_line = int(round(size_px * 1.38))
                    if curr_y + th_line > a4_h - margin_bottom:
                        pdf_pages.append(canvas)
                        canvas = Image.new("RGB", (a4_w, a4_h), (255, 255, 255))
                        draw = ImageDraw.Draw(canvas)
                        curr_y = margin_top
                    _draw_wrapped_in_slot(draw, [(l_words, l_ind, is_last)], fnt, size_px, margin_left, content_w, curr_y, align, is_underline)
                    curr_y += th_line
                curr_y += 6

            prev_row_bottom = row_bottom

        pdf_pages.append(canvas)

    buf = io.BytesIO()
    pdf_pages[0].save(
        buf,
        format="PDF",
        resolution=200.0,
        save_all=True,
        append_images=pdf_pages[1:],
    )
    return buf.getvalue()
