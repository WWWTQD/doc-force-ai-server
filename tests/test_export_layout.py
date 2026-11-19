import io
import docx

from worker.export_layout import (
    classify_row_columns,
    generate_docx_from_snapshot,
    generate_pdf_from_snapshot,
    group_blocks_into_rows,
    is_right_half_signature_block,
)


def _sample_certificate_blocks():
    return [
        {
            "id": "b-1",
            "type": "heading",
            "order": 1,
            "bbox": [310, 140, 820, 185],
            "text": "ĐẠI HỌC ĐÀ NẴNG",
            "confidence": 0.98,
            "style": {"bold": True, "fontSizePt": 13.0, "textAlign": "center"},
        },
        {
            "id": "b-2",
            "type": "heading",
            "order": 2,
            "bbox": [1120, 138, 2240, 184],
            "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "confidence": 0.99,
            "style": {"bold": True, "fontSizePt": 13.0, "textAlign": "center"},
        },
        {
            "id": "b-3",
            "type": "heading",
            "order": 3,
            "bbox": [240, 195, 910, 240],
            "text": "TRƯỜNG ĐẠI HỌC BÁCH KHOA",
            "confidence": 0.98,
            "style": {"bold": True, "underline": True, "fontSizePt": 13.0, "textAlign": "center"},
        },
        {
            "id": "b-4",
            "type": "heading",
            "order": 4,
            "bbox": [1260, 196, 2090, 242],
            "text": "Độc lập - Tự do - Hạnh phúc",
            "confidence": 0.98,
            "style": {"bold": True, "underline": True, "fontSizePt": 13.0, "textAlign": "center"},
        },
        {
            "id": "b-5",
            "type": "heading",
            "order": 5,
            "bbox": [460, 390, 2120, 445],
            "text": "TRƯỜNG ĐẠI HỌC BÁCH KHOA - ĐẠI HỌC ĐÀ NẴNG",
            "confidence": 0.98,
            "style": {"bold": True, "fontSizePt": 16.0, "textAlign": "center"},
        },
        {
            "id": "b-6",
            "type": "heading",
            "order": 6,
            "bbox": [1100, 465, 1480, 520],
            "text": "XÁC NHẬN",
            "confidence": 0.99,
            "style": {"bold": True, "fontSizePt": 15.5, "textAlign": "center"},
        },
        {
            "id": "b-7",
            "type": "paragraph",
            "order": 7,
            "bbox": [240, 620, 880, 665],
            "text": "Ông/bà: Nguyễn Văn A",
            "confidence": 0.97,
            "style": {"fontSizePt": 13.0, "textAlign": "left"},
        },
        {
            "id": "b-8",
            "type": "paragraph",
            "order": 8,
            "bbox": [1050, 619, 1590, 664],
            "text": "Sinh ngày: 01/01/2006",
            "confidence": 0.97,
            "style": {"fontSizePt": 13.0, "textAlign": "left"},
        },
        {
            "id": "b-9",
            "type": "paragraph",
            "order": 9,
            "bbox": [1820, 621, 2240, 665],
            "text": "Giới tính: Nam",
            "confidence": 0.97,
            "style": {"fontSizePt": 13.0, "textAlign": "left"},
        },
        {
            "id": "b-10",
            "type": "paragraph",
            "order": 10,
            "bbox": [240, 740, 2290, 860],
            "text": "Đã trúng tuyển và nhập học tại Trường Đại học Bách khoa năm 2024 với thông tin chi tiết như sau:",
            "confidence": 0.96,
            "style": {"fontSizePt": 13.0, "textAlign": "justify", "isIndented": False},
        },
        {
            "id": "b-11",
            "type": "paragraph",
            "order": 11,
            "bbox": [240, 890, 760, 936],
            "text": "- Lớp: 24T_DT1",
            "confidence": 0.95,
            "style": {"fontSizePt": 13.0, "textAlign": "left"},
        },
        {
            "id": "b-12",
            "type": "paragraph",
            "order": 12,
            "bbox": [1120, 891, 2200, 936],
            "text": "thuộc khoa: Công nghệ thông tin",
            "confidence": 0.95,
            "style": {"fontSizePt": 13.0, "textAlign": "left"},
        },
        {
            "id": "b-13",
            "type": "paragraph",
            "order": 13,
            "bbox": [240, 960, 950, 1005],
            "text": "- Khóa học: 2024 - 2028",
            "confidence": 0.96,
            "style": {"fontSizePt": 13.0, "textAlign": "left"},
        },
        {
            "id": "b-14",
            "type": "paragraph",
            "order": 14,
            "bbox": [340, 1040, 2290, 1140],
            "text": "Đoạn văn bản thụt đầu dòng bình thường kéo dài hết chiều ngang cột chữ.",
            "confidence": 0.97,
            "style": {"fontSizePt": 13.0, "textAlign": "justify", "isIndented": True},
        },
        {
            "id": "b-15",
            "type": "paragraph",
            "order": 15,
            "bbox": [1320, 1800, 2190, 1848],
            "text": "Đà Nẵng, ngày 27 tháng 8 năm 2024",
            "confidence": 0.97,
            "style": {"fontSizePt": 13.0, "italic": True, "textAlign": "right"},
        },
        {
            "id": "b-16",
            "type": "paragraph",
            "order": 16,
            "bbox": [1460, 1865, 2040, 1910],
            "text": "TL. HIỆU TRƯỞNG",
            "confidence": 0.98,
            "style": {"fontSizePt": 13.0, "bold": True, "textAlign": "right"},
        },
        {
            "id": "b-17",
            "type": "paragraph",
            "order": 17,
            "bbox": [1470, 2190, 2030, 2240],
            "text": "TS. Huỳnh Hữu Hưng",
            "confidence": 0.98,
            "style": {"fontSizePt": 13.0, "bold": True, "textAlign": "right"},
        },
    ]


def test_group_and_classify_horizontal_rows():
    blocks = _sample_certificate_blocks()
    metrics, rows = group_blocks_into_rows(blocks, 2576, 3632)

    assert len(rows[0]) == 2
    assert rows[0][0]["text"] == "ĐẠI HỌC ĐÀ NẴNG"
    assert rows[0][1]["text"] == "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
    mode0, specs0 = classify_row_columns(rows[0], metrics)
    assert mode0 == "center_pair"
    assert abs(sum(w for w, _ in specs0) - 1.0) < 1e-6

    # Row with 3 form fields
    form_rows = [r for r in rows if len(r) == 3]
    assert len(form_rows) == 1
    assert [b["text"].split(":")[0] for b in form_rows[0]] == ["Ông/bà", "Sinh ngày", "Giới tính"]
    mode_f, specs_f = classify_row_columns(form_rows[0], metrics)
    assert mode_f == "form_grid"
    assert [a for _, a in specs_f] == ["left", "left", "left"]

    # Standalone signature detection
    assert is_right_half_signature_block(blocks[-1], metrics) is True
    assert is_right_half_signature_block(blocks[4], metrics) is False


def test_generate_docx_and_pdf_preserve_multicolumn_and_signature():
    blocks = _sample_certificate_blocks()
    snapshot = {
        "documentId": "test-doc",
        "title": "Giay_xac_nhan",
        "pages": [
            {
                "pageNumber": 1,
                "layout": {
                    "schemaVersion": 1,
                    "pageNumber": 1,
                    "width": 2576,
                    "height": 3632,
                    "blocks": blocks,
                },
            }
        ],
    }

    docx_bytes = generate_docx_from_snapshot(snapshot)
    assert len(docx_bytes) > 2000
    parsed = docx.Document(io.BytesIO(docx_bytes))
    # 1) Quốc hiệu table (2x2), 2) Ông/bà table (1x3), 3) - Lớp | thuộc khoa table (1x2)
    assert len(parsed.tables) == 3
    assert parsed.tables[0].rows[0].cells[0].text.strip() == "ĐẠI HỌC ĐÀ NẴNG"
    assert parsed.tables[0].rows[0].cells[1].text.strip() == "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
    assert len(parsed.tables[1].columns) == 3
    assert parsed.tables[2].rows[0].cells[1].text.strip() == "thuộc khoa: Công nghệ thông tin"

    sig_paras = [p for p in parsed.paragraphs if "Huỳnh Hữu Hưng" in p.text]
    assert len(sig_paras) == 1
    assert sig_paras[0].paragraph_format.left_indent is not None
    assert sig_paras[0].paragraph_format.space_before.pt >= 30

    bullet_paras = [p for p in parsed.paragraphs if "Khóa học" in p.text]
    assert len(bullet_paras) == 1
    assert bullet_paras[0].paragraph_format.first_line_indent is None

    indented_paras = [p for p in parsed.paragraphs if "thụt đầu dòng" in p.text]
    assert len(indented_paras) == 1
    assert indented_paras[0].paragraph_format.first_line_indent.pt == 28

    pdf_bytes = generate_pdf_from_snapshot(snapshot)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000
