import pytest
from app.services.structure_parser import StructureParser
from app.schemas.structure import AnalyzeResponse


def test_normalize_label():
    norm, raw = StructureParser.normalize_label("doc_title")
    assert norm == "title"
    assert raw == "doc_title"

    norm, raw = StructureParser.normalize_label("figure")
    assert norm == "image"
    assert raw == "figure"

    norm, raw = StructureParser.normalize_label("equation")
    assert norm == "formula"
    assert raw == "equation"

    norm, raw = StructureParser.normalize_label("custom_seal")
    assert norm == "custom_seal"


def test_structure_parser_parse_simple():
    sample_raw = [
        {
            "page_index": 1,
            "width": 1000,
            "height": 800,
            "parsing_res_list": [
                {
                    "label": "doc_title",
                    "bbox": [50, 50, 500, 100],
                    "score": 0.98,
                    "text": "BÁO CÁO PBL4",
                },
                {
                    "label": "table",
                    "bbox": [50, 150, 950, 600],
                    "score": 0.92,
                    "html": "<table><tr><td>Header</td></tr></table>",
                    "cell_box_list": [[50, 150, 950, 600]],
                },
            ],
        }
    ]

    response = StructureParser.parse(
        raw_results=sample_raw,
        file_id="test-uuid-123",
        filename="test_doc.png",
        processing_time_ms=150.5,
    )

    assert isinstance(response, AnalyzeResponse)
    assert response.success is True
    assert response.file_id == "test-uuid-123"
    assert response.filename == "test_doc.png"
    assert len(response.pages) == 1

    page = response.pages[0]
    assert page.page_number == 1
    assert page.width == 1000
    assert page.height == 800
    assert len(page.elements) == 2

    title_elem = page.elements[0]
    assert title_elem.id == "element-1"
    assert title_elem.type == "title"
    assert title_elem.text == "BÁO CÁO PBL4"
    assert title_elem.bbox == [50.0, 50.0, 500.0, 100.0]
    assert title_elem.confidence == 0.98

    table_elem = page.elements[1]
    assert table_elem.id == "element-2"
    assert table_elem.type == "table"
    assert table_elem.html == "<table><tr><td>Header</td></tr></table>"
