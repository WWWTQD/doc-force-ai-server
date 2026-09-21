import os
import pytest
from app.services.reconstruction import DocumentReconstructionEngine, BlockType
from app.services.pdf_exporter import PDFExporter
from app.services.docx_exporter import DOCXExporter


def test_document_reconstruction_engine():
    dummy_pp_results = [
        {
            "width": 800,
            "height": 1000,
            "parsing_res_list": [
                {"block_label": "doc_title", "block_content": "PBL4 Report", "block_bbox": [50, 50, 400, 90]},
                {"block_label": "text", "block_content": "Sample paragraph text.", "block_bbox": [50, 100, 700, 150]},
                {"block_label": "table", "block_content": "Col1 | Col2\nVal1 | Val2", "block_bbox": [50, 200, 700, 350]}
            ]
        }
    ]

    dom = DocumentReconstructionEngine.build_dom(dummy_pp_results, title="Test Report")
    assert dom.title == "Test Report"
    assert len(dom.pages) == 1
    assert len(dom.pages[0].blocks) == 3

    blocks = dom.pages[0].blocks
    assert blocks[0].block_type == BlockType.TITLE
    assert blocks[0].content == "PBL4 Report"
    assert blocks[1].block_type == BlockType.PARAGRAPH
    assert blocks[2].block_type == BlockType.TABLE


def test_pdf_exporter(tmp_path):
    dummy_pp_results = [
        {
            "width": 800,
            "height": 1000,
            "parsing_res_list": [
                {"block_label": "doc_title", "block_content": "PDF Export Test", "block_bbox": [50, 50, 400, 90]}
            ]
        }
    ]
    dom = DocumentReconstructionEngine.build_dom(dummy_pp_results)
    pdf_out = str(tmp_path / "test_doc.pdf")
    res_path = PDFExporter.export_to_pdf(dom, pdf_out)
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 0


def test_docx_exporter(tmp_path):
    dummy_pp_results = [
        {
            "width": 800,
            "height": 1000,
            "parsing_res_list": [
                {"block_label": "doc_title", "block_content": "DOCX Export Test", "block_bbox": [50, 50, 400, 90]}
            ]
        }
    ]
    dom = DocumentReconstructionEngine.build_dom(dummy_pp_results)
    docx_out = str(tmp_path / "test_doc.docx")
    res_path = DOCXExporter.export_to_docx(dom, docx_out)
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 0
