#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Raw Documents (PDF, DOCX, DOC) to Clean Ground Truth PNG Images.

Pipeline:
1. Scan input directories for .pdf, .docx, .doc files.
2. Convert Word (.docx, .doc) to PDF using Word COM automation (Method 1)
   with per-file timeout and automatic fallback to python-docx + PIL rendering (Method 2).
3. Render PDF pages to sharp lossless PNGs at 150-200 DPI using PyMuPDF (fitz).
4. Filter out blank / near-blank pages (dark pixel density < 0.5%).
5. Normalize pure white background while strictly preserving anti-aliased font edges.
6. Save ground truth images to datasets/clean_originals/ with naming:
   clean_<clean_stem>_p<page_number>.png
"""

import os
import sys
import re
import gc
import time
import shutil
import logging
import argparse
import tempfile
import textwrap
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 output encoding for Vietnamese paths and text
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("convert_docs_to_clean")

# PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import pymupdf as fitz
    except ImportError:
        logger.error("PyMuPDF is not installed. Please run: pip install pymupdf")
        sys.exit(1)

# python-docx for fallback
try:
    import docx
except ImportError:
    docx = None


def sanitize_filename_stem(stem: str) -> str:
    """Sanitize original document stem into a clean filesystem-friendly identifier."""
    clean = re.sub(r"\s+", "_", stem.strip())
    clean = re.sub(r'[\\/*?:"<>|]', "", clean)
    clean = re.sub(r"_+", "_", clean)
    clean = clean.strip("._")
    return clean if clean else "doc"


def scan_input_documents(search_dirs: List[Path]) -> Dict[str, List[Path]]:
    """
    Scan target directories for .pdf, .docx, and .doc documents.
    Excludes temporary lock files (e.g., ~$*.docx).
    """
    collected = {".pdf": [], ".docx": [], ".doc": []}
    seen_canonical = set()

    for directory in search_dirs:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            # Skip temporary Office lock files
            if path.name.startswith("~$") or path.name.startswith("._") or path.suffix.lower() == ".tmp":
                continue

            ext = path.suffix.lower()
            if ext in collected:
                canonical = str(path.resolve()).lower()
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    collected[ext].append(path)

    return collected


def unblock_files(file_paths: List[Path]):
    """Remove Windows Zone.Identifier (Mark of the Web) to prevent Office Protected View locks."""
    if not file_paths:
        return
    try:
        for p in file_paths:
            # Delete Zone.Identifier stream if exists
            zone_stream = Path(f"{str(p)}:Zone.Identifier")
            if zone_stream.exists():
                zone_stream.unlink(missing_ok=True)
    except Exception:
        pass


def convert_single_word_doc_com(
    doc_path: Path, temp_dir: Path, timeout: int = 20
) -> Optional[Path]:
    """
    Convert a single Word document to PDF using MS Word COM Automation via PowerShell.
    Protected with a strict timeout; cleans up any COM process if timed out.
    """
    clean_stem = sanitize_filename_stem(doc_path.stem)
    out_pdf = temp_dir / f"com_{clean_stem}.pdf"

    # PowerShell script to convert a single file safely
    ps_content = f"""# Single file Word COM
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$src = @'
{str(doc_path.resolve())}
'@
$dst = @'
{str(out_pdf.resolve())}
'@

$word = $null
try {{
    Unblock-File -LiteralPath $src -ErrorAction SilentlyContinue
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($src, $false, $true) # ReadOnly
    if ($doc -ne $null) {{
        $doc.SaveAs([ref]$dst, [ref]17) # wdFormatPDF = 17
        $doc.Close([ref]0)
        exit 0
    }}
    exit 2
}} catch {{
    Write-Error $_.Exception.Message
    exit 1
}} finally {{
    if ($word -ne $null) {{
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }}
}}
"""
    ps_file = temp_dir / f"convert_{clean_stem}.ps1"
    with open(ps_file, "w", encoding="utf-8-sig") as f:
        f.write(ps_content)

    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        ps_file.unlink(missing_ok=True)
        if proc.returncode == 0 and out_pdf.exists() and out_pdf.stat().st_size > 0:
            return out_pdf
        else:
            logger.warning(f"  [Word COM Warning] {doc_path.name}: {proc.stderr.strip()[:150]}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"  [Word COM Timeout] {doc_path.name} exceeded {timeout}s. Falling back to Method 2.")
        ps_file.unlink(missing_ok=True)
        # Kill any hung WINWORD process created for this call
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-Process WINWORD -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning(f"  [Word COM Error] {doc_path.name}: {e}")
        ps_file.unlink(missing_ok=True)
        return None


def render_docx_fallback(doc_path: Path) -> List[Image.Image]:
    """
    Method 2 (Fallback): Parse paragraphs and tables from .docx using python-docx
    and render to clean white 1240 x 1754 px A4 pages using PIL.ImageDraw with system fonts.
    Optimized for high performance using textwrap.
    """
    if docx is None:
        logger.error(f"Cannot run fallback for {doc_path.name}: python-docx not installed.")
        return []

    logger.info(f"Rendering '{doc_path.name}' via Method 2 Fallback (python-docx + PIL)...")
    try:
        document = docx.Document(str(doc_path))
    except Exception as e:
        logger.error(f"Failed to open {doc_path.name} with python-docx: {e}")
        return []

    page_w, page_h = 1240, 1754
    margin_x = 80
    margin_y = 90
    max_w = page_w - 2 * margin_x
    max_h = page_h - margin_y

    fonts_dir = Path("C:/Windows/Fonts")
    font_body = None
    font_bold = None
    font_heading = None

    for f_name, f_bold in [
        ("times.ttf", "timesbd.ttf"),
        ("arial.ttf", "arialbd.ttf"),
        ("calibri.ttf", "calibrib.ttf"),
    ]:
        p_body = fonts_dir / f_name
        p_bold = fonts_dir / f_bold
        if p_body.exists():
            try:
                font_body = ImageFont.truetype(str(p_body), 20)
                font_bold = ImageFont.truetype(str(p_bold if p_bold.exists() else p_body), 20)
                font_heading = ImageFont.truetype(str(p_bold if p_bold.exists() else p_body), 26)
                break
            except Exception:
                pass

    if font_body is None:
        font_body = ImageFont.load_default()
        font_bold = font_body
        font_heading = font_body

    pages: List[Image.Image] = []
    current_img = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    draw = ImageDraw.Draw(current_img)
    cursor_y = margin_y

    def new_page():
        nonlocal current_img, draw, cursor_y
        pages.append(current_img)
        current_img = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        draw = ImageDraw.Draw(current_img)
        cursor_y = margin_y

    # Process paragraphs
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            cursor_y += 14
            if cursor_y > max_h:
                new_page()
            continue

        is_heading = para.style.name.lower().startswith("heading")
        fnt = font_heading if is_heading else font_body
        line_height = 36 if is_heading else 28
        wrap_chars = 65 if is_heading else 80

        lines = textwrap.wrap(text, width=wrap_chars)
        for line in lines:
            if cursor_y + line_height > max_h:
                new_page()
            draw.text((margin_x, cursor_y), line, fill=(0, 0, 0), font=fnt)
            cursor_y += line_height

        cursor_y += 8

    # Process tables
    for tbl in document.tables:
        if cursor_y + 60 > max_h:
            new_page()

        num_cols = len(tbl.columns)
        if num_cols == 0:
            continue
        col_w = max_w // num_cols
        col_chars = max(10, (col_w - 20) // 12)

        for row in tbl.rows:
            row_texts = [cell.text.strip() for cell in row.cells]
            cell_lines = [textwrap.wrap(t, width=col_chars) for t in row_texts]
            max_lines = max((len(cl) for cl in cell_lines), default=1)
            row_h = max(35, max_lines * 24 + 12)

            if cursor_y + row_h > max_h:
                new_page()

            for c_idx, cl in enumerate(cell_lines):
                cx0 = margin_x + c_idx * col_w
                cx1 = cx0 + col_w
                draw.rectangle([(cx0, cursor_y), (cx1, cursor_y + row_h)], outline=(180, 180, 180), width=1)
                text_y = cursor_y + 6
                for l in cl:
                    draw.text((cx0 + 8, text_y), l, fill=(20, 20, 20), font=font_body)
                    text_y += 24

            cursor_y += row_h
        cursor_y += 18

    if cursor_y > margin_y:
        pages.append(current_img)

    logger.info(f"  Fallback rendered {len(pages)} page(s) for '{doc_path.name}'")
    return pages


def calculate_dark_text_ratio(
    rgb_arr: np.ndarray, dark_threshold: int = 220
) -> Tuple[float, float]:
    """
    Calculate the ratio of dark (text / stroke) pixels on a page.
    Returns (dark_ratio, median_luminance).
    """
    r = rgb_arr[:, :, 0].astype(np.float32)
    g = rgb_arr[:, :, 1].astype(np.float32)
    b = rgb_arr[:, :, 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    total_pixels = lum.size
    dark_pixels = np.sum(lum < dark_threshold)
    dark_ratio = float(dark_pixels) / float(total_pixels)
    median_lum = float(np.median(lum))

    return dark_ratio, median_lum


def normalize_clean_background(rgb_arr: np.ndarray, median_lum: float) -> np.ndarray:
    """
    Normalize near-white background pixels to pure 255 (RGB 255,255,255) for vector/digital docs.
    STRICT SAFETY RULE: Absolutely NO Otsu or Sauvola binarization!
    All anti-aliased font strokes (values < 240) remain 100% natural and unquantized.
    """
    if median_lum >= 235.0:
        bg_mask = (rgb_arr[:, :, 0] > 250) & (rgb_arr[:, :, 1] > 250) & (rgb_arr[:, :, 2] > 250)
        rgb_arr[bg_mask] = 255
    return rgb_arr


def process_pdf_to_clean_images(
    pdf_path: Path,
    original_name: str,
    output_dir: Path,
    dpi_scale: float = 2.0,
    min_text_ratio: float = 0.005,
    dark_threshold: int = 220,
) -> List[Tuple[Path, Tuple[int, int], int]]:
    """
    Render PDF pages to sharp lossless PNG ground truth images using PyMuPDF.
    Returns list of (output_file_path, (width, height), file_size_bytes).
    """
    created_images = []
    clean_stem = sanitize_filename_stem(Path(original_name).stem)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error(f"Cannot open PDF {pdf_path.name}: {e}")
        return created_images

    if doc.is_encrypted:
        auth = doc.authenticate("")
        if auth == 0:
            logger.warning(
                f"[SKIPPED] PDF '{original_name}' is password-protected and cannot be opened without credentials."
            )
            doc.close()
            return created_images

    total_pages = doc.page_count
    logger.info(f"Processing PDF '{original_name}' ({total_pages} page(s))...")

    matrix = fitz.Matrix(dpi_scale, dpi_scale)

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        try:
            page = doc.load_page(page_idx)
            # Render RGB onto white background (alpha=False ensures transparent areas composite to pure white)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).copy().reshape(pix.height, pix.width, 3)

            # Check text density
            dark_ratio, median_lum = calculate_dark_text_ratio(arr, dark_threshold=dark_threshold)

            if dark_ratio < min_text_ratio:
                logger.info(
                    f"  [FILTERED BLANK] Page {page_num}/{total_pages}: text ratio = {dark_ratio*100:.3f}% < {min_text_ratio*100:.1f}%. Skipped."
                )
                del pix, arr
                continue

            # Pure white background normalization (keeping smooth anti-aliased font edges)
            arr = normalize_clean_background(arr, median_lum)

            out_filename = f"clean_{clean_stem}_p{page_num}.png"
            out_path = output_dir / out_filename

            # Save as lossless PNG via Pillow
            out_img = Image.fromarray(arr, mode="RGB")
            out_img.save(str(out_path), format="PNG", compress_level=4)

            file_size = out_path.stat().st_size
            created_images.append((out_path, (pix.width, pix.height), file_size))
            logger.info(
                f"  -> Generated: {out_filename} ({pix.width}x{pix.height}, text={dark_ratio*100:.2f}%, {file_size/1024:.1f} KB)"
            )

            del pix, arr, out_img
        except Exception as e:
            logger.error(f"  Error rendering page {page_num} of '{original_name}': {e}")

        if page_num % 30 == 0:
            gc.collect()

    doc.close()
    gc.collect()
    return created_images


def main():
    parser = argparse.ArgumentParser(
        description="Convert Word (.docx, .doc) and PDF documents to sharp lossless clean PNG images."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=[
            r"C:\PBL4\doc-force-ai-server\datasets\raw_docs",
            r"C:\PBL4\doc-force-ai-server\datasets\clean_originals",
            r"C:\PBL4\doc-force-ai-server\datasets",
        ],
        help="Directories to scan for documents.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\PBL4\doc-force-ai-server\datasets\clean_originals",
        help="Output directory for clean ground truth images.",
    )
    parser.add_argument(
        "--dpi_scale",
        type=float,
        default=2.0,
        help="PyMuPDF render matrix scale (2.0 ~ 150-200 DPI).",
    )
    parser.add_argument(
        "--min_text_ratio",
        type=float,
        default=0.005,
        help="Minimum dark pixel ratio to keep page (default 0.005 = 0.5%%).",
    )
    parser.add_argument(
        "--dark_threshold",
        type=int,
        default=220,
        help="Pixel luminance threshold to count as text/stroke.",
    )
    parser.add_argument(
        "--skip_word_com",
        action="store_true",
        help="Skip Word COM automation and use python-docx fallback directly.",
    )

    args = parser.parse_args()

    start_time = time.time()
    search_dirs = [Path(d) for d in args.input_dirs]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="doc_clean_convert_"))
    logger.info(f"Starting Document Conversion Pipeline")
    logger.info(f"Target Output Directory: {out_dir}")
    logger.info(f"Temporary Work Directory: {temp_dir}")

    # 1. Scan input documents
    docs = scan_input_documents(search_dirs)
    all_pdfs = docs[".pdf"]
    all_docx = docs[".docx"] + docs[".doc"]

    total_scanned = len(all_pdfs) + len(all_docx)
    logger.info(
        f"Scan Complete: {total_scanned} document(s) found "
        f"({len(all_pdfs)} PDF, {len(all_docx)} Word)"
    )

    if total_scanned == 0:
        logger.warning("No documents found in specified directories.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Unblock any web files
    unblock_files(all_docx + all_pdfs)

    all_created_images: List[Tuple[Path, Tuple[int, int], int]] = []
    processed_files_count = 0

    # 2. Process Word Documents (.docx, .doc)
    logger.info("\n" + "=" * 50)
    logger.info(f"PHASE 1: Processing {len(all_docx)} Word Document(s)...")
    logger.info("=" * 50)

    for doc_idx, doc_file in enumerate(all_docx, start=1):
        logger.info(f"\n[{doc_idx}/{len(all_docx)}] Handling Word document: '{doc_file.name}'")
        com_pdf = None

        if not args.skip_word_com:
            com_pdf = convert_single_word_doc_com(doc_file, temp_dir, timeout=25)

        if com_pdf and com_pdf.exists():
            logger.info(f"  [Word COM Succeeded] Converted '{doc_file.name}' -> {com_pdf.stat().st_size:,} bytes")
            res = process_pdf_to_clean_images(
                pdf_path=com_pdf,
                original_name=doc_file.name,
                output_dir=out_dir,
                dpi_scale=args.dpi_scale,
                min_text_ratio=args.min_text_ratio,
                dark_threshold=args.dark_threshold,
            )
            all_created_images.extend(res)
            com_pdf.unlink(missing_ok=True)
            processed_files_count += 1
        else:
            # Fallback to python-docx + PIL
            logger.info(f"  [Using Fallback] Rendering '{doc_file.name}' with python-docx + PIL...")
            pages = render_docx_fallback(doc_file)
            clean_stem = sanitize_filename_stem(doc_file.stem)
            for p_idx, page_img in enumerate(pages):
                page_num = p_idx + 1
                arr = np.array(page_img)
                dark_ratio, med_lum = calculate_dark_text_ratio(arr, args.dark_threshold)
                if dark_ratio < args.min_text_ratio:
                    logger.info(
                        f"  [FILTERED BLANK] Fallback Page {page_num}: text ratio = {dark_ratio*100:.3f}%. Skipped."
                    )
                    continue
                arr = normalize_clean_background(arr, med_lum)
                out_name = f"clean_{clean_stem}_p{page_num}.png"
                out_path = out_dir / out_name
                Image.fromarray(arr).save(str(out_path), format="PNG", compress_level=4)
                file_size = out_path.stat().st_size
                all_created_images.append((out_path, (arr.shape[1], arr.shape[0]), file_size))
                logger.info(
                    f"  -> Generated (Fallback): {out_name} ({arr.shape[1]}x{arr.shape[0]}, text={dark_ratio*100:.2f}%, {file_size/1024:.1f} KB)"
                )
            processed_files_count += 1

    # 3. Process Original PDF Documents (.pdf)
    logger.info("\n" + "=" * 50)
    logger.info(f"PHASE 2: Processing {len(all_pdfs)} PDF Document(s)...")
    logger.info("=" * 50)

    for pdf_idx, pdf_file in enumerate(all_pdfs, start=1):
        logger.info(f"\n[{pdf_idx}/{len(all_pdfs)}] Handling PDF: '{pdf_file.name}'")
        res = process_pdf_to_clean_images(
            pdf_path=pdf_file,
            original_name=pdf_file.name,
            output_dir=out_dir,
            dpi_scale=args.dpi_scale,
            min_text_ratio=args.min_text_ratio,
            dark_threshold=args.dark_threshold,
        )
        all_created_images.extend(res)
        processed_files_count += 1

    # Cleanup temp folder
    shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time

    # 4. Final Summary & Report
    print("\n" + "=" * 70)
    print("           DOCUMENT TO CLEAN GROUND TRUTH CONVERSION REPORT")
    print("=" * 70)
    print(f"Total input documents scanned : {total_scanned} ({len(all_pdfs)} PDF, {len(all_docx)} Word)")
    print(f"Total documents processed     : {processed_files_count}")
    print(f"Total clean PNG pages created : {len(all_created_images)}")
    print(f"Output directory              : {out_dir}")
    print(f"Elapsed time                  : {elapsed:.1f} seconds ({elapsed / 60:.2f} mins)")

    if all_created_images:
        print("\nSample Created Images (first 10):")
        for img_path, (w, h), size_b in all_created_images[:10]:
            print(f"  • {img_path.name} | Resolution: {w} x {h} px | Size: {size_b / 1024:.1f} KB")
        if len(all_created_images) > 10:
            print(f"  ... and {len(all_created_images) - 10} more pages.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
