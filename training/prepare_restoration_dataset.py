"""
Prepare the document-restoration dataset (many degraded inputs -> ONE clean target).

Layout (relative to --dataset-root, default ``datasets``)::

    original/<AnhXX>.<ext>                      clean images (never modified)
    synthetic/**/<LoaiLoi>_<AnhXX>_<YY>.png     degraded images (never modified)
    processed/<split>/input/<LoaiLoi>_<AnhXX>_<YY>.png
    processed/<split>/target/<AnhXX>.png        ONE target per original id
    metadata/dataset_manifest.csv

Manifest columns: split, original_id, degradation_type, variant_id, input_path, target_path
(paths are relative to --dataset-root, with forward slashes, so they work on Colab too).

Guarantees:
  * originals / synthetic files are only read, never changed or deleted
  * existing files in processed/ are never overwritten or deleted (conflicts are reported)
  * re-running produces the same result (no duplicates); --dry-run writes nothing
  * split is done by ORIGINAL ID (no leakage); with too few originals everything is 'train'

Usage (PowerShell, from repo root)::

    python training/prepare_restoration_dataset.py --dry-run --resize-target-to-input
    python training/prepare_restoration_dataset.py --resize-target-to-input
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("prepare_restoration_dataset")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".webp", ".tif", ".tiff"}

# canonical degradation name  <-  accepted spellings (compared after lower-casing and
# dropping every non-alphanumeric char). 'brightnesstexurize' covers the existing folder typo.
DEGRADATION_ALIASES: Dict[str, str] = {
    "gaussnoise": "GaussNoise",
    "gaussiannoise": "GaussNoise",
    "brightnesstexturize": "BrightnessTexturize",
    "brightnesstexurize": "BrightnessTexturize",
}

MANIFEST_COLUMNS = ["split", "original_id", "degradation_type", "variant_id", "input_path", "target_path"]

_ID_RE = re.compile(r"^anh0*(\d+)$", re.IGNORECASE)
_FULL_RE = re.compile(r"^(?P<type>[A-Za-z][A-Za-z _\-]*?)_(?P<id>[Aa]nh\d+)_(?P<var>\d+)$")
_NOID_RE = re.compile(r"^(?P<type>[A-Za-z][A-Za-z _\-]*?)_(?P<var>\d+)$")


# --------------------------------------------------------------------------- parsing
def normalize_id(raw: str) -> Optional[str]:
    m = _ID_RE.match(raw.strip())
    return f"Anh{int(m.group(1)):02d}" if m else None


def canonical_degradation(raw: str) -> Optional[str]:
    return DEGRADATION_ALIASES.get(re.sub(r"[^a-z0-9]", "", raw.lower()))


def parse_degraded_name(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int], str]:
    """Return (degradation_type, original_id_or_None, variant_int, error). error=='' when OK."""
    stem = path.stem
    m = _FULL_RE.match(stem)
    if m:
        deg = canonical_degradation(m.group("type"))
        oid = normalize_id(m.group("id"))
        if deg is None:
            return None, None, None, f"unknown degradation type '{m.group('type')}'"
        if oid is None:
            return None, None, None, f"cannot read original id '{m.group('id')}'"
        return deg, oid, int(m.group("var")), ""
    m = _NOID_RE.match(stem)
    if m:
        deg = canonical_degradation(m.group("type"))
        if deg is None:
            return None, None, None, f"unknown degradation type '{m.group('type')}'"
        return deg, None, int(m.group("var")), ""
    return None, None, None, "name does not match <LoaiLoi>_AnhXX_YY"


# --------------------------------------------------------------------------- data classes
@dataclass
class Original:
    oid: str
    path: Path
    size: Tuple[int, int] = (0, 0)  # (w, h)


@dataclass
class Degraded:
    degradation: str
    oid: str
    variant: int
    src: Path
    id_inferred: bool = False
    size: Tuple[int, int] = (0, 0)

    @property
    def name(self) -> str:
        return f"{self.degradation}_{self.oid}_{self.variant:02d}.png"

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.degradation, self.oid, self.variant)


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)


# --------------------------------------------------------------------------- image helpers
def read_rgb(path: Path) -> np.ndarray:
    """Decode fully (raises on corrupt/truncated files). Returns RGB uint8 [H,W,3]."""
    with Image.open(path) as im:
        im.load()
        return np.array(im.convert("RGB"))


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="PNG")


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def alignment_metrics(inp: np.ndarray, tgt: np.ndarray, width: int = 256) -> Tuple[float, float]:
    """Return (shift_px_at_full_res, correlation) between two same-size RGB images."""
    def prep(a: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        scale = width / g.shape[1]
        g = cv2.resize(g, (width, max(1, int(round(g.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        return g, scale

    gi, scale = prep(inp)
    gt, _ = prep(tgt)
    # local-contrast normalise: brightness/texture degradations change global tone
    def norm(g):
        g = g - cv2.GaussianBlur(g, (0, 0), 12)
        return g / (g.std() + 1e-6)
    ni, nt = norm(gi), norm(gt)
    corr = float((ni * nt).mean())
    win = cv2.createHanningWindow((ni.shape[1], ni.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(nt.astype(np.float32), ni.astype(np.float32), win)
    shift = float(np.hypot(dx, dy) / scale)
    return shift, corr


# --------------------------------------------------------------------------- scanning
def scan_originals(root: Path, rep: Report) -> Dict[str, Original]:
    out: Dict[str, Original] = {}
    d = root / "original"
    if not d.is_dir():
        rep.error(f"original directory not found: {d}")
        return out
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        oid = normalize_id(p.stem)
        if oid is None:
            rep.warn(f"ignored original with unexpected name (expected AnhXX): {p.name}")
            continue
        if oid in out:
            rep.error(f"duplicate original id {oid}: {out[oid].path.name} and {p.name}")
            continue
        try:
            h, w = read_rgb(p).shape[:2]
        except Exception as e:
            rep.error(f"corrupt original image {p}: {e}")
            continue
        out[oid] = Original(oid, p, (w, h))
    return out


def scan_degraded(root: Path, originals: Dict[str, Original], rep: Report) -> List[Degraded]:
    d = root / "synthetic"
    if not d.is_dir():
        rep.error(f"synthetic directory not found: {d}")
        return []
    items: Dict[Tuple[str, str, int], Degraded] = {}
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        deg, oid, var, err = parse_degraded_name(p)
        if err and deg is None:
            # maybe the parent folder carries the degradation name
            deg_folder = canonical_degradation(p.parent.name)
            m = re.match(r"^(?:[A-Za-z_\- ]+_)?(?:(?P<id>[Aa]nh\d+)_)?(?P<var>\d+)$", p.stem)
            if deg_folder and m:
                deg, oid, var, err = deg_folder, normalize_id(m.group("id") or "") if m.group("id") else None, int(m.group("var")), ""
        if err:
            rep.error(f"cannot parse degraded file name: {p.relative_to(root)} ({err})")
            continue
        inferred = False
        if oid is None:
            if len(originals) == 1:
                oid = next(iter(originals))
                inferred = True
                rep.warn(f"{p.relative_to(root)}: no original id in name; inferred '{oid}' "
                         f"(only one original present). Rename to {deg}_{oid}_{var:02d}.png to be explicit.")
            else:
                rep.error(f"{p.relative_to(root)}: no original id in name and {len(originals)} originals exist; "
                          f"cannot map to a target")
                continue
        if oid not in originals:
            rep.error(f"{p.relative_to(root)}: missing original image for id {oid}")
            continue
        item = Degraded(deg, oid, var, p, inferred)
        if item.key in items:
            prev = items[item.key]
            if file_sha1(prev.src) == file_sha1(p):
                rep.warn(f"identical duplicate ignored: {p.relative_to(root)} == {prev.src.relative_to(root)}")
            else:
                rep.error(f"duplicate name after normalisation ({item.name}) with DIFFERENT content: "
                          f"{prev.src.relative_to(root)} vs {p.relative_to(root)}")
            continue
        try:
            h, w = read_rgb(p).shape[:2]
        except Exception as e:
            rep.error(f"corrupt degraded image {p.relative_to(root)}: {e}")
            continue
        item.size = (w, h)
        items[item.key] = item
    return sorted(items.values(), key=lambda x: (x.oid, x.degradation, x.variant))


# --------------------------------------------------------------------------- split
def assign_splits(ids: List[str], mode: str, ratios: Tuple[float, float, float], seed: int,
                  min_ids: int, rep: Report) -> Dict[str, str]:
    ids = sorted(ids)
    if mode == "train-only" or (mode == "auto" and len(ids) < min_ids):
        if mode == "auto":
            rep.warn(f"only {len(ids)} original id(s) (< {min_ids}): everything goes to 'train' "
                     f"(smoke/overfit test only, NO validation/test).")
        return {i: "train" for i in ids}
    n = len(ids)
    if n < 3:
        rep.error(f"by-id split needs >= 3 original ids, found {n}")
        return {i: "train" for i in ids}
    _, val_r, test_r = ratios
    n_test = max(1, round(n * test_r))
    n_val = max(1, round(n * val_r))
    while n - n_val - n_test < 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        else:
            n_val -= 1
    # stable order: depends only on (seed, id), not on how many ids exist
    ordered = sorted(ids, key=lambda i: hashlib.sha1(f"{seed}:{i}".encode()).hexdigest())
    out = {}
    for k, i in enumerate(ordered):
        out[i] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return out


# --------------------------------------------------------------------------- main logic
def prepare(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).resolve()
    rep = Report()
    dry = args.dry_run
    logger.info(f"dataset root: {root} | {'DRY-RUN (nothing is written)' if dry else 'WRITE mode'}")

    originals = scan_originals(root, rep)
    degraded = scan_degraded(root, originals, rep)

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        rep.error(f"split ratios must sum to 1.0, got {sum(ratios):.3f}")
        return 1
    by_id: Dict[str, List[Degraded]] = defaultdict(list)
    for d in degraded:
        by_id[d.oid].append(d)

    for oid in sorted(originals):
        if oid not in by_id:
            rep.warn(f"original {oid} has no degraded images (skipped)")

    splits = assign_splits(sorted(by_id), args.split_mode, ratios, args.seed, args.min_ids_for_split, rep)

    # ---- plan per id: target image + validated inputs
    manifest_rows: List[Dict[str, str]] = []
    expected_files: set = set()
    actions = Counter()
    align_stats: List[Tuple[str, float, float]] = []

    for oid in sorted(by_id):
        split = splits[oid]
        orig = originals[oid]
        items = by_id[oid]
        orig_rgb = read_rgb(orig.path)

        sizes = Counter(i.size for i in items)
        ref_size = sizes.most_common(1)[0][0]  # most common input size
        tgt_rgb = orig_rgb
        tgt_resized = False
        if orig.size != ref_size:
            ow, oh = orig.size
            rw, rh = ref_size
            aspect_diff = abs(ow / oh - rw / rh) / (rw / rh)
            if args.resize_target_to_input and aspect_diff <= args.aspect_tol:
                tgt_rgb = cv2.resize(orig_rgb, ref_size, interpolation=cv2.INTER_LANCZOS4)
                tgt_resized = True
                rep.warn(f"{oid}: original {ow}x{oh} resized to {rw}x{rh} for target "
                         f"(aspect diff {aspect_diff * 100:.2f}%); original file untouched.")
            else:
                why = ("pass --resize-target-to-input to align" if aspect_diff <= args.aspect_tol
                       else f"aspect ratio differs by {aspect_diff * 100:.1f}%")
                rep.error(f"{oid}: target size {ow}x{oh} != input size {rw}x{rh} ({why}); "
                          f"inputs of {oid} skipped")
                continue

        valid: List[Degraded] = []
        for it in items:
            if it.size != ref_size:
                rep.error(f"{it.src.name}: size {it.size[0]}x{it.size[1]} != expected {ref_size[0]}x{ref_size[1]}")
                continue
            shift, corr = alignment_metrics(read_rgb(it.src), tgt_rgb)
            align_stats.append((it.name, shift, corr))
            if shift > args.max_shift or corr < args.min_corr:
                rep.error(f"{it.name}: poor alignment with target (shift={shift:.1f}px, corr={corr:.2f})")
                continue
            valid.append(it)
        if not valid:
            continue

        # ---- target (exactly one file per id)
        tgt_rel = f"processed/{split}/target/{oid}.png"
        tgt_dst = root / tgt_rel
        expected_files.add(tgt_dst.resolve())
        if tgt_dst.exists():
            try:
                same = np.array_equal(read_rgb(tgt_dst), tgt_rgb)
            except Exception:
                same = False
            if same:
                actions["target_unchanged"] += 1
            else:
                rep.error(f"{tgt_rel} exists with DIFFERENT content; not overwritten. "
                          f"Skipping all inputs of {oid}.")
                continue
        else:
            actions["target_new"] += 1
            if not dry:
                write_png(tgt_dst, tgt_rgb)

        # ---- inputs
        for it in valid:
            in_rel = f"processed/{split}/input/{it.name}"
            in_dst = root / in_rel
            expected_files.add(in_dst.resolve())
            src_rgb = None
            if in_dst.exists():
                try:
                    src_rgb = read_rgb(it.src)
                    same = np.array_equal(read_rgb(in_dst), src_rgb)
                except Exception:
                    same = False
                if not same:
                    rep.error(f"{in_rel} exists with DIFFERENT content; not overwritten (entry skipped)")
                    continue
                actions["input_unchanged"] += 1
            else:
                actions["input_new"] += 1
                if not dry:
                    if it.src.suffix.lower() == ".png":
                        in_dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(it.src, in_dst)
                    else:
                        write_png(in_dst, read_rgb(it.src))
            manifest_rows.append({
                "split": split, "original_id": oid, "degradation_type": it.degradation,
                "variant_id": f"{it.variant:02d}", "input_path": in_rel, "target_path": tgt_rel,
            })

    # ---- stale / untracked files in processed (never deleted)
    proc = root / "processed"
    if proc.is_dir():
        for p in sorted(proc.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.resolve() not in expected_files:
                rep.warn(f"untracked file in processed/ (not in manifest, NOT deleted): {p.relative_to(root)}")

    manifest_rows.sort(key=lambda r: (r["split"], r["original_id"], r["degradation_type"], r["variant_id"]))

    # ---- write manifest + report
    manifest_path = root / "metadata" / "dataset_manifest.csv"
    report_path = root / "metadata" / "dataset_report.json"
    summary = build_summary(manifest_rows, originals, rep, actions, align_stats)
    if not dry:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            w.writeheader()
            w.writerows(manifest_rows)
        os.replace(tmp, manifest_path)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print_summary(summary, dry, manifest_path)
    return 1 if rep.errors else 0


def build_summary(rows, originals, rep: Report, actions, align_stats) -> dict:
    per_split = defaultdict(lambda: {"inputs": 0, "targets": set(), "originals": set()})
    per_type = Counter()
    for r in rows:
        s = per_split[r["split"]]
        s["inputs"] += 1
        s["targets"].add(r["target_path"])
        s["originals"].add(r["original_id"])
        per_type[r["degradation_type"]] += 1
    return {
        "originals_found": sorted(originals),
        "total_inputs": len(rows),
        "unique_targets": len({r["target_path"] for r in rows}),
        "inputs_per_degradation": dict(per_type),
        "per_split": {k: {"inputs": v["inputs"], "unique_targets": len(v["targets"]),
                          "original_ids": sorted(v["originals"])} for k, v in per_split.items()},
        "actions": dict(actions),
        "alignment": [{"input": n, "shift_px": round(s, 2), "corr": round(c, 3)} for n, s, c in align_stats],
        "errors": rep.errors,
        "warnings": rep.warnings,
    }


def print_summary(s: dict, dry: bool, manifest_path: Path) -> None:
    print("\n==== SUMMARY ====" + (" (dry-run)" if dry else ""))
    print(f"originals found      : {s['originals_found']}")
    print(f"total inputs         : {s['total_inputs']}")
    print(f"unique targets       : {s['unique_targets']}")
    for k, v in sorted(s["inputs_per_degradation"].items()):
        print(f"  {k:<22}: {v}")
    for k, v in sorted(s["per_split"].items()):
        print(f"split {k:<5}: {v['inputs']} inputs -> {v['unique_targets']} target(s) {v['original_ids']}")
    print(f"actions              : {s['actions']}")
    if s["alignment"]:
        sh = max(a["shift_px"] for a in s["alignment"])
        co = min(a["corr"] for a in s["alignment"])
        print(f"alignment            : max shift {sh}px, min corr {co}")
    print(f"warnings: {len(s['warnings'])} | errors: {len(s['errors'])}")
    if not dry:
        print(f"manifest             : {manifest_path}")


def get_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare restoration dataset (many inputs -> one target)")
    p.add_argument("--dataset-root", default="datasets")
    p.add_argument("--dry-run", action="store_true", help="validate and report only; write nothing")
    p.add_argument("--resize-target-to-input", action="store_true",
                   help="if the original differs in size from its degraded images (same aspect ratio), "
                        "write the processed TARGET resized to the input size (original untouched)")
    p.add_argument("--sync-processed", action="store_true",
                   help="scan datasets/processed/{train,val,test} directly and regenerate all metadata CSVs (train, val, test, manifest)")
    p.add_argument("--aspect-tol", type=float, default=0.01, help="max relative aspect-ratio difference")
    p.add_argument("--split-mode", choices=["auto", "train-only", "by-id"], default="auto")
    p.add_argument("--min-ids-for-split", type=int, default=5,
                   help="auto mode: below this many original ids everything stays in train")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-shift", type=float, default=4.0, help="max allowed misalignment in pixels")
    p.add_argument("--min-corr", type=float, default=0.3, help="min structural correlation input vs target")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    if getattr(args, "sync_processed", False):
        try:
            from training.generate_metadata import generate_metadata
        except ImportError:
            from generate_metadata import generate_metadata
        generate_metadata()
        sys.exit(0)
    sys.exit(prepare(args))
