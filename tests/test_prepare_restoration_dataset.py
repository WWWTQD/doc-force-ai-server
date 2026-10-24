"""Tests for training/prepare_restoration_dataset.py and manifest loading."""
import csv
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training import prepare_restoration_dataset as prep  # noqa: E402
from training.restoration_dataset import DocumentRestorationDataset  # noqa: E402


def _doc(seed, size=(96, 128)):
    rng = np.random.RandomState(seed)
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    for _ in range(25):  # random dark blocks = "text"
        x, y = rng.randint(0, size[0] - 20), rng.randint(0, size[1] - 6)
        img[y:y + 4, x:x + rng.randint(8, 20)] = 20
    return img


def _save(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _noisy(arr, seed):
    n = np.random.RandomState(seed).normal(0, 8, arr.shape)
    return np.clip(arr + n, 0, 255).astype(np.uint8)


def _build(root, ids):
    for k, oid in enumerate(ids):
        clean = _doc(k)
        _save(clean, root / "original" / f"{oid}.png")
        for v in (1, 2):
            _save(_noisy(clean, v), root / "synthetic" / "gaussian_noise" / f"GaussNoise_{oid}_{v:02d}.png")
        _save(np.clip(clean * 0.8, 0, 255).astype(np.uint8),
              root / "synthetic" / "bt" / f"BrightnessTexturize_{oid}_01.png")


def _run(root, *extra):
    return prep.prepare(prep.get_args(["--dataset-root", str(root), *extra]))


def _rows(root):
    with open(root / "metadata" / "dataset_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_name_parsing():
    from pathlib import Path
    assert prep.parse_degraded_name(Path("GaussNoise_Anh01_03.png")) == ("GaussNoise", "Anh01", 3, "")
    assert prep.parse_degraded_name(Path("brightness_texturize_Anh2_1.png"))[:3] == ("BrightnessTexturize", "Anh02", 1)
    assert prep.parse_degraded_name(Path("BrightnessTexturize_05.png"))[:3] == ("BrightnessTexturize", None, 5)
    assert prep.parse_degraded_name(Path("foo.png"))[3] != ""


def test_single_id_train_only_one_target_and_idempotent(tmp_path):
    _build(tmp_path, ["Anh01"])
    assert _run(tmp_path, "--dry-run") == 0
    assert not (tmp_path / "metadata").exists()  # dry-run writes nothing
    assert _run(tmp_path) == 0
    rows = _rows(tmp_path)
    assert len(rows) == 3 and {r["split"] for r in rows} == {"train"}
    assert {r["target_path"] for r in rows} == {"processed/train/target/Anh01.png"}
    assert len(list((tmp_path / "processed" / "train" / "target").iterdir())) == 1
    before = sorted(p.name for p in (tmp_path / "processed").rglob("*.png"))
    assert _run(tmp_path) == 0
    assert before == sorted(p.name for p in (tmp_path / "processed").rglob("*.png"))
    assert len(_rows(tmp_path)) == 3


def test_new_ids_matched_and_split_by_id_without_leakage(tmp_path):
    ids = [f"Anh{i:02d}" for i in range(1, 8)]
    _build(tmp_path, ids)
    assert _run(tmp_path) == 0
    rows = _rows(tmp_path)
    assert len(rows) == 7 * 3
    for r in rows:  # every input maps to the target of ITS OWN original id
        assert r["input_path"].split("_")[-2].split("/")[-1] == r["original_id"] or r["original_id"] in r["input_path"]
        assert r["target_path"].endswith(f"/{r['original_id']}.png")
    split_of = {}
    for r in rows:
        assert split_of.setdefault(r["original_id"], r["split"]) == r["split"]
    assert {"train", "val", "test"} <= set(split_of.values())


def test_missing_original_and_corrupt_and_size_mismatch(tmp_path):
    _build(tmp_path, ["Anh01"])
    _save(_doc(5), tmp_path / "synthetic" / "gaussian_noise" / "GaussNoise_Anh09_01.png")  # no original
    (tmp_path / "synthetic" / "gaussian_noise" / "GaussNoise_Anh01_07.png").write_bytes(b"not an image")
    _save(_doc(1, (64, 64)), tmp_path / "synthetic" / "gaussian_noise" / "GaussNoise_Anh01_08.png")
    assert _run(tmp_path) == 1
    errs = " | ".join(prep.json.loads((tmp_path / "metadata" / "dataset_report.json").read_text())["errors"])
    assert "Anh09" in errs and "corrupt" in errs and "GaussNoise_Anh01_08" in errs
    assert len(_rows(tmp_path)) == 3  # valid ones still processed


def test_loader_uses_manifest_not_filenames_and_shares_target(tmp_path):
    _build(tmp_path, ["Anh01"])
    _run(tmp_path)
    ds = DocumentRestorationDataset(str(tmp_path / "metadata" / "dataset_manifest.csv"),
                                    image_size=(64, 64), split="train", return_metadata=True)
    assert len(ds) == 3 and len(DocumentRestorationDataset(
        str(tmp_path / "metadata" / "dataset_manifest.csv"), split="val")) == 0
    x, y, meta = ds[0]
    assert x.shape == y.shape == (1, 64, 64)
    assert os.path.basename(meta["target"]) == "Anh01.png" != os.path.basename(meta["input"])
    for i in range(len(ds)):
        ds[i]
    assert len(ds._target_cache) == 1
