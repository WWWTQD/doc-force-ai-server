import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "worker")))

from document_scanner import DocumentScanner, ScanGeometry


def _dummy_portrait() -> np.ndarray:
    return np.full((800, 600, 3), 240, dtype=np.uint8)


def test_orient_prior_bias_keeps_zero_without_gate():
    # Tầng 1 lệch nhẹ 180° (0.25 vs 0.303 như real_23 cũ) -> không được lật ngược 180°
    scanner = DocumentScanner(
        orientation_scorer=lambda im: 0.303 if im[0, 0, 0] == 240 else 0.25,
        axis_detector=lambda _im: "horizontal",
    )
    rot, verified = scanner.orient_with_verification(_dummy_portrait())
    assert rot == 0
    assert verified is False


def test_orient_tier2_gate_rejects_false_180_and_verifies_true_180():
    img = _dummy_portrait()

    # Giả lập Tầng 1 thiên vị 180° (0: 0.25, 180: 0.31), Tầng 2 xác nhận 0° đọc đúng -> giữ 0°
    scanner_keep = DocumentScanner(
        orientation_scorer=lambda _im: 0.25,
        orientation_gate=lambda _im, a, b, _sc: (0, True),
        axis_detector=lambda _im: "horizontal",
    )
    rot_keep, ver_keep = scanner_keep.orient_with_verification(img)
    assert rot_keep == 0
    assert ver_keep is True

    # Giả lập ảnh thực sự ngược 180° nhưng Tầng 1 phân vân (0: 0.42, 180: 0.35) -> Tầng 2 bắt buộc chạy và xoay 180°
    called = []

    def _gate(_im, a, b, sc):
        called.append((a, b, sc))
        return 180, True

    scanner_flip = DocumentScanner(
        orientation_scorer=lambda _im: 0.38,
        orientation_gate=_gate,
        axis_detector=lambda _im: "horizontal",
    )
    rot_flip, ver_flip = scanner_flip.orient_with_verification(img)
    assert rot_flip == 180
    assert ver_flip is True
    assert len(called) == 1


def test_orient_vertical_uses_gate_and_metadata_records_verified():
    img = np.full((600, 800, 3), 240, dtype=np.uint8)
    scanner = DocumentScanner(
        orientation_scorer=lambda _im: 0.40,
        orientation_gate=lambda _im, a, b, _sc: (270, True),
        axis_detector=lambda _im: "vertical",
    )
    rot, ver = scanner.orient_with_verification(img)
    assert rot == 270
    assert ver is True

    geom = ScanGeometry(
        image=img,
        transform=np.eye(3),
        original_size=(800, 600),
        rotation=rot,
        orientation_verified=ver,
    )
    meta = geom.to_metadata()
    assert meta["rotation"] == 270
    assert meta["orientationVerified"] is True
