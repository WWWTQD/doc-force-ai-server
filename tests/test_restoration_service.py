import os
import cv2
import numpy as np
import pytest
from app.services.restoration_service import RestorationService
from app.core.config import settings


def test_restoration_service_singleton():
    service1 = RestorationService.get_instance()
    service2 = RestorationService.get_instance()
    assert service1 is service2


def test_restoration_service_load_and_infer_bgr():
    service = RestorationService.get_instance()
    service.initialize()

    # Create synthetic degraded test image (gray background, dark text simulation)
    test_img = np.full((300, 400, 3), 180, dtype=np.uint8)
    cv2.putText(test_img, "DOC-FORCE AI TEST", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    # Add a red seal simulation
    cv2.circle(test_img, (320, 150), 30, (0, 0, 220), -1)

    restored_bgr = service.restore_image_bgr(
        test_img,
        tile_size=256,
        overlap=32,
        preserve_color=True,
    )

    assert restored_bgr is not None
    assert restored_bgr.shape == test_img.shape
    assert restored_bgr.dtype == np.uint8

    # Background should be significantly lighter/cleaned (approaching white >= 220)
    bg_pixel = restored_bgr[20, 20]
    assert np.mean(bg_pixel) >= 180


def test_restoration_service_bytes():
    service = RestorationService.get_instance()
    service.initialize()

    # Create image and encode to bytes
    test_img = np.full((200, 200, 3), 160, dtype=np.uint8)
    cv2.putText(test_img, "RESTORE", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, encoded = cv2.imencode(".jpg", test_img)
    raw_bytes = encoded.tobytes()

    restored_bytes = service.restore_image_bytes(raw_bytes, tile_size=256, overlap=32)
    assert restored_bytes is not None
    assert len(restored_bytes) > 0

    # Verify decoded restored bytes
    nparr = np.frombuffer(restored_bytes, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (200, 200, 3)


def test_restoration_fallback_on_corrupt_data():
    service = RestorationService.get_instance()
    # Sending corrupt bytes should safely return original bytes without crashing
    corrupt_bytes = b"not_an_image_data_at_all"
    res = service.restore_image_bytes(corrupt_bytes)
    assert res == corrupt_bytes
