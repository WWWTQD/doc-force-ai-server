"""Unit tests for worker.vietnamese_syllable_validator."""

from __future__ import annotations

import sys

from worker.vietnamese_syllable_validator import invalid_syllables


def test_golden_cases_1_to_5_and_8_recall() -> None:
    cases = [
        (
            "Thi hn np đon đăng ký mua tài sán công něu trên, k t ngày 06/01/2021",
            ["hn", "np", "něu", "k", "t"],
        ),
        (
            "QLTT ngày 06/01/2021 cùa Đôi QLTT s 4, vè vic xin ý kién phé duyět phuong",
            ["s", "vic", "kién", "duyět", "phuong"],
        ),
        (
            "tang vt vi phąm hành chinh tjch thu sung quý nhà nuóc theo Công văn s: 20/CV-",
            ["vt", "phąm", "tjch", "nuóc", "s"],
        ),
        (
            "Hài Phòng. ngày th tháng 01 năm 2021",
            ["th"],
        ),
        (
            "ngly .. thíng 01 nim 2021",
            ["ngly", "thíng"],
        ),
        (
            "tang vật tjch thu nhà nuóc",
            ["tjch", "nuóc"],
        ),
    ]
    for line, expected_tokens in cases:
        spans = invalid_syllables(line)
        got = [s.token for s in spans]
        for tok in expected_tokens:
            assert tok in got, f"Expected {tok!r} flagged in {line!r}, got {got!r}"
        for sp in spans:
            assert line[sp.start : sp.end] == sp.token


def test_clean_vietnamese_sentences_zero_false_positives() -> None:
    clean_sentences = [
        "Ủy ban nhân dân thành phố ban hành quyết định mới.",
        "Số: 123/QĐ-UBND ngày 13-08-1945 về công tác tổ chức.",
        "Hợp đồng số 14/2024/HĐ-DOCFORCE có hiệu lực kể từ ngày ký.",
        "Mọi liên hệ gửi về địa chỉ email a@doc-forge.vn hoặc phòng hành chính.",
        "Thời hạn nộp đơn đăng ký mua tài sản công nêu trên, kể từ ngày 06/01/2021.",
        "Đội quản lý thị trường số 4 báo cáo kết quả kiểm tra định kỳ.",
        "Tang vật vi phạm hành chính bị tịch thu sung quỹ nhà nước.",
        "Kính gửi các đơn vị trực thuộc sở khoa học và công nghệ.",
        "Người đại diện theo pháp luật chịu trách nhiệm trước cơ quan nhà nước.",
        "Gìn giữ bản sắc văn hóa dân tộc là trách nhiệm của mỗi công dân.",
        "Giếng nước đầu làng gắn liền với ký ức tuổi thơ của nhiều người.",
        "Giải pháp chuyển đổi số giúp rút ngắn quy trình xử lý hồ sơ.",
        "Quyết định phê duyệt dự án đầu tư xây dựng hạ tầng kỹ thuật.",
        "Quý khách vui lòng kiểm tra kỹ thông tin cá nhân trước khi xác nhận.",
        "Quỳnh Anh và Khuỷu tay là hai ví dụ về vần khó trong tiếng Việt.",
        "Thuở xưa, người dân thường dùng thuyền gỗ để vượt sông.",
        "Nghiêng mình trước tượng đài liệt sĩ vào sáng ngày giỗ tổ.",
        "Nghệ thuật truyền thống cần được bảo tồn và phát huy rộng rãi.",
        "Kiên quyết đấu tranh phòng chống tham nhũng, lãng phí và tiêu cực.",
        "Chính quyền địa phương tổ chức hội nghị tổng kết cuối năm.",
        "Mức kinh phí hỗ trợ dự kiến khoảng 50.000.000 đồng cho mỗi xã.",
        "Các mục chính gồm: a. Giới thiệu; b) Nội dung; c. Kết luận; d) Kiến nghị.",
        "Chiều dài tuyến đường đạt 12 km, tải trọng thiết kế 30 tấn, v.v.",
        "Bà Nguyễn Thị Lan giữ chức vụ trưởng phòng tài chính kế toán.",
        "Hệ thống sử dụng API REST với Python và Java để xử lý văn bản.",
        "Dịch vụ backend và frontend trao đổi dữ liệu qua chuẩn JSON.",
        "Mô hình OCR trích xuất văn bản tiếng Việt đạt độ tin cậy cao.",
        "Giấy chứng nhận quyền sử dụng đất được cấp theo đúng thẩm quyền.",
        "Ghi chú quan trọng: không sửa đổi chữ số trong văn bản hành chính.",
        "Khu vực hành lang an toàn giao thông phải được giải tỏa dứt điểm.",
        "Sáng kiến cải tiến kỹ thuật góp phần nâng cao năng suất lao động.",
        "Việc lưu trữ hồ sơ điện tử tuân thủ nghiêm ngặt quy định bảo mật.",
    ]
    assert len(clean_sentences) >= 30
    for idx, sent in enumerate(clean_sentences):
        spans = invalid_syllables(sent)
        assert spans == [], f"Unexpected FP at sentence #{idx}: {sent!r} -> {spans!r}"


def test_b5_exemptions_and_reason_codes() -> None:
    assert invalid_syllables("Truy cập https://doc-forge.vn:8080/tai-lieu ngay hôm nay.") == []
    assert invalid_syllables("Xem tt và stt trong bảng tổng hợp số liệu.") == []
    assert invalid_syllables("Tài liệu đối chiếu Tiếng Việt và 中文 日本語 한국어 Москва.") == []
    assert invalid_syllables("Chiến thắng lịch sử ngày 30/04/1975.") == []

    # Invalid stop-final tone, multi-tone, orthography, bound nucleus
    assert [s.reason for s in invalid_syllables("viêc")] == ["STOP_FINAL_TONE"]
    assert [s.reason for s in invalid_syllables("a\u0301\u0300")] == ["MULTI_TONE"]
    assert [s.reason for s in invalid_syllables("ge")] == ["ORTHO_CONSONANT"]
    assert [s.reason for s in invalid_syllables("biê")] == ["BOUND_NUCLEUS"]

    # Ensure heavy libraries were never imported
    for mod in ("torch", "paddle", "paddleocr", "vietocr", "ultralytics"):
        assert mod not in sys.modules
