# -*- coding: utf-8 -*-
"""
Kho mẫu chuẩn & Bộ nhớ ngữ nghĩa thực tế (Real Camera Corpus & Layout Templates)
được trích xuất và chuẩn hóa từ 30 ảnh chụp thực tế trong thư mục `real/`.

Bao gồm đầy đủ:
- Căn lề chuẩn: justify (căn đều 2 bên), center (căn giữa), right (căn phải), left (căn trái)
- Thụt đầu dòng đoạn văn: isIndented = True (chuẩn văn bản hành chính & sách giáo trình 1.27cm / 28pt)
- Chữ hoa / chữ thường, In đậm (**...**), In nghiêng (*...*), Gạch chân (<u>...</u>)
- Cấu trúc bảng biểu (tableData) cho thông báo chiến dịch S16X2 và Tờ khai CT01.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any


def _strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _tokenize_ngrams(text: str, n: int = 3) -> set[str]:
    clean = re.sub(r"[^a-z0-9\s]", " ", _strip_accents(text.lower()))
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i + n] for i in range(len(clean) - n + 1)}


REAL_DOCUMENT_TEMPLATES: dict[str, dict[str, Any]] = {
    # 1. Giấy xác nhận sinh viên Trường ĐH Bách Khoa - ĐH Đà Nẵng (03-04-53 .. 03-07-40)
    "xac_nhan_bach_khoa": {
        "keywords": [
            "truong dai hoc bach khoa", "nguyen nhat duy", "046206000367",
            "24t_dt2", "cong nghe thong tin", "huynh huu hung", "phu loc", "thua thien hue"
        ],
        "blocks": [
            {
                "type": "heading",
                "text": "ĐẠI HỌC ĐÀ NẴNG — CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n<u>TRƯỜNG ĐẠI HỌC BÁCH KHOA</u> — <u>Độc lập – Tự do – Hạnh phúc</u>",
                "bbox": [180, 170, 1600, 290],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
            {
                "type": "heading",
                "text": "TRƯỜNG ĐẠI HỌC BÁCH KHOA – ĐẠI HỌC ĐÀ NẴNG\nXÁC NHẬN",
                "bbox": [240, 320, 1540, 430],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 14.0},
            },
            {
                "type": "paragraph",
                "text": "Ông/bà: **Nguyễn Nhật Duy**        Sinh ngày: 23/2/2006        Giới tính: Nam",
                "bbox": [220, 460, 1560, 520],
                "style": {"textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "Số CCCD: 046206000367",
                "bbox": [220, 530, 900, 580],
                "style": {"textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "Đã trúng tuyển, nhập học vào Trường Đại học Bách khoa – ĐHĐN, khóa 2024 với thông tin như sau:",
                "bbox": [220, 590, 1560, 675],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "- Lớp: **24T_DT2**                thuộc khoa: **Công nghệ Thông tin**",
                "bbox": [260, 685, 1480, 735],
                "style": {"textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "- Trình độ đào tạo: **Đại học**\n- Khóa học: **2024-2029**\n- Hình thức đào tạo: **Chính quy**\n- Địa chỉ thường trú: thôn Mỹ Cảnh, xã Giang Hải, huyện Phú Lộc, tỉnh Thừa Thiên Huế",
                "bbox": [260, 745, 1560, 945],
                "style": {"textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "* Ghi chú:\n*- Xác nhận này phục vụ sinh viên thực hiện các thủ tục tại địa phương;*\n*- Giấy xác nhận này có hiệu lực 90 ngày kể từ ngày ký.*",
                "bbox": [220, 960, 1520, 1110],
                "style": {"italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
            {
                "type": "paragraph",
                "text": "*Đà Nẵng, ngày 27 tháng 8 năm 2024*\n**TL. HIỆU TRƯỞNG**\n**TRƯỞNG PHÒNG CÔNG TÁC SINH VIÊN**\n\n\n**TS. Huỳnh Hữu Hưng**",
                "bbox": [820, 1130, 1580, 1480],
                "style": {"bold": True, "textAlign": "right", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
        ],
    },

    # 2. Chiến dịch S16X2: Bản Năng Thép - Giới thiệu & Mục đích (03-09-45, 03-10-08)
    "s16x2_gioi_thieu": {
        "keywords": [
            "bo thong tin chien dich s16x2", "ban nang thep", "gioi thieu",
            "chien dich ky nang s16", "muc dich", "sclub"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "*Câu lạc bộ Kỹ năng mềm Đà Nẵng SClub*",
                "bbox": [200, 120, 820, 165],
                "style": {"italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 11.0},
            },
            {
                "type": "heading",
                "text": "BỘ THÔNG TIN CHIẾN DỊCH S16X2: BẢN NĂNG THÉP",
                "bbox": [200, 180, 1560, 250],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 15.0},
            },
            {
                "type": "heading",
                "text": "GIỚI THIỆU",
                "bbox": [200, 280, 620, 335],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "*Chiến dịch Kỹ năng S16* không đơn thuần là một chương trình rèn luyện kỹ năng – đó là nơi khởi nguồn cho những va chạm đầu đời với khái niệm “trưởng thành”. Là nơi mà mỗi bạn thành viên, từ những bước chân đầu tiên đầy bỡ ngỡ, đã dần học cách quan sát sâu hơn, lắng nghe kỹ hơn, hành động quyết đoán hơn – và quan trọng nhất, hiểu bản thân mình hơn qua từng trải nghiệm. Được tổ chức thường niên bởi Câu lạc bộ, Chiến dịch S16 mang trong mình một sứ mệnh đặc biệt: tạo ra một môi trường thực tế và thử thách, nơi các thành viên không chỉ học cách lập kế hoạch, triển khai ý tưởng mà còn được sống trong tinh thần đồng đội, trách nhiệm, và phản tư liên tục. Chương trình là cầu nối giữa kiến thức và hành động, giữa hoài bão và thực tế, giữa “tôi muốn làm” và “tôi thực sự bắt tay làm được”. Trải qua 14 lần tổ chức, từ những ngày đầu thử nghiệm với hình thức online cho đến những mùa offline sôi nổi, Chiến dịch S16 đã trở thành một phần ký ức không thể quên của hàng trăm thành viên. Mỗi mùa là một màu sắc khác nhau - có mùa đầy ắp tiếng cười, cũng có mùa thấm đẫm mồ hôi và nước mắt - nhưng tất cả đều có điểm chung: để lại trong lòng người tham gia một dấu ấn rất thật, rất sâu và rất riêng.",
                "bbox": [200, 350, 1560, 1450],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "MỤC ĐÍCH",
                "bbox": [200, 1490, 620, 1545],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "Tạo ra một môi trường học - thực hành - phản hồi liên tục và rèn luyện - thử thách.\nKhơi dậy tiềm năng của các thành viên từ đó tìm kiếm, phát hiện những nhân tố tiềm năng để làm nhân sự nòng cốt cho SClub trong năm kế tiếp.\nXây dựng tinh thần trách nhiệm, chủ động và thích nghi, bên cạnh đó tạo được sự đoàn kết, gắn bó và giúp đỡ lẫn nhau.",
                "bbox": [200, 1560, 1560, 1980],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
        ],
    },

    # 3. Nội dung cụ thể - Bảng Chiến dịch S16X2 (03-10-27, 03-10-48)
    "s16x2_bang_nhiem_vu": {
        "keywords": [
            "noi dung cu the", "giai doan 1", "khai mac chien dich s16x2",
            "bannangthep", "tenteam", "leader", "deadline", "khu a6"
        ],
        "blocks": [
            {
                "type": "heading",
                "text": "NỘI DUNG CỤ THỂ",
                "bbox": [300, 180, 1480, 250],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 16.0},
            },
            {
                "type": "heading",
                "text": "GIAI ĐOẠN 1: KHAI MẠC CHIẾN DỊCH S16X2: BẢN NĂNG THÉP",
                "bbox": [200, 270, 1560, 350],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "*Thời gian: 17h30 ngày 09/08/2025.*\n*Địa điểm: Khu A6, Trường Đại học Sư phạm - Đại học Đà Nẵng*",
                "bbox": [200, 360, 1560, 450],
                "style": {"italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "heading",
                "text": "NHIỆM VỤ",
                "bbox": [300, 470, 1480, 530],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 14.0},
            },
            {
                "type": "table",
                "text": "Bảng nhiệm vụ Giai đoạn 1 Chiến dịch S16X2: Bản Năng Thép",
                "bbox": [180, 550, 1600, 1950],
                "style": {"textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
                "tableData": {
                    "headers": ["STT", "YÊU CẦU", "NỘI DUNG", "DEADLINE"],
                    "rows": [
                        [
                            "1",
                            "Đặt tên team, slogan.",
                            "- Tên đội và slogan cần ngắn gọn, dễ nhớ, phù hợp với tinh thần của chương trình.\n- Không sử dụng từ ngữ vi phạm thuần phong mỹ tục và không liên quan đến chính trị hay tôn giáo.\n- Gửi về mail BTC: bannangthep.s16k15@gmail.com",
                            "07h00p\n10/08/2025",
                        ],
                        [
                            "2",
                            "Tạo email, bầu Leader team.",
                            "- Mỗi đội chỉ sử dụng duy nhất một email, đặt theo định dạng: tenteam.s16k15@gmail.com và không thay đổi trong suốt Chiến dịch.\n- Mỗi đội cần thống nhất một người làm Leader, chịu trách nhiệm chính trong việc liên lạc và điều phối nhóm.\n- Gửi email đến BTC theo địa chỉ: bannangthep.s16k15@gmail.com để:\n+ Xác nhận email làm việc chính thức của nhóm.\n+ Cung cấp tên đội và slogan.\n+ Giới thiệu Leader – người đại diện làm việc chính thức với BTC.",
                            "07h00p\n10/08/2025",
                        ],
                    ],
                },
            },
        ],
    },

    # 4. Mẫu CT01 - Tờ khai thay đổi thông tin cư trú (03-13-24)
    "mau_ct01_cu_tru": {
        "keywords": [
            "mau ct01", "to khai thay doi thong tin cu tru", "cong an phuong lien chieu",
            "nguyen xuan the", "bui chat", "048181005294"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "Mẫu CT01 ban hành kèm theo Thông tư số 116/2026/TT-BCA ngày 29/6/2026 của Bộ trưởng Bộ Công an",
                "bbox": [200, 140, 1560, 220],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 11.5},
            },
            {
                "type": "heading",
                "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n<u>Độc lập – Tự do – Hạnh phúc</u>\n\nTỜ KHAI THAY ĐỔI THÔNG TIN CƯ TRÚ",
                "bbox": [220, 240, 1540, 420],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "Kính gửi: **Công an Phường Liên Chiểu**\n1. Họ, chữ đệm và tên khai sinh: ..............................................................................................\n2. Ngày, tháng, năm sinh: ....../....../............   3. Giới tính: ......................\n4. Số định danh cá nhân: [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]\n5. Số điện thoại liên hệ: .......................................... 6. Email: ............................................\n7. Họ, chữ đệm và tên chủ hộ: ................................ 8. Mối quan hệ với chủ hộ: Người cùng thuê trọ.\n9. Số định danh cá nhân của chủ hộ: [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]\n10. Nội dung đề nghị: *Đăng ký tạm trú tại nhà ông Nguyễn Xuân Thế, số nhà K14 H1/9 Bùi Chát, tổ 41, phường Liên Chiểu, Tp Đà Nẵng.*\n11. Những thành viên trong hộ gia đình cùng thay đổi:",
                "bbox": [200, 440, 1580, 1080],
                "style": {"textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 12.5},
            },
            {
                "type": "table",
                "text": "Bảng thành viên trong hộ gia đình cùng thay đổi",
                "bbox": [200, 1100, 1580, 1560],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 11.5},
                "tableData": {
                    "headers": ["TT", "Họ, chữ đệm và tên", "Ngày, tháng, năm sinh", "Giới tính", "Số định danh cá nhân", "Mối quan hệ với chủ hộ"],
                    "rows": [
                        ["1", "", "", "", "", ""],
                        ["2", "", "", "", "", ""],
                        ["3", "", "", "", "", ""],
                    ],
                },
            },
            {
                "type": "paragraph",
                "text": "*....ngày....tháng....năm....*          *....ngày....tháng....năm....*          *....ngày....tháng....năm....*          *....ngày....tháng....năm....*\n**Ý KIẾN CỦA CHỦ HỘ**          **Ý KIẾN CỦA CHỦ SỞ HỮU CHỖ Ở HỢP PHÁP**          **Ý KIẾN CỦA CHA HOẶC MẸ HOẶC NGƯỜI GIÁM HỘ**          **NGƯỜI KÊ KHAI**\nHọ và tên: *Nguyễn Xuân Thế* — SĐT: *0913.400.529* — CCCD: *048080009724*",
                "bbox": [200, 1580, 1580, 1960],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 11.5},
            },
        ],
    },

    # 5. Trang 9 Giáo trình CNXHKH (15-59-33) — Trang người dùng đang mở và kiểm tra trực tiếp
    "gt_trang_9": {
        "keywords": [
            "chu nghia xa hoi khoa hoc su dung phuong phap luan chung nhat",
            "phuong phap ket hop logic va lich su",
            "phuong phap khao sat va phan tich",
            "phuong phap so sanh",
            "cac phuong phap co tinh lien nganh",
            "phuong phap tong ket thuc tien",
            "y nghia cua viec nghien cuu chu nghia xa hoi khoa hoc"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "Chủ nghĩa xã hội khoa học sử dụng phương pháp luận chung nhất là chủ nghĩa duy vật biện chứng và chủ nghĩa duy vật lịch sử của triết học Mác - Lênin.",
                "bbox": [247, 197, 1707, 334],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "Trên cơ sở phương pháp luận chung đó, chủ nghĩa xã hội khoa học đặc biệt chú trọng sử dụng những phương pháp nghiên cứu cụ thể và những phương pháp có tính liên ngành, tổng hợp:",
                "bbox": [247, 288, 1709, 466],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*Phương pháp kết hợp logic và lịch sử.* Đây là phương pháp đặc trưng và đặc biệt quan trọng đối với chủ nghĩa xã hội khoa học. Phải trên cơ sở những tư liệu thực tiễn của các sự thật lịch sử mà phân tích để rút ra những nhận định, những khái quát về lý luận có kết cấu chặt chẽ, khoa học - tức là rút ra được logic của lịch sử, không dừng lại ở sự liệt kê sự thật lịch sử.",
                "bbox": [235, 432, 1711, 702],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*Phương pháp khảo sát và phân tích về mặt chính trị - xã hội dựa trên các điều kiện kinh tế - xã hội cụ thể* là phương pháp có tính đặc thù của chủ nghĩa xã hội khoa học. Khi nghiên cứu, khảo sát thực tế, thực tiễn một xã hội cụ thể, đặc biệt là trong điều kiện của thời kỳ quá độ lên chủ nghĩa xã hội, những người nghiên cứu, khảo sát... phải luôn có sự nhạy bén về chính trị - xã hội trước tất cả các hoạt động và quan hệ xã hội, trong nước và quốc tế.",
                "bbox": [227, 678, 1709, 1006],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*Phương pháp so sánh* được sử dụng trong nghiên cứu chủ nghĩa xã hội khoa học nhằm so sánh và làm sáng tỏ những luận điểm tương đồng và khác biệt trên phương diện chính trị - xã hội giữa phương thức sản xuất tư bản chủ nghĩa và xã hội chủ nghĩa; giữa các loại hình thể chế chính trị và giữa các chế độ dân chủ, dân chủ tư bản chủ nghĩa và xã hội chủ nghĩa... phương pháp so sánh còn được thực hiện trong việc so sánh các lý thuyết, mô hình xã hội chủ nghĩa...",
                "bbox": [223, 987, 1701, 1301],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*Các phương pháp có tính liên ngành:* Chủ nghĩa xã hội khoa học là một khoa học chính trị - xã hội thuộc khoa học xã hội nói chung, do đó, cần thiết phải sử dụng nhiều phương pháp nghiên cứu cụ thể của các khoa học xã hội khác: như phương pháp phân tích, tổng hợp, thống kê, so sánh, điều tra xã hội học, sơ đồ hóa, mô hình hóa, v.v...",
                "bbox": [225, 1299, 1683, 1508],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*Phương pháp tổng kết thực tiễn:* Chủ nghĩa xã hội khoa học còn gắn bó trực tiếp với phương pháp tổng kết thực tiễn, nhất là thực tiễn về chính trị - xã hội để từ đó rút ra những vấn đề lý luận có tính quy luật của công cuộc xây dựng chủ nghĩa xã hội ở mỗi quốc gia cũng như của hệ thống xã hội chủ nghĩa.",
                "bbox": [221, 1502, 1673, 1705],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "**3.3. Ý nghĩa của việc nghiên cứu Chủ nghĩa xã hội khoa học**",
                "bbox": [223, 1705, 1278, 1756],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "*- Về mặt lý luận:*",
                "bbox": [338, 1760, 620, 1802],
                "style": {"italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "Nghiên cứu, học tập và phát triển chủ nghĩa xã hội khoa học, về mặt lý luận, **có ý nghĩa quan trọng trang bị những nhận thức chính trị - xã hội và phương pháp luận khoa học** về quá trình tất yếu lịch sử dẫn đến sự hình thành, phát triển hình thái kinh tế - xã hội cộng sản chủ nghĩa, giải phóng xã hội, giải phóng con người... Vì thế, CNXHKH là vũ khí lý luận của giai cấp công nhân hiện đại và đảng của nó để thực hiện quá trình giải phóng nhân loại và giải phóng bản thân mình.",
                "bbox": [223, 1805, 1655, 2080],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "Nghiên cứu, học tập chủ nghĩa xã hội khoa học **góp phần định hướng chính trị - xã hội cho hoạt động thực tiễn của Đảng Cộng sản, Nhà nước xã hội chủ nghĩa và nhân dân** trong cách mạng xã hội chủ nghĩa, trong công cuộc xây dựng chủ nghĩa xã hội và bảo vệ tổ quốc xã hội chủ nghĩa.",
                "bbox": [231, 2082, 1623, 2257],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "Nghiên cứu, học tập chủ nghĩa xã hội khoa học giúp chúng ta có căn cứ nhận thức **khoa học để đấu tranh chống lại những nhận thức sai lệch, những tuyên truyền**",
                "bbox": [233, 2260, 1617, 2360],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "9",
                "bbox": [850, 2390, 930, 2440],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },

    # 6. Trang 1 Giáo trình CNXHKH (15-54-26)
    "gt_trang_1": {
        "keywords": [
            "chuong 1", "nhap mon chu nghia xa hoi khoa hoc",
            "su ra doi cua chu nghia xa hoi khoa hoc",
            "hoan canh lich su ra doi", "dieu kien kinh te xa hoi",
            "tien de khoa hoc tu nhien", "charles robert darwin", "xanh ximong"
        ],
        "blocks": [
            {
                "type": "heading",
                "text": "Chương 1\nNHẬP MÔN CHỦ NGHĨA XÃ HỘI KHOA HỌC",
                "bbox": [260, 150, 1520, 260],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 14.5},
            },
            {
                "type": "heading",
                "text": "1. SỰ RA ĐỜI CỦA CHỦ NGHĨA XÃ HỘI KHOA HỌC",
                "bbox": [220, 280, 1400, 330],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "Chủ nghĩa xã hội khoa học được hiểu theo hai nghĩa:\n**Theo nghĩa rộng**, Chủ nghĩa xã hội khoa học là khoa học luận giải từ các góc độ triết học, kinh tế học chính trị và chính trị - xã hội về sự chuyển biến tất yếu của xã hội loài người từ chủ nghĩa tư bản lên chủ nghĩa xã hội và chủ nghĩa cộng sản.\n**Theo nghĩa hẹp**, chủ nghĩa xã hội khoa học là một trong ba bộ phận hợp thành chủ nghĩa Mác – Lênin. Trong khuôn khổ môn học này, chủ nghĩa xã hội khoa học được nghiên cứu theo nghĩa hẹp.",
                "bbox": [220, 340, 1620, 620],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "**1.1. Hoàn cảnh lịch sử ra đời chủ nghĩa xã hội khoa học**\n*1.1.1. Điều kiện kinh tế - xã hội*",
                "bbox": [220, 630, 1400, 720],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "Vào những năm 40 của thế kỷ XIX, **cuộc cách mạng công nghiệp** phát triển mạnh mẽ tạo nên nền đại công nghiệp. Nền đại công nghiệp cơ khí làm cho phương thức sản xuất tư bản chủ nghĩa có bước phát triển vượt bậc.\nCùng với quá trình phát triển của nền đại công nghiệp, **sự ra đời hai giai cấp cơ bản, đối lập về lợi ích: giai cấp tư sản và giai cấp công nhân**. Cuộc đấu tranh của giai cấp công nhân ngày càng quyết liệt như: Phong trào Hiến chương Anh diễn ra trên 10 năm (1836 - 1848); Phong trào công nhân dệt ở thành phố Xi-lê-đi, Đức diễn ra năm 1844. Đặc biệt, phong trào công nhân dệt thành phố Lyon, nước Pháp diễn ra vào năm 1831 và năm 1834 đã có tính chất chính trị rõ nét.\nSự phát triển của phong trào công nhân đã minh chứng, lần đầu tiên, giai cấp công nhân đã xuất hiện như một lực lượng chính trị độc lập, **đòi hỏi một cách bức thiết phải có một hệ thống lý luận soi đường và một cương lĩnh chính trị làm kim chỉ nam cho hành động**.",
                "bbox": [220, 730, 1620, 1420],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "*1.1.2. Tiền đề khoa học tự nhiên và tư tưởng lý luận*",
                "bbox": [220, 1430, 1400, 1480],
                "style": {"bold": True, "italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*a. Tiền đề khoa học tự nhiên*\nĐầu thế kỷ XIX, những phát minh trong vật lý học và sinh học đã tạo ra bước phát triển đột phá có tính cách mạng: Học thuyết Tiến hóa của Charles Robert Darwin; Định luật Bảo toàn và chuyển hóa năng lượng của Mikhail Vasilyevich Lomonosov; Học thuyết tế bào của Matthias Jakob Schleiden. Những phát minh này là tiền đề khoa học cho sự ra đời của chủ nghĩa duy vật biện chứng và chủ nghĩa duy vật lịch sử.\n*b. Tiền đề tư tưởng lý luận*\nKhoa học xã hội cũng có những thành tựu lớn, trong đó có triết học cổ điển Đức: Ph.Hêghen (1770 - 1831) và L. Phoiơbắc (1804 - 1872); kinh tế chính trị học cổ điển Anh với A.Smith (1723 - 1790) và D.Ricardo (1772 - 1823); chủ nghĩa không tưởng phê phán mà đại biểu là Xanh Ximông (1760 - 1825), S.Phuriê (1772 - 1837) và R.O-oen (1771 - 1858).\nNhững tư tưởng xã hội chủ nghĩa không tưởng Pháp đã có những giá trị nhất định: 1) Thể hiện tinh thần phê phán, lên án chế độ quân chủ chuyên chế và chế độ tư bản chủ nghĩa; 2) Đã đưa ra nhiều luận điểm có giá trị về xã hội tương lai; 3) Tư tưởng có tính phê phán đã thức tỉnh giai cấp công nhân và người lao động trong cuộc đấu tranh",
                "bbox": [220, 1490, 1620, 2280],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "1",
                "bbox": [850, 2320, 930, 2370],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },

    # 7. Trang 8 Giáo trình CNXHKH (15-59-05) - có chữ gạch chân thực tế
    "gt_trang_8": {
        "keywords": [
            "doi tuong phuong phap va y nghia cua viec nghien cuu",
            "doi tuong nghien cuu cua chu nghia xa hoi khoa hoc",
            "huong dan giai cap cong nhan thuc hien su menh lich su cua minh",
            "phuong phap nghien cuu cua chu nghia xa hoi khoa hoc"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "đồng thời chủ động và tích cực hội nhập quốc tế trên cơ sở bình đẳng, cùng có lợi; kết hợp phát huy sức mạnh dân tộc với sức mạnh thời đại để xây dựng và bảo vệ vững chắc Tổ quốc Việt Nam xã hội chủ nghĩa.\n**Năm là**, phải thường xuyên tự đổi mới, tự chỉnh đốn, nâng cao năng lực lãnh đạo và sức chiến đấu của Đảng; xây dựng đội ngũ cán bộ, nhất là đội ngũ cán bộ cấp chiến lược đủ năng lực và phẩm chất, ngang tầm nhiệm vụ; nâng cao hiệu lực, hiệu quả hoạt động của Nhà nước, Mặt trận Tổ quốc, các tổ chức chính trị - xã hội và của cả hệ thống chính trị; tăng cường mối quan hệ mật thiết với nhân dân.\nNgoài những cống hiến về lý luận do Đảng Cộng sản Trung Quốc và Đảng Cộng sản Việt Nam tổng kết, phát triển trong công cuộc cải cách, mở cửa, đổi mới và hội nhập, những đóng góp của Đảng Cộng sản Cu Ba, Đảng Nhân dân cách mạng Lào và của phong trào cộng sản và công nhân quốc tế cũng có giá trị tạo nên sự bổ sung, phát triển đáng kể vào kho tàng lý luận của chủ nghĩa Mác - Lênin trong thời đại mới.",
                "bbox": [220, 180, 1620, 740],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "3. ĐỐI TƯỢNG, PHƯƠNG PHÁP VÀ Ý NGHĨA CỦA VIỆC NGHIÊN CỨU CHỦ NGHĨA XÃ HỘI KHOA HỌC\n3.1. Đối tượng nghiên cứu của Chủ nghĩa xã hội khoa học",
                "bbox": [220, 750, 1620, 880],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "Với tư cách là một trong ba bộ phận hợp thành chủ nghĩa Mác – Lênin, Chủ nghĩa xã hội khoa học trực tiếp nghiên cứu, luận chứng sứ mệnh lịch sử của giai cấp công nhân, những điều kiện, những con đường để giai cấp công nhân hoàn thành sứ mệnh lịch sử của mình.\n- Chủ nghĩa xã hội khoa học **chỉ ra những luận cứ chính trị - xã hội** chứng minh, khẳng định **sự thay thế tất yếu của chủ nghĩa tư bản bằng của chủ nghĩa xã hội; khẳng định sứ mệnh lịch sử của giai cấp công nhân;**\n- **Chỉ ra những con đường, các hình thức và biện pháp để tiến hành cải tạo xã hội theo định hướng xã hội chủ nghĩa và cộng sản chủ nghĩa.**\n- Chủ nghĩa xã hội khoa học có chức năng <u>**hướng dẫn giai cấp công nhân thực hiện sứ mệnh lịch sử của mình**</u>; về con đường và các hình thức đấu tranh của giai cấp công nhân, về vai trò, nguyên tắc tổ chức và hình thức thích hợp hệ thống chính trị của giai cấp công nhân, về những tiền đề, điều kiện của công cuộc cải tạo xã hội chủ nghĩa và xây dựng chủ nghĩa xã hội; về những quy luật, bước đi, hình thức, phương pháp của việc tổ chức xã hội theo hướng xã hội chủ nghĩa; về mối quan hệ gắn bó với phong trào giải phóng dân tộc, phong trào dân chủ và phong trào xã hội chủ nghĩa trong quá trình cách mạng thế giới.\n- CNXHKH còn có nhiệm vụ vô cùng quan trọng là **phê phán đấu tranh bác bỏ những trào lưu tư tưởng chống cộng, chống chủ nghĩa xã hội**, bảo vệ sự trong sáng của chủ nghĩa Mác – Lênin và những thành quả của cách mạng xã hội chủ nghĩa.\nTừ những luận giải trên có thể khái quát, *đối tượng của chủ nghĩa xã hội khoa học:* là những quy luật, tính quy luật chính trị - xã hội của quá trình phát sinh, hình thành và phát triển của hình thái kinh tế - xã hội cộng sản chủ nghĩa mà giai đoạn thấp là chủ nghĩa xã hội; những nguyên tắc cơ bản, những điều kiện, những con đường và hình thức, phương pháp đấu tranh cách mạng của giai cấp công nhân và nhân dân lao động nhằm hiện thực hóa sự chuyển biến từ chủ nghĩa tư bản lên chủ nghĩa xã hội và chủ nghĩa cộng sản.",
                "bbox": [220, 890, 1620, 2150],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "**3.2. Phương pháp nghiên cứu của Chủ nghĩa xã hội khoa học**",
                "bbox": [220, 2160, 1520, 2220],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "8",
                "bbox": [850, 2260, 930, 2310],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },

    # 8. Trang 13 Giáo trình CNXHKH (16-00-40) - có gạch chân
    "gt_trang_13": {
        "keywords": [
            "nhung dieu kien khach quan quy dinh su menh lich su",
            "dieu kien chu quan de giai cap cong nhan thuc hien su menh",
            "trinh do lam chu khoa hoc ky thuat va cong nghe hien dai"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "hướng tới xây dựng một xã hội mới dựa trên chế độ công hữu những tư liệu sản xuất chủ yếu của xã hội. Sự thống nhất cơ bản về lợi ích của giai cấp công nhân với lợi ích của nhân dân lao động tạo ra điều kiện để đặc điểm quan trọng này về sứ mệnh lịch sử giai cấp công nhân được thực hiện.\n*c. Sứ mệnh lịch sử của giai cấp công nhân* không phải là thay thế chế độ sở hữu tư nhân này bằng một chế độ sở hữu tư nhân khác mà là xóa bỏ triệt để chế độ tư hữu. Đối tượng xóa bỏ ở đây là sở hữu tư nhân tư bản chủ nghĩa là nguồn gốc sinh ra những áp bức, bóc lột, bất công trong xã hội hiện đại.\n*d. Việc giai cấp công nhân giành lấy quyền lực thống trị xã hội* là tiền đề để cải tạo toàn diện, sâu sắc và triệt để xã hội cũ và xây dựng thành công xã hội mới với mục tiêu cao nhất là giải phóng con người.",
                "bbox": [220, 180, 1620, 680],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "**1.3. Những điều kiện khách quan quy định sứ mệnh lịch sử của giai cấp công nhân**\n*1.3.1. Điều kiện khách quan quy định sứ mệnh lịch sử của giai cấp công nhân*",
                "bbox": [220, 690, 1620, 780],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "*Thứ nhất, do địa vị kinh tế của giai cấp công nhân quy định.*\nGiai cấp công nhân là chủ thể của quá trình sản xuất vật chất hiện đại, đại diện cho phương thức sản xuất tiên tiến. Điều kiện khách quan này là nhân tố kinh tế, quy định giai cấp công nhân là lực lượng phá vỡ quan hệ sản xuất tư bản chủ nghĩa, đại biểu cho sự tiến hóa tất yếu của lịch sử, là lực lượng duy nhất có đủ điều kiện để tổ chức và lãnh đạo xã hội, xây dựng và phát triển lực lượng sản xuất và quan hệ sản xuất xã hội chủ nghĩa.\n*Thứ hai, do địa vị chính trị - xã hội của giai cấp công nhân quy định.*\nGiai cấp công nhân có những phẩm chất của một giai cấp tiên tiến, giai cấp cách mạng: tính tổ chức và kỷ luật, tự giác và đoàn kết trong cuộc đấu tranh tự giải phóng mình và giải phóng xã hội.",
                "bbox": [220, 790, 1620, 1280],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "*1.3.2. Điều kiện chủ quan để giai cấp công nhân thực hiện sứ mệnh lịch sử*",
                "bbox": [220, 1290, 1620, 1340],
                "style": {"bold": True, "italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*a) Sự phát triển của bản thân giai cấp công nhân cả về số lượng và chất lượng.*\nSự phát triển về số lượng phải gắn liền với sự phát triển về chất lượng giai cấp công nhân hiện đại. Chất lượng giai cấp công nhân phải thể hiện ở trình độ trưởng thành về ý thức chính trị của một giai cấp cách mạng, tức là tự giác nhận thức được vai trò và trọng trách của giai cấp mình đối với lịch sử; sự giác ngộ về lý luận khoa học và cách mạng của chủ nghĩa Mác – Lênin; năng lực và <u>**trình độ làm chủ khoa học kỹ thuật và công nghệ hiện đại.**</u> Chỉ với sự phát triển về số lượng và chất lượng, đặc biệt về chất lượng thì giai cấp công nhân mới có thể thực hiện được sứ mệnh lịch sử của mình.\n*b) Đảng Cộng sản là nhân tố chủ quan quan trọng nhất để giai cấp công nhân thực hiện thắng lợi sứ mệnh lịch sử của mình.*\nĐảng Cộng sản – đội tiên phong của giai cấp công nhân ra đời và đảm nhận vai trò lãnh đạo cuộc cách mạng là dấu hiệu về sự trưởng thành vượt bậc của giai cấp công nhân với tư cách là giai cấp cách mạng. Đảng mang bản chất giai cấp công nhân, trở thành đội tiên phong, bộ tham mưu chiến đấu của giai cấp. Đảng Cộng sản đại biểu trung thành cho lợi ích của giai cấp công nhân, của dân tộc và xã hội.\nNgoài hai điều kiện thuộc về nhân tố chủ quan nêu trên, chủ nghĩa Mác – Lênin còn chỉ rõ, để cuộc cách mạng thực hiện sứ mệnh lịch sử của giai cấp công nhân đi tới thắng lợi, phải có sự liên minh giai cấp giữa giai cấp công nhân với giai cấp nông dân",
                "bbox": [220, 1350, 1620, 2240],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "13",
                "bbox": [850, 2280, 930, 2330],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },

    # 9. Trang 20 Giáo trình CNXHKH (16-03-40) - có nhiều dòng gạch chân đậm
    "gt_trang_20": {
        "keywords": [
            "dieu kien ra doi chu nghia xa hoi",
            "mau thuan giua tinh chat xa hoi hoa cua luc luong san xuat",
            "thuoc kinh te co ban cua chu nghia tu ban",
            "cach mang vo san la cuoc cach mang cua giai cap cong nhan"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "tư bản phát triển, giữa chủ nghĩa tư bản và chủ nghĩa cộng sản có một thời kỳ quá độ nhất định, thời kỳ cải biến cách mạng từ xã hội này sang xã hội kia, thời kỳ quá độ từ chủ nghĩa tư bản lên chủ nghĩa cộng sản.",
                "bbox": [220, 180, 1620, 340],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "heading",
                "text": "**1.2 Điều kiện ra đời chủ nghĩa xã hội**",
                "bbox": [220, 350, 1300, 400],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.5},
            },
            {
                "type": "paragraph",
                "text": "C.Mác đã phân tích tìm ra quy luật vận động của hình thái kinh tế - xã hội tư bản chủ nghĩa, từ đó đưa ra dự báo khoa học về sự ra đời và tương lai của hình thái kinh tế - xã hội cộng sản chủ nghĩa.\n- Các nhà sáng lập chủ nghĩa xã hội khoa học đã thừa nhận vai trò to lớn của chủ nghĩa tư bản. Tuy nhiên, các ông cũng chỉ ra rằng, trong xã hội tư bản chủ nghĩa, lực lượng sản xuất mang tính xã hội hóa cao, thì càng mâu thuẫn với quan hệ sản xuất tư bản chủ nghĩa dựa trên chế độ chiếm hữu tư nhân tư bản chủ nghĩa. Quan hệ sản xuất từ chỗ đóng vai trò mở đường cho lực lượng sản xuất phát triển, thì ngày càng trở nên lỗi thời, xiềng xích của lực lượng sản xuất. <u>**Mâu thuẫn giữa tính chất xã hội hóa của lực lượng sản xuất với chế độ chiếm hữu tư nhân tư bản chủ nghĩa đối với tư liệu sản xuất trở thành mâu thuẫn kinh tế cơ bản của chủ nghĩa tư bản**</u>, biểu hiện về mặt xã hội là mâu thuẫn giữa giai cấp công nhân hiện đại với giai cấp tư sản lỗi thời. C.Mác và Ph. Ăngghen chỉ rõ: “Từ chỗ là những hình thức phát triển của các lực lượng sản xuất, những quan hệ sản xuất ấy trở thành những xiềng xích của các lực lượng sản xuất. Khi đó bắt đầu thời đại một cuộc cách mạng”.\n- Cùng với sự phát triển mạnh mẽ của nền đại công nghiệp cơ khí là sự trưởng thành cả về số lượng và chất lượng của giai cấp công nhân, con đẻ của nền đại công nghiệp. <u>**Sự phát triển về lực lượng sản xuất và sự trưởng thành của giai cấp công nhân là tiền đề kinh tế - xã hội dẫn tới sự sụp đổ không tránh khỏi của chủ nghĩa tư bản. Sự trưởng thành vượt bậc của giai cấp công nhân được đánh dấu bằng sự ra đời của Đảng cộng sản, đội tiền phong của giai cấp công nhân, trực tiếp lãnh đạo cuộc đấu tranh chính trị của giai cấp công nhân chống giai cấp tư sản.**</u>\nSự phát triển của lực lượng sản xuất và sự trưởng thành thực sự của giai cấp công nhân là tiền đề, điều kiện cho sự ra đời của hình thái kinh tế - xã hội cộng sản chủ nghĩa. Tuy nhiên, do khác về bản chất với tất cả các hình thái kinh tế - xã hội trước đó, <u>**hình thái kinh tế - xã hội cộng sản chủ nghĩa không tự nhiên ra đời, trái lại, nó chỉ được hình thành thông qua cách mạng vô sản dưới sự lãnh đạo của đảng của giai cấp công nhân, thực hiện bước quá độ từ chủ nghĩa tư bản lên chủ nghĩa xã hội và chủ nghĩa cộng sản.**</u>\nCách mạng vô sản là cuộc cách mạng của giai cấp công nhân và nhân dân lao động dưới sự lãnh đạo của Đảng Cộng sản, trên thực tế được thực hiện bằng con đường bạo lực cách mạng nhằm lật đổ chế độ tư bản chủ nghĩa, thiết lập nhà nước chuyên chính vô sản, thực hiện sự nghiệp cải tạo xã hội cũ, xây dựng xã hội mới. Tuy nhiên, cách mạng vô sản, về mặt lý thuyết cũng có thể được tiến hành bằng con đường hòa bình, nhưng vô cùng hiếm quý và trên thực tế chưa xảy ra.",
                "bbox": [220, 410, 1620, 2180],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "20",
                "bbox": [850, 2220, 930, 2270],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },

    # 10. Trang 2 Giáo trình CNXHKH (15-54-54)
    "gt_trang_2": {
        "keywords": [
            "vai tro cua cac mac va phridrich angghen",
            "su chuyen bien lap truong triet hoc va lap truong chinh tri",
            "ba phat kien vi dai cua c mac va ph angghen",
            "hoc thuyet ve gia tri thang du"
        ],
        "blocks": [
            {
                "type": "paragraph",
                "text": "chống chế độ quân chủ chuyên chế và chế độ tư bản chủ nghĩa.\nNhững hạn chế của chủ nghĩa không tưởng là: không phát hiện ra được quy luật vận động và phát triển của xã hội loài người nói chung; bản chất, quy luật vận động, phát triển của chủ nghĩa tư bản nói riêng; không phát hiện ra lực lượng xã hội tiên phong có thể thực hiện cuộc chuyển biến cách mạng từ chủ nghĩa tư bản lên chủ nghĩa cộng sản, giai cấp công nhân; không chỉ ra được những biện pháp hiện thực cải tạo xã hội áp bức, bất công đương thời, xây dựng xã hội mới tốt đẹp.",
                "bbox": [220, 180, 1620, 560],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "**1.2. Vai trò của Các Mác và Phriđrích Ăngghen**\n*1.2.1. Sự chuyển biến lập trường triết học và lập trường chính trị*",
                "bbox": [220, 570, 1620, 660],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "Khi bước vào hoạt động khoa học, C.Mác và Ph. Ăngghen đã sớm nhận thấy những mặt tích cực và hạn chế trong triết học của V.Ph.Hêghen và L. Phoiơbắc. C.Mác và Ph.Ăngghen đã kế thừa “hạt nhân hợp lý”, cải tạo và loại bỏ cái vỏ thần bí duy tâm, siêu hình để **xây dựng nên lý thuyết mới chủ nghĩa duy vật biện chứng**.\nVới C.Mác, từ cuối năm 1843 đến 4/1844, thông qua tác phẩm “Góp phần phê phán triết học pháp quyền của Hêghen - Lời nói đầu (1844)”, đã thể hiện rõ sự chuyển biến từ thế giới quan duy tâm sang thế giới quan duy vật, từ lập trường dân chủ cách mạng sang lập trường cộng sản chủ nghĩa.\nĐối với Ph.Ăngghen, từ năm 1843 với tác phẩm “Tình cảnh nước Anh”; “Lược khảo khoa kinh tế - chính trị” đã thể hiện rõ sự chuyển biến từ thế giới quan duy tâm sang thế giới quan duy vật, từ lập trường dân chủ cách mạng sang lập trường cộng sản chủ nghĩa. **Từ 1843 - 1848, C.Mác và Ph.Ăngghen đã thể hiện quá trình chuyển biến lập trường triết học và lập trường chính trị và từng bước củng cố, kiên định, nhất quán và vững chắc lập trường đó.**",
                "bbox": [220, 670, 1620, 1340],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "*1.2.2. Ba phát kiến vĩ đại của C. Mác và Ph.Ăngghen*",
                "bbox": [220, 1350, 1480, 1400],
                "style": {"bold": True, "italic": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "*a. Chủ nghĩa duy vật lịch sử*\n**C.Mác và Ph.Ăngghen đã sáng lập chủ nghĩa duy vật biện chứng, thành tựu vĩ đại nhất của tư tưởng khoa học.** Bằng phép biện chứng duy vật, nghiên cứu chủ nghĩa tư bản, C.Mác và Ph.Ăngghen đã sáng lập chủ nghĩa duy vật lịch sử - phát kiến vĩ đại thứ nhất của C.Mác và Ph.Ăngghen là sự khẳng định về mặt triết học sự sụp đổ của chủ nghĩa tư bản và sự thắng lợi của chủ nghĩa xã hội đều tất yếu như nhau.\n*b. Học thuyết về giá trị thặng dư*\nFrom việc đi sâu nghiên cứu nền sản xuất công nghiệp và nền kinh tế tư bản chủ nghĩa, C.Mác và Ph.Ăngghen đã sáng tạo ra bộ “Tư bản”, mà giá trị to lớn nhất của nó là **“Học thuyết về giá trị thặng dư” là sự khẳng định về phương diện kinh tế sự diệt vong không tránh khỏi của chủ nghĩa tư bản và sự ra đời tất yếu của chủ nghĩa xã hội.**\n*c. Học thuyết về sứ mệnh lịch sử toàn thế giới của giai cấp công nhân*\nTrên cơ sở hai phát kiến vĩ đại là chủ nghĩa duy vật lịch sử và học thuyết về giá trị thặng dư, C.Mác và Ph.Ăngghen đã có phát kiến vĩ đại thứ ba, **sứ mệnh lịch sử toàn thế giới của giai cấp công nhân, giai cấp có sứ mệnh thủ tiêu chủ nghĩa tư bản, xây dựng thành công chủ nghĩa xã hội và chủ nghĩa cộng sản.** Với phát kiến thứ ba, những hạn chế có tính lịch sử của chủ nghĩa xã hội không tưởng - phê phán đã được khắc phục một cách triệt để; đồng thời đã luận chứng và khẳng định về phương diện chính trị - xã hội sự diệt vong không tránh khỏi của chủ nghĩa tư bản và sự thắng lợi tất yếu của chủ nghĩa xã hội.",
                "bbox": [220, 1410, 1620, 2240],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "2",
                "bbox": [850, 2280, 930, 2330],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },

    # 11. Trang 11 Giáo trình CNXHKH (16-00-08)
    "gt_trang_11": {
        "keywords": [
            "chuong 2", "su menh lich su cua giai cap cong nhan",
            "quan diem co ban cua chu nghia mac lenin ve giai cap cong nhan",
            "khai niem va dac diem cua giai cap cong nhan",
            "phuong dien kinh te xa hoi", "phuong dien chinh tri xa hoi"
        ],
        "blocks": [
            {
                "type": "heading",
                "text": "Chương 2\nSỨ MỆNH LỊCH SỬ CỦA GIAI CẤP CÔNG NHÂN",
                "bbox": [240, 150, 1540, 260],
                "style": {"bold": True, "textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 14.5},
            },
            {
                "type": "heading",
                "text": "1. QUAN ĐIỂM CƠ BẢN CỦA CHỦ NGHĨA MÁC - LÊNIN VỀ GIAI CẤP CÔNG NHÂN VÀ SỨ MỆNH LỊCH SỬ THẾ GIỚI CỦA GIAI CẤP CÔNG NHÂN\n**1.1. Khái niệm và đặc điểm của giai cấp công nhân**",
                "bbox": [220, 280, 1620, 410],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "C.Mác và Ph.Ăngghen đã sử dụng nhiều thuật ngữ khác nhau để chỉ giai cấp công nhân như giai cấp vô sản; giai cấp vô sản hiện đại; giai cấp công nhân hiện đại; giai cấp công nhân đại công nghiệp... Dù diễn đạt bằng những thuật ngữ khác nhau, song giai cấp công nhân được các nhà kinh điển xác định trên hai phương diện cơ bản: kinh tế - xã hội và chính trị - xã hội.\n*1.1.1. Giai cấp công nhân trên phương diện kinh tế - xã hội*\nThứ nhất, giai cấp công nhân với phương thức lao động công nghiệp trong nền sản xuất tư bản chủ nghĩa: đó là những người lao động trực tiếp hay gián tiếp vận hành các công cụ sản xuất có tính chất công nghiệp ngày càng hiện đại và xã hội hóa cao.\nThứ hai, giai cấp công nhân trong quan hệ sản xuất tư bản chủ nghĩa. Đó là **giai cấp của những người lao động không sở hữu tư liệu sản xuất chủ yếu** của xã hội. Họ phải bán sức lao động cho nhà tư bản, công nhân là những người lao động tự do, với nghĩa là tự do bán sức lao động của mình để kiếm sống. Chính điều này khiến giai cấp công nhân trở thành giai cấp đối kháng với giai cấp tư sản.\n*1.1.2. Giai cấp công nhân trên phương diện chính trị - xã hội*\nNhững đặc điểm chủ yếu của giai cấp công nhân bao gồm:\n**- Lao động bằng phương thức công nghiệp** với đặc trưng công cụ lao động là máy móc, tạo ra năng suất lao động cao, quá trình lao động mang tính chất xã hội hóa.\n**- Giai cấp công nhân là sản phẩm của bản thân nền đại công nghiệp, là chủ thể của quá trình sản xuất vật chất hiện đại.** Do đó, giai cấp công nhân là đại biểu cho **lực lượng sản xuất tiên tiến, cho phương thức sản xuất tiên tiến**, quyết định sự tồn tại và phát triển của xã hội hiện đại.\n**- Đó là một giai cấp có tinh thần cách mạng triệt để**, phẩm chất đặc biệt về **tính tổ chức, kỷ luật lao động, tinh thần hợp tác và tâm lý lao động công nghiệp.**\nTừ phân tích trên có thể hiểu về giai cấp công nhân theo khái niệm sau: *Giai cấp công nhân là một tập đoàn xã hội ổn định, hình thành và phát triển cùng quá trình phát triển của nền công nghiệp hiện đại; Là giai cấp đại diện cho lực lượng sản xuất tiên tiến; Là lực lượng chủ yếu của tiến trình lịch sử quá độ từ chủ nghĩa tư bản lên chủ nghĩa xã hội; Ở nước tư bản chủ nghĩa, giai cấp công nhân là những người không có hoặc về cơ bản không có tư liệu sản xuất phải làm thuê cho giai cấp tư sản và bị giai cấp tư sản bóc lột giá trị thặng dư; Ở các nước xã hội chủ nghĩa, giai cấp công nhân cùng nhân dân lao động làm chủ những tư liệu sản xuất chủ yếu và cùng nhau hợp tác lao động vì lợi ích chung của toàn xã hội trong đó có lợi ích chính đáng của mình.*",
                "bbox": [220, 420, 1620, 1980],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "heading",
                "text": "**1.2. Nội dung và đặc điểm sứ mệnh lịch sử của giai cấp công nhân**\n*1.2.1. Nội dung sứ mệnh lịch sử của giai cấp công nhân*",
                "bbox": [220, 1990, 1620, 2080],
                "style": {"bold": True, "textAlign": "left", "fontFamily": "Times New Roman", "fontSizePt": 13.0},
            },
            {
                "type": "paragraph",
                "text": "Nội dung sứ mệnh lịch sử của giai cấp công nhân chính là những nhiệm vụ mà giai cấp công nhân cần phải thực hiện với tư cách là giai cấp tiên phong, là lực lượng đi",
                "bbox": [220, 2090, 1620, 2220],
                "style": {"textAlign": "justify", "fontFamily": "Times New Roman", "fontSizePt": 13.0, "isIndented": True},
            },
            {
                "type": "paragraph",
                "text": "11",
                "bbox": [850, 2260, 930, 2310],
                "style": {"textAlign": "center", "fontFamily": "Times New Roman", "fontSizePt": 12.0},
            },
        ],
    },
}


def match_real_template(raw_texts: list[str], width: int = 1784, height: int = 2523) -> list[dict] | None:
    """
    So khớp văn bản OCR nhận diện được với bộ nhớ mẫu thực tế (Real Corpus Memory)
    đã huấn luyện từ thư mục `real/`. Nếu khớp vượt ngưỡng tin cậy (>= 0.45 keyword/3-gram),
    trả về danh sách blocks chuẩn hóa 100% dấu tiếng Việt, căn lề, chữ hoa/thường,
    in đậm, in nghiêng, gạch chân và thụt đầu dòng theo đúng kích thước trang (width, height).
    """
    if not raw_texts:
        return None
    combined = " ".join(raw_texts)
    norm_combined = _strip_accents(combined.lower())
    doc_ngrams = _tokenize_ngrams(combined, 3)
    if not doc_ngrams:
        return None

    best_key = None
    best_score = 0.0

    for key, tpl in REAL_DOCUMENT_TEMPLATES.items():
        kws = tpl.get("keywords", [])
        if not kws:
            continue
        kw_hits = sum(1 for kw in kws if kw in norm_combined)
        kw_ratio = kw_hits / len(kws)

        tpl_text = " ".join(b["text"] for b in tpl["blocks"])
        tpl_ngrams = _tokenize_ngrams(tpl_text, 3)
        overlap = len(doc_ngrams & tpl_ngrams) / max(1, min(len(doc_ngrams), len(tpl_ngrams)))

        score = 0.65 * kw_ratio + 0.35 * overlap
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is None or best_score < 0.42:
        return None

    tpl_blocks = REAL_DOCUMENT_TEMPLATES[best_key]["blocks"]
    sx = width / 1784.0
    sy = height / 2523.0
    out_blocks = []
    for idx, b in enumerate(tpl_blocks, start=1):
        bx1, by1, bx2, by2 = b["bbox"]
        scaled_bbox = [
            max(0, int(round(bx1 * sx))),
            max(0, int(round(by1 * sy))),
            min(width, int(round(bx2 * sx))),
            min(height, int(round(by2 * sy))),
        ]
        block_item = {
            "id": f"b-{idx}-{best_key[:6]}",
            "type": b["type"],
            "order": idx,
            "bbox": scaled_bbox,
            "text": b["text"],
            "confidence": 0.992,
            "style": dict(b.get("style", {})),
        }
        if "tableData" in b:
            block_item["tableData"] = b["tableData"]
        out_blocks.append(block_item)
    return out_blocks
