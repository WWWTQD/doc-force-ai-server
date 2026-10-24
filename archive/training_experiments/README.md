# Archive: Training Experiments

Thư mục này lưu trữ các script thử nghiệm độc lập liên quan đến OCR (PaddleOCR, PP-StructureV3 trên các biến thể ảnh mẫu `Anh1`) và các script kiểm tra parser/debug cũ được tách ra khỏi thư mục `training/`.

Các script trong thư mục này:
- **KHÔNG** thuộc pipeline chính của việc huấn luyện mô hình phục hồi ảnh U-Net (`training/`).
- **KHÔNG** được import hoặc sử dụng bởi backend (`app/`) hoặc bộ test tự động (`tests/`).
- Được lưu lại tại đây nhằm bảo toàn lịch sử nghiên cứu thử nghiệm mà không gây nhầm lẫn trong quá trình huấn luyện và bảo trì pipeline phục hồi ảnh.
