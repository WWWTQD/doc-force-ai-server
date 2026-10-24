# Document Image Restoration Training Pipeline (U-Net)

Thư mục này chứa pipeline huấn luyện và đánh giá mô hình AI phục hồi ảnh tài liệu bị mờ/suy giảm chất lượng (**Document Image Restoration**) sử dụng kiến trúc **Lightweight Residual U-Net** trên nền tảng PyTorch.

> **Lưu ý quan trọng về Dataset:**
> Hiện tại, các file ảnh trong `datasets/images/` chưa có sẵn trên đĩa (đã bị xóa trong working tree của branch `vanh_backup`), dẫn đến **0/15 cặp ảnh hợp lệ** khi đối chiếu với `datasets/metadata/*.csv`. Các thư mục `datasets/processed/`, `datasets/synthetic/`, `datasets/original/`, `datasets/real/` hiện đang rỗng. Pipeline đọc dữ liệu từ các file CSV trong `datasets/metadata/`. Khi bổ sung dữ liệu thật, chỉ cần đặt ảnh đúng đường dẫn được chỉ định trong CSV.

---

## 1. Cấu trúc và vai trò các file

| File | Vai trò |
|---|---|
| `train_restoration.py` | **Entry point huấn luyện chính.** Định nghĩa kiến trúc `DocumentRestorationUNet`, hàm mất mát kết hợp ($L_1 + 0.5 \times \text{MSE} + 2.0 \times \text{SobelLoss}$), vòng lặp huấn luyện, validation, lưu checkpoint tốt nhất (`restoration_best.pth`), checkpoint cuối cùng (`restoration_last.pth`) và nhật ký huấn luyện (`training_log.json`). |
| `restoration_dataset.py` | **Dataset & Tiện ích xử lý ảnh.** Định nghĩa class `DocumentRestorationDataset` kế thừa `torch.utils.data.Dataset`, đọc cặp `(input, target)` từ metadata CSV, kiểm tra tính hợp lệ của file trên đĩa và cảnh báo số cặp thiếu; cung cấp các hàm dùng chung: `load_image`, `save_image`, `resize_with_padding`, `remove_padding`, `unsharp_mask`. |
| `evaluate_restoration.py` | **Đánh giá mô hình đã huấn luyện.** Đọc tập kiểm thử từ CSV, tính toán các chỉ số PSNR, SSIM, so sánh mức độ cải thiện chất lượng trước/sau phục hồi và xuất ảnh đối chiếu (contact sheet) ra `outputs/restoration/`. |
| `infer_restoration.py` | **Inference trên ảnh tài liệu thực tế.** Tải checkpoint U-Net đã train, thực hiện phục hồi chi tiết chữ và khử mờ trên thư mục ảnh đầu vào, áp dụng unsharp mask hậu xử lý. |
| `generate_blur_dataset.py` | **Tạo dữ liệu suy giảm tổng hợp.** Sinh các biến thể mờ (Gaussian blur, motion blur) từ thư mục ảnh tài liệu gốc và tự động tạo `datasets/metadata/blur_pairs.csv`. |
| `split_dataset.py` | **Chia tập train / val / test.** Phân chia dataset theo `source_id` của ảnh gốc (mặc định tỉ lệ 60/20/20) để đảm bảo không rò rỉ dữ liệu giữa các tập, xuất ra `train.csv`, `val.csv`, `test.csv`. |

*Ghi chú: Các script thử nghiệm OCR, kiểm tra PP-Structure và script debug cũ đã được chuyển vào `archive/training_experiments/` để giữ pipeline tinh gọn.*

---

## 2. Hướng dẫn chạy trên Windows

### Cài đặt dependencies
Các thư viện cần thiết được liệt kê trong `requirements-training.txt` (khuyến nghị Python 3.10 / 3.11):
```powershell
pip install -r requirements-training.txt
```

### Chạy huấn luyện (Training)
```powershell
.venv311\Scripts\python.exe training/train_restoration.py `
    --train-csv datasets/metadata/train.csv `
    --val-csv datasets/metadata/val.csv `
    --output-dir models `
    --epochs 20 `
    --batch-size 4 `
    --lr 0.0001 `
    --image-size 512
```

### Đánh giá chất lượng mô hình (Evaluation)
```powershell
.venv311\Scripts\python.exe training/evaluate_restoration.py `
    --model models/restoration_best.pth `
    --test-csv datasets/metadata/test.csv `
    --image-size 512
```

### Chạy Inference trên ảnh thực tế
```powershell
.venv311\Scripts\python.exe training/infer_restoration.py `
    --model models/restoration_best.pth `
    --input-dir path/to/input_images
```

### Chạy kiểm thử tự động pipeline (PyTest)
Bộ kiểm thử độc lập kiểm tra U-Net forward pass, Sobel loss, Dataset loader, 1 epoch mini-train và inference:
```powershell
.venv311\Scripts\python.exe -m pytest tests/test_restoration_pipeline.py -v
```
