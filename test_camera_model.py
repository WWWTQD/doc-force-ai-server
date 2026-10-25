import os
import cv2
from ultralytics import YOLO

# 1. Nạp model nhận diện góc giấy bạn vừa tự train
model_path = os.path.join("worker", "models", "document_detector.pt")
model = YOLO(model_path)

# 2. Đường dẫn tới 1 ảnh chụp bất kỳ từ camera của bạn
test_image_path = "data/input/anh_test.jpg"  # Hãy nhớ để 1 ảnh chụp tờ giấy vào thư mục này nhé

if not os.path.exists(test_image_path):
    print(f"Vui lòng để 1 ảnh chụp thực tế vào đường dẫn: {test_image_path}")
else:
    # 3. Chạy nhận diện
    results = model.predict(source=test_image_path, conf=0.35)
    
    # 4. Vẽ khung nhận diện lên ảnh và hiển thị
    for r in results:
        annotated_img = r.plot()
        cv2.imshow("Ket qua nhan dien tu Camera cua ban", annotated_img)
        print("Đã nhận diện thành công! Nhấn phím bất kỳ trên cửa sổ ảnh để tắt.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()