# Hướng dẫn Cộng tác MLOps (Data & Code Workflow)

Dự án này áp dụng mô hình quản lý hiện đại **Git + DVC (Data Version Control)**. 
- **Git (Github):** Dùng để lưu trữ Code (Dung lượng siêu nhẹ).
- **DVC (DAGsHub):** Dùng để lưu trữ Dữ liệu Ảnh/Video (Dung lượng khổng lồ).

Tài liệu này là "Kim chỉ nam" giúp bất kỳ thành viên nào (Kể cả sinh viên mới) tải dự án về và làm việc một cách mượt mà nhất.

---

## 1. Dành cho Thành viên mới (Setup Dự án lần đầu)

Khi bạn vừa mới gia nhập Team, ổ cứng của bạn chưa có gì cả. Hãy làm đúng 3 bước sau:

**Bước 1: Tải Code từ Github về máy**
```bash
git clone https://github.com/dut-anhtu24/driver-monitoring-system.git
cd driver-monitoring-system
git checkout dev
```

**Bước 2: Cài đặt Môi trường & Thư viện**
Đảm bảo bạn đã chọn đúng môi trường Python (Khuyên dùng Python 3.10 - 3.13) trong VS Code, sau đó gõ:
```bash
pip install -r requirements.txt
# (Hoặc cài thủ công: pip install dvc)
```

**Bước 3: Dùng "Phép thuật" triệu hồi Dữ liệu**
Lúc này, thư mục `dataset/processed/` của bạn đang TRỐNG RỖNG (Hoặc chỉ có file `.gitignore`). Đừng hoảng sợ, chỉ cần gõ đúng 1 dòng này:
```bash
dvc pull
```
*Giải thích: DVC sẽ đọc file `dataset/processed.dvc`, lên kho DAGsHub và tự động tải toàn bộ hàng ngàn bức ảnh Vàng về máy của bạn. Máy bạn giờ đã sẵn sàng để Train YOLO!*

---

## 2. Dành cho Data Engineer (Cập nhật Dữ liệu mới)

Giả sử tuần sau bạn thu thập thêm được 1.000 ảnh mới và muốn tạo ra **Data Version 2**. Hãy làm theo quy trình "Chốt sổ Kép" (Double Push) sau đây:

**Bước 1: Chạy lại Tiền xử lý**
Chép ảnh mới vào `dataset/raw/`, sau đó chạy lại script Tiền xử lý để sinh ra thư mục `processed/` mới.
```bash
python training/data/preprocessing.py
```

**Bước 2: Nạp Data mới vào DVC**
```bash
dvc add dataset/processed
```
*(Lệnh này sẽ cập nhật lại đoạn mã Hash trong file `dataset/processed.dvc`)*

**Bước 3: Đẩy Dữ liệu (Nặng) lên DAGsHub**
```bash
dvc push -r origin
```

**Bước 4: Đẩy Code (Nhẹ) lên Github**
```bash
git add dataset/processed.dvc
git commit -m "build(data): update dataset to Version 2 (added 1000 images)"
git push origin dev
```

---

## 3. Cách Du hành Thời gian (Rollback Version)

Nếu mô hình học Data Version 2 bị ngu đi, và bạn muốn quay về Version 1:
```bash
# Lùi Code về ngày sửa Version 1 (Lấy mã Hash commit trên Github)
git checkout <mã-commit-của-V1>

# Ép Data lùi thời gian theo Code
dvc checkout
```
Ngay lập tức, 1.000 bức ảnh mới của Version 2 sẽ tự động bốc hơi, trả lại cho bạn thư mục Version 1 nguyên vẹn như ngày xưa!
