# THIẾT KẾ CHI TIẾT: AI MODEL & INFERENCE LOGIC

Phần này do Người A (Data & Model Engineer) đảm nhiệm (có sự hỗ trợ phần mạng của C).

## 1. Công nghệ cốt lõi
- **Library**: OpenCV (đọc ảnh), Google MediaPipe Face Mesh (trích xuất 468 landmark khuôn mặt).
- **Tiếp cận**: Rule-based Thresholding (Đánh giá theo ngưỡng tĩnh hoặc động dựa trên chỉ số khuôn mặt), không dùng mô hình Deep Learning (CNN/LSTM) trong 5 ngày đầu.

## 2. Chiến lược sử dụng Dataset (YawDD)
- **Dataset duy nhất**: Dự án CHỈ SỬ DỤNG bộ dataset YawDD (gồm các video mô phỏng trạng thái bình thường, nói chuyện, ngáp, và hát).
- **Cách đánh giá Rule-based trên YawDD**:
  - **MAR (Ngáp)**: Rất phù hợp vì YawDD có sẵn nhãn/video người đang ngáp, giúp tinh chỉnh ngưỡng MAR chính xác.
  - **EAR & PERCLOS (Nhắm mắt/Ngủ gật)**: Vì YawDD tập trung vào biểu cảm miệng và không có nhãn "ngủ gật" chuyên biệt, quá trình tinh chỉnh ngưỡng EAR sẽ dựa vào các khoảnh khắc chớp mắt tự nhiên trong video. Để kiểm thử luồng PERCLOS khắt khe (nhắm mắt > 2s), nhóm cần tự quay bổ sung các clip test thực tế (nhắm mắt giả định) vì YawDD không bao phủ hết case này.

## 3. Quy trình xử lý ảnh (Inference Pipeline)
Pipeline xử lý tại Edge sẽ chạy liên tục trên từng frame:
1. **Tiền xử lý (Pre-processing):** 
   - Thay đổi kích thước (Resize) để tối ưu FPS.
   - Chuyển đổi BGR (OpenCV) sang RGB (MediaPipe yêu cầu).
2. **Inference (Face Mesh):** 
   - Đưa ảnh vào `FaceLandmarker`.
   - Trích xuất danh sách điểm landmark 3D/2D.
3. **Tính toán chỉ số (Feature Extraction):**
   - Lọc ra mảng index của mắt trái (vd: 362, 385, 387, 263, 373, 380), mắt phải (33, 160, 158, 133, 153, 144) và miệng.
   - Tính `EAR_left`, `EAR_right` -> `EAR_avg`.
   - Tính `MAR`.
4. **Logic Phân loại (Classification Logic):**
   - Quản lý trạng thái thông qua Bộ nhớ đệm (Ring Buffer/Deque) lưu `EAR` của N frame gần nhất để tính **PERCLOS**.

## 4. Thuật toán Rule-based chi tiết

### 4.1. Theo dõi ngáp (Yawning)
- Điều kiện: `MAR > MAR_THRESHOLD` (vd: 0.5).
- Rule chống nhiễu (Debounce): Tình trạng `MAR > 0.5` phải duy trì liên tục trong tối thiểu `MIN_YAWN_FRAMES` (vd: 15-20 frames ~ 1 giây).
- **Trạng thái**: `YAWNING`

### 4.2. Theo dõi nhắm mắt (Micro-sleep)
- Điều kiện: `EAR_avg < EAR_THRESHOLD` (vd: 0.22).
- Rule cảnh báo nhắm mắt tạm thời: Duy trì liên tục trong `MIN_EYE_CLOSED_FRAMES` (vd: 10-15 frames).
- **Trạng thái**: `MICRO_SLEEP`

### 4.3. Theo dõi buồn ngủ dài hạn (PERCLOS)
PERCLOS đánh giá tỷ lệ nhắm mắt trong 1 khoảng thời gian (Window).
- **Cấu hình**: `WINDOW_SIZE = 300` frames (khoảng 20-30 giây).
- **Logic**: 
  - Khởi tạo queue (maxsize=300).
  - Với mỗi frame, đưa kết quả `1` (nếu nhắm) hoặc `0` (nếu mở) vào queue.
  - Tính tổng phần tử có giá trị `1` chia cho số phần tử hiện tại.
  - Nếu `PERCLOS > 0.15` (15%) -> **Trạng thái**: `DROWSY_WARNING`.
  - Nếu `PERCLOS > 0.25` (25%) -> **Trạng thái**: `DROWSY_CRITICAL`.

## 5. Thiết kế cấu trúc file cấu hình `edge.yaml`
```yaml
# configs/edge.yaml
camera:
  source: 0
  resolution: [640, 480]
  fps: 15

model:
  ear_threshold: 0.22
  mar_threshold: 0.5
  perclos_window_frames: 300
  consecutive_closed_frames: 15
  consecutive_yawn_frames: 20

network:
  ws_url: "ws://localhost:8000/ws/edge/rpi_01"
  send_interval_ms: 100 # Gửi 10 dữ liệu 1 giây
```
