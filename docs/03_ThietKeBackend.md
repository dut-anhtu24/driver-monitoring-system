# THIẾT KẾ CHI TIẾT: BACKEND LOGIC & GIAO THỨC TRUYỀN THÔNG

Phần này do Người B (Web & Backend Engineer) đảm nhiệm.

## 1. Công nghệ
- **Framework**: FastAPI (hỗ trợ native async và WebSocket rất tốt).
- **Server**: Uvicorn (hoặc Gunicorn + Uvicorn worker).
- **Validation**: Pydantic.

## 2. Giao thức WebSocket

Vì yêu cầu Real-time, truyền tải dữ liệu qua HTTP REST API sẽ tạo overhead rất lớn. Sử dụng WebSocket là tối ưu.
Hệ thống có 2 loại kết nối WebSocket:
1. **Producer (Edge Device)**: RPi gửi dữ liệu telemetry lên liên tục.
2. **Consumer (Web Dashboard)**: Trình duyệt web kết nối để nhận luồng dữ liệu real-time.

### 2.1. Quản lý Connection (ConnectionManager)
Backend cần một class `ConnectionManager` để:
- Lưu trữ danh sách active websockets của các thiết bị Edge.
- Lưu trữ danh sách active websockets của các Dashboard.
- Cung cấp hàm `broadcast_to_dashboards(data)` để khi Edge A gửi lên, tất cả Dashboard đang xem sẽ nhận được.

### 2.2. Schema Payload (Edge -> Backend)
Gói tin JSON từ Edge Device định nghĩa (dùng Pydantic để validate tại backend nếu cần, nhưng Websocket thường dùng dict trực tiếp cho nhanh):

```json
{
  "device_id": "rpi_01",
  "timestamp": 1690001234.567,
  "metrics": {
    "ear": 0.18,
    "mar": 0.22,
    "perclos": 0.26
  },
  "status": "DROWSY_CRITICAL",
  "alerts": ["EAR_DROP", "HIGH_PERCLOS"]
}
```

### 2.3. Logic Xử lý tại Backend (API Endpoints)
1. **`WS /ws/edge/{device_id}`**: 
   - Nhận kết nối từ Raspberry Pi.
   - Hàm lặp `while True`: `await websocket.receive_json()`.
   - Lưu trạng thái vào Redis hoặc Memory nếu cần.
   - Gọi hàm `manager.broadcast_to_dashboards(data)`.

2. **`WS /ws/dashboard`**:
   - Nhận kết nối từ Frontend Web.
   - Nhận các gói tin broadcast và chuyển xuống Browser.

3. **`POST /api/logs`** *(Optional)*:
   - Dùng để Edge Device gửi các bản ghi (log) quan trọng dạng REST thay vì WS (vd khi trạng thái vượt ngưỡng CRITICAL, gửi ảnh snapshot bằng Base64 lên để lưu lại làm bằng chứng).
   - Backend sẽ lưu vào SQLite hoặc JSON file.

## 3. Quản lý trạng thái và rủi ro rớt mạng
- **Ngắt kết nối ngẫu nhiên**: Backend phải bọc trong khối `try...except WebSocketDisconnect` để dọn dẹp các connection chết, tránh rò rỉ bộ nhớ.
- **Heartbeat (Ping/Pong)**: FastAPI/websockets tự động hỗ trợ tính năng này, cấu hình thời gian timeout khoảng 30s để phát hiện RPi mất mạng và hiển thị "OFFLINE" trên web.
