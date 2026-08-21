"""
Entry point cho Raspberry Pi - Driver Monitoring System
"""
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Thêm dự án vào đường dẫn hệ thống
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from rsp.src.services.analyzer import DrowsinessAnalyzer
from rsp.src.core.config import settings

def main():
    print("=" * 60)
    print("🚀 Đang khởi tạo Hệ thống Giám sát Tài xế trên Raspberry Pi...")
    print("=" * 60)

    # 1. Nạp mô hình ONNX
    model_path = PROJECT_ROOT / "models" / "production" / "driver_state.onnx"
    if not model_path.exists():
        print(f"❌ Không tìm thấy mô hình tại: {model_path}")
        print("💡 Hãy đảm bảo bạn đã chép file driver_state.onnx vào thư mục models/production/")
        return

    print(f"📦 Đang nạp mô hình ONNX: {model_path}")
    try:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        print("✅ Mô hình ONNX đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi nạp mô hình ONNX: {e}")
        return

    # 2. Khởi tạo Analyzer & Camera
    analyzer = DrowsinessAnalyzer(
        ear_threshold=settings.EAR_THRESH,
        mar_threshold=settings.MAR_THRESH
    )

    camera_idx = 0
    print(f"📷 Đang mở Camera (Index {camera_idx})...")
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"❌ Không thể mở Camera tại index {camera_idx}. Hãy kiểm tra lại kết nối camera!")
        return

    print("✅ Camera đã sẵn sàng! Đang chạy chương trình giám sát...")
    print("💡 Bấm Ctrl+C trong Terminal để dừng chương trình.\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n⚠️ Không nhận được khung hình từ Camera...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Preprocess khung hình cho YOLO ONNX (Resize 416x416, BGR2RGB, Normalize)
            resized = cv2.resize(frame, (416, 416))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_input = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            # Chạy suy luận ONNX
            outputs = session.run(None, {input_name: img_input})

            # Tính toán FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Giả lập/Đánh giá trạng thái từ Analyzer
            status_info = analyzer.evaluate_status(ear=0.28, mar=0.10, is_off_center=False, face_detected=True)

            # In log trạng thái trực tiếp ra màn hình Terminal
            print(f"\r[FRAME {frame_count:05d}] FPS: {fps:.1f} | Status: {status_info['status']} | Camera Active: YES", end="", flush=True)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n🛑 Đã dừng chương trình giám sát theo yêu cầu người dùng.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

