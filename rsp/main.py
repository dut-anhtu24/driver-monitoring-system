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
from rsp.src.services.feature_extractor import FacialFeatureExtractor
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

    # 2. Khởi tạo Feature Extractor & Analyzer
    extractor = FacialFeatureExtractor()
    if not extractor.has_mp:
        print("⚠️ Lưu ý: Chưa cài đặt 'mediapipe' trên Pi. Vui lòng chạy: pip install mediapipe")

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

    print("✅ Camera đã sẵn sàng! Đang chạy nhận diện mắt/miệng thời gian thực...")
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
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Trích xuất EAR (chỉ số mắt) & MAR (chỉ số miệng) thực tế từ khuôn mặt
            ear, mar, landmarks = extractor.extract_features(img_rgb)
            face_detected = landmarks is not None

            # 2. Đánh giá trạng thái buồn ngủ thực tế bằng DrowsinessAnalyzer
            status_info = analyzer.evaluate_status(
                ear=ear,
                mar=mar,
                is_off_center=False,
                face_detected=face_detected
            )

            # 3. Tính toán FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # In thông số EAR/MAR thực tế lên màn hình Terminal
            ear_str = f"{ear:.3f}" if ear is not None else "N/A"
            mar_str = f"{mar:.3f}" if mar is not None else "N/A"

            print(f"\r[FRAME {frame_count:05d}] FPS: {fps:.1f} | Status: {status_info['status']:<20} | EAR: {ear_str} | MAR: {mar_str}", end="", flush=True)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n🛑 Đã dừng chương trình giám sát theo yêu cầu người dùng.")
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


