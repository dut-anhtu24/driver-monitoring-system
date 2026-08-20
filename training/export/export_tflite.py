import os
import sys
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

def export_tflite():
    weights_path = PROJECT_ROOT / "models" / "yolo11n_driver_face" / "weights" / "best.pt"
    prod_dir = PROJECT_ROOT / "models" / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"❌ Không tìm thấy weights tại: {weights_path}")
        print("💡 Hãy chạy huấn luyện trước!")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Lỗi: Chưa cài đặt ultralytics.")
        return

    print("🚀 Bắt đầu export mô hình YOLO11 sang TFLite INT8 Quantized (Tối ưu cho Edge AI)...")
    model = YOLO(str(weights_path))
    
    # Export TFLite with int8 quantization for Edge AI CPU acceleration
    tflite_path = model.export(format="tflite", imgsz=416, int8=True)
    
    # Copy to models/production
    dest_path = prod_dir / "driver_state.tflite"
    if os.path.exists(tflite_path):
        shutil.copy(tflite_path, dest_path)
        print(f"✅ Export TFLite thành công! Đã lưu mô hình sản phẩm tại: {dest_path}")
    else:
        print(f"⚠️ Kiểm tra kết quả export tại: {tflite_path}")

if __name__ == "__main__":
    export_tflite()
