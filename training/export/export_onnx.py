import os
import sys
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

def export_onnx():
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

    print("🚀 Bắt đầu export mô hình YOLO11 sang ONNX format...")
    model = YOLO(str(weights_path))
    
    # Export ONNX
    onnx_path = model.export(format="onnx", imgsz=416, simplify=True)
    
    # Copy to models/production
    dest_path = prod_dir / "driver_state.onnx"
    shutil.copy(onnx_path, dest_path)
    print(f"✅ Export ONNX thành công! Đã lưu mô hình sản phẩm tại: {dest_path}")

if __name__ == "__main__":
    export_onnx()
