import os
import sys
import shutil
import platform
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

    print("🚀 Bắt đầu export mô hình YOLO11...")
    model = YOLO(str(weights_path))

    if platform.system() == "Windows":
        print("⚠️ Lưu ý: LiteRT (TFLite Exporter mới của Ultralytics) hiện chưa hỗ trợ export trực tiếp trên Windows.")
        print("👉 Đang chuyển sang export định dạng ONNX (Format tối ưu chạy mượt trên Raspberry Pi với ONNXRuntime)...")
        try:
            onnx_path = model.export(format="onnx", imgsz=416, simplify=True)
            dest_path = prod_dir / "driver_state.onnx"
            shutil.copy(onnx_path, dest_path)
            print(f"✅ Export ONNX thành công cho Raspberry Pi! Đã lưu mô hình sản phẩm tại: {dest_path}")
            return
        except Exception as e:
            print(f"❌ Lỗi khi export ONNX: {e}")
            return

    try:
        # Export TFLite với int8 quantization (chỉ chạy trên Linux x86 / macOS)
        tflite_path = model.export(format="tflite", imgsz=416, int8=True)
        dest_path = prod_dir / "driver_state.tflite"
        if os.path.exists(tflite_path):
            shutil.copy(tflite_path, dest_path)
            print(f"✅ Export TFLite thành công! Đã lưu mô hình sản phẩm tại: {dest_path}")
        else:
            print(f"⚠️ Kiểm tra kết quả export tại: {tflite_path}")
    except Exception as e:
        print(f"❌ Lỗi export TFLite: {e}")
        print("💡 Gợi ý: Nếu cần dùng TFLite, hãy export bằng WSL2 (Linux) hoặc export trực tiếp trên Raspberry Pi!")

if __name__ == "__main__":
    export_tflite()

