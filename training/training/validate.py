import os
import sys
import yaml
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

# Khắc phục lỗi encode Unicode/Emoji trên console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def check_hardware():
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Đánh giá sử dụng GPU: {gpu_name}")
        return 0
    else:
        print("⚠️ Đánh giá sử dụng CPU")
        return "cpu"


def validate():
    device = check_hardware()

    # Đường dẫn mô hình best.pt sau khi train
    weights_path = PROJECT_ROOT / "models" / "yolo11n_driver_face" / "weights" / "best.pt"
    data_yaml = str(PROJECT_ROOT / "dataset" / "processed" / "data.yaml")

    if not weights_path.exists():
        print(f"❌ Không tìm thấy trọng số mô hình tại: {weights_path}")
        print("💡 Vui lòng hoàn thành quá trình huấn luyện train.py trước!")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Lỗi: Thư viện 'ultralytics' chưa được cài đặt.")
        return

    print(f"\n🎯 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH YOLOV11 NANO TRÊN TẬP TEST...")
    print(f"• Weights: {weights_path}")
    print(f"• Data YAML: {data_yaml}")
    print(f"• Split: test\n")

    model = YOLO(str(weights_path))

    # Evaluate on test set
    metrics = model.val(
        data=data_yaml,
        split="test",
        batch=16,
        imgsz=416,
        device=device
    )

    print("\n=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TEST METRICS)")
    print("=" * 60)
    print(f"• mAP50:       {metrics.box.map50:.4f}")
    print(f"• mAP50-95:    {metrics.box.map:.4f}")
    print(f"• Precision:   {metrics.box.mp:.4f}")
    print(f"• Recall:      {metrics.box.mr:.4f}")
    print(f"• Speed (Inference): {metrics.speed['inference']:.2f} ms/image (~{1000/metrics.speed['inference']:.1f} FPS)")
    print("=" * 60)


if __name__ == "__main__":
    validate()
