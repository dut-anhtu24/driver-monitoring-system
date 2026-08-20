import os
import sys
import yaml
import torch
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import dễ dàng
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

# Khắc phục lỗi encode Unicode/Emoji trên console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def check_hardware():
    """
    Kiểm tra và hiển thị thông tin phần cứng (GPU/CPU) chi tiết.
    Trả về device string ('0' hoặc 'cpu').
    """
    print("=" * 60)
    print("🔍 KIỂM TRA CẤU HÌNH PHẦN CỨNG (HARDWARE DETECTION)")
    print("=" * 60)
    print(f"📌 PyTorch Version: {torch.__version__}")
    
    is_cuda_available = torch.cuda.is_available()
    print(f"📌 CUDA Available: {is_cuda_available}")

    if is_cuda_available:
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
        print(f"🚀 THIẾT BỊ SỬ DỤNG: GPU (NVIDIA)")
        print(f"   • Tên GPU: {gpu_name}")
        print(f"   • Số lượng GPU: {device_count}")
        print(f"   • CUDA Version: {cuda_version}")
        print(f"   • Tổng VRAM: {vram_gb:.2f} GB")
        print("=" * 60)
        return 0  # Trả về ID GPU đầu tiên cho Ultralytics
    else:
        cpu_count = os.cpu_count()
        print(f"⚠️ THIẾT BỊ SỬ DỤNG: CPU")
        print(f"   • Số nhân CPU: {cpu_count}")
        print("   • Lưu ý: Huấn luyện trên CPU sẽ chậm hơn đáng kể so với GPU.")
        print("=" * 60)
        return "cpu"


def train():
    # 1. Kiểm tra phần cứng
    device = check_hardware()

    # 2. Đọc cấu hình huấn luyện
    config_path = PROJECT_ROOT / "configs" / "training.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    data_yaml = str(PROJECT_ROOT / "dataset" / "processed" / "data.yaml")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Không tìm thấy tệp data.yaml tại: {data_yaml}")

    print("\n🎯 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN YOLO11...")
    print(f"• Baseline Model: {model_name}")
    print(f"• Dataset Config: {data_yaml}")
    print(f"• Epochs: {cfg['train']['epochs']}")
    print(f"• Image Size: {cfg['train']['imgsz']}")
    print(f"• Batch Size: {cfg['train']['batch_size']}")
    print(f"• Device: {device}\n")

    # 3. Import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Lỗi: Thư viện 'ultralytics' chưa được cài đặt.")
        print("💡 Hãy chạy: pip install ultralytics")
        sys.exit(1)

    # 4. Initialize Model
    model = YOLO(model_name)

    # 5. Execute Training
    results = model.train(
        data=data_yaml,
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["train"]["imgsz"],
        batch=cfg["train"]["batch_size"],
        workers=cfg["train"]["workers"],
        device=device,
        patience=cfg["train"]["patience"],
        project=str(PROJECT_ROOT / "models"),
        name=cfg["train"]["name"],
        exist_ok=cfg["train"]["exist_ok"],
        lr0=cfg["optimizer"]["lr0"],
        lrf=cfg["optimizer"]["lrf"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"]
    )

    print("\n✅ HOÀN THÀNH HUẤN LUYỆN!")
    print(f"📦 Mô hình và log đã được lưu tại: {results.save_dir}")


if __name__ == "__main__":
    train()
