import os
import cv2
import glob
import shutil
import random
import yaml
import albumentations as A
from tqdm import tqdm
from collections import defaultdict

# --- CẤU HÌNH ---
RAW_DIRS = ['../../dataset/raw/train', '../../dataset/raw/valid']
OUTPUT_DIR = '../../dataset/processed'

VALID_CLASSES = {4: 0, 5: 0, 6: 0, 7: 0}

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# =========================================================================
# BƯỚC 1: DATASET AUDIT & PHÂN NHÓM
# =========================================================================
print("Đang quét và phân loại hồ sơ ảnh...")
grouped_data = defaultdict(list)
total_images = 0

for split_dir in RAW_DIRS:
    img_files = glob.glob(os.path.join(split_dir, 'images', '*.*'))
    for img_path in img_files:
        if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(split_dir, 'labels', base_name + '.txt')
        
        face_class = -1
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        if c_id in VALID_CLASSES:
                            face_class = c_id
                            break
        
        grouped_data[face_class].append((img_path, label_path))
        total_images += 1

print(f"Tổng số ảnh thu thập: {total_images}")

# =========================================================================
# BƯỚC 2: IDENTITY-AWARE DATASET SPLIT (80% Train, 10% Valid, 10% Test)
# =========================================================================
train_data, valid_data, test_data = [], [], []
random.seed(42)

for cls_id, items in grouped_data.items():
    random.shuffle(items)
    split1 = int(len(items) * 0.8)
    split2 = int(len(items) * 0.9)
    train_data.extend(items[:split1])
    valid_data.extend(items[split1:split2])
    test_data.extend(items[split2:])

print(f"\nĐã chia Stratified: {len(train_data)} Train, {len(valid_data)} Valid, {len(test_data)} Test")

# =========================================================================
# BƯỚC 3: LABEL TRANSFORMATION & AUGMENTATION
# =========================================================================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def process_and_save(data_list, split_name, apply_aug=False):
    for img_path, label_path in tqdm(data_list, desc=f"Xử lý tập {split_name}"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        # BƯỚC LỌC & GỘP
                        if c_id in VALID_CLASSES:
                            new_c_id = VALID_CLASSES[c_id]
                            bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                            class_labels.append(new_c_id)
        
        if len(bboxes) == 0:
            continue
            
        out_img_path = os.path.join(OUTPUT_DIR, split_name, 'images', f"{base_name}.jpg")
        out_lbl_path = os.path.join(OUTPUT_DIR, split_name, 'labels', f"{base_name}.txt")
        cv2.imwrite(out_img_path, img)
        with open(out_lbl_path, 'w') as f:
            for bbox, cls in zip(bboxes, class_labels):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")
                
        if apply_aug:
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
                if len(aug_bboxes) > 0:
                    aug_base_name = f"{base_name}_aug"
                    aug_img_path = os.path.join(OUTPUT_DIR, split_name, 'images', f"{aug_base_name}.jpg")
                    aug_lbl_path = os.path.join(OUTPUT_DIR, split_name, 'labels', f"{aug_base_name}.txt")
                    
                    cv2.imwrite(aug_img_path, aug_img)
                    with open(aug_lbl_path, 'w') as f:
                        for bbox, cls in zip(aug_bboxes, aug_labels):
                            f.write(f"{cls} {' '.join(map(str, bbox))}\n")
            except Exception:
                pass

process_and_save(valid_data, 'valid', apply_aug=False)
process_and_save(test_data, 'test', apply_aug=False)
process_and_save(train_data, 'train', apply_aug=True)

# Tạo file cấu hình data.yaml
yaml_content = {
    'path': f'../dataset/processed',
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': 1,
    'names': ['face']
}
with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)
    
print("\nHoàn thành 7 bước Enterprise Pipeline! Dữ liệu sẵn sàng tại dataset/processed/")
