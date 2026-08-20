import os
import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path

# Thêm đường dẫn gốc dự án vào sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

from training.features.feature_extraction import FacialFeatureExtractor
from rsp.src.services.analyzer import DrowsinessAnalyzer
from ultralytics import YOLO


import threading

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlarmPlayer:
    """
    Phát tiếng còi báo động (Audio Alarm Beep) trên background thread
    giúp cảnh báo tài xế tức thì mà không gây lag luồng video real-time.
    """
    def __init__(self):
        self.is_playing = False
        self.last_played = 0

    def trigger_alarm(self, freq=2600, duration=300, cooldown=0.1):
        now = time.time()
        if self.is_playing or (now - self.last_played < cooldown):
            return

        def _play():
            self.is_playing = True
            try:
                if HAS_WINSOUND:
                    winsound.Beep(freq, duration)
                else:
                    print('\a', end='', flush=True)
            except Exception:
                pass
            finally:
                self.is_playing = False
                self.last_played = time.time()

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()


def run_demo(source=0, conf_threshold=0.30):
    """
    Chạy thử nghiệm thời gian thực Cảnh báo Buồn ngủ & Ngáp trên Webcam/Video.
    """
    # 1. Đường dẫn mô hình YOLO11
    weights_path = PROJECT_ROOT / "models" / "yolo11n_driver_face" / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"❌ Không tìm thấy weights tại: {weights_path}")
        return

    print("🚀 Khởi tạo hệ thống Driver Drowsiness Detection...")
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # 2. Khởi tạo mô hình YOLO11, Feature Extractor, Analyzer và Alarm Player
    yolo_model = YOLO(str(weights_path))
    feature_extractor = FacialFeatureExtractor()
    analyzer = DrowsinessAnalyzer(
        ear_threshold=0.20,
        mar_threshold=0.50,
        drowsy_warning_frames=6,
        drowsy_danger_frames=15,
        yawn_frames=8
    )
    alarm_player = AlarmPlayer()

    # 3. Mở luồng camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Không thể mở nguồn camera/video: {source}")
        return

    print("✅ Hệ thống đã sẵn sàng! Bấm 'q' hoặc 'Esc' để thoát.")

    prev_time = time.time()
    fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Kết thúc luồng video hoặc không đọc được camera.")
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # A. YOLO11 Face Detection (Dùng conf_threshold = 0.30 để không rớt mặt khi ngáp)
        results = yolo_model.predict(frame, conf=conf_threshold, device=device, verbose=False)
        boxes = results[0].boxes

        ear, mar = None, None
        crop_x1, crop_y1 = 0, 0
        landmarks = None
        detected_face = False
        is_off_center = False

        if len(boxes) > 0:
            detected_face = True
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # 1. Tính độ lệch tâm khuôn mặt so với trung tâm camera
            face_cx = (x1 + x2) / 2.0
            face_cy = (y1 + y2) / 2.0
            offset_x = abs(face_cx - (w / 2.0)) / w
            offset_y = abs(face_cy - (h / 2.0)) / h

            # Ngưỡng lệch: lệch quá 25% chiều ngang hoặc 28% chiều dọc
            if offset_x > 0.25 or offset_y > 0.28:
                is_off_center = True

            # 2. Bounding box padding rộng (20% x, 35% y) để bao trọn cằm khi ngáp
            pad_x = int((x2 - x1) * 0.20)
            pad_y = int((y2 - y1) * 0.35)
            crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            crop_x2, crop_y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            face_roi_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

            if face_roi_rgb.size > 0:
                ear, mar, landmarks = feature_extractor.extract_features(face_roi_rgb)
                if landmarks:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

        # B. Fallback: Nếu YOLO không bắt được hoặc ROI bị rớt, quét MediaPipe trên toàn bộ khung hình
        if landmarks is None and detected_face:
            ear, mar, landmarks = feature_extractor.extract_features(frame_rgb)
            crop_x1, crop_y1 = 0, 0

        # C. Vẽ mốc mắt và miệng nếu có
        if landmarks:
            for idx in FacialFeatureExtractor.LEFT_EYE_INDICES + FacialFeatureExtractor.RIGHT_EYE_INDICES:
                lx, ly = landmarks[idx]
                cv2.circle(frame, (crop_x1 + lx, crop_y1 + ly), 2, (255, 255, 0), -1)

            for idx in FacialFeatureExtractor.MOUTH_INDICES:
                lx, ly = landmarks[idx]
                cv2.circle(frame, (crop_x1 + lx, crop_y1 + ly), 2, (0, 255, 255), -1)

        # D. Đánh giá trạng thái (Bao gồm kiểm tra lệch camera / mất mặt)
        status_info = analyzer.evaluate_status(ear, mar, is_off_center=is_off_center, face_detected=detected_face)
        color = status_info["color"]

        # D. Trực quan hóa bảng điều khiển UI
        # 1. Tính FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        # 2. Vẽ khung nét đứt định vị góc camera lý tưởng ở giữa màn hình
        guide_x1, guide_y1 = int(w * 0.22), int(h * 0.12)
        guide_x2, guide_y2 = int(w * 0.78), int(h * 0.88)
        cv2.rectangle(frame, (guide_x1, guide_y1), (guide_x2, guide_y2), (100, 100, 100), 1)

        # 3. Thanh trạng thái Cảnh báo phía trên (Banner)
        status_text = status_info["status"]
        banner_color = status_info["color"]
        
        cv2.rectangle(frame, (0, 0), (w, 50), banner_color, -1)
        cv2.putText(
            frame,
            f"DRIVER STATUS: {status_text}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 4. Hiển thị thông số chi tiết ở góc trên trái
        ear_str = f"{ear:.2f}" if ear is not None else "N/A"
        mar_str = f"{mar:.2f}" if mar is not None else "N/A"

        info_bg = np.zeros((100, 240, 3), dtype=np.uint8)
        cv2.putText(info_bg, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(info_bg, f"EAR: {ear_str} (Th: 0.20)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_bg, f"MAR: {mar_str} (Th: 0.50)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Overlay info box onto frame
        frame[60:160, 20:260] = cv2.addWeighted(frame[60:160, 20:260], 0.3, info_bg, 0.7, 0)

        # 5. Phát Âm thanh Cảnh báo & Hiển thị chữ nhấp nháy
        if status_text == "DROWSY_DANGER":
            # Còi báo động dồn dập 2600Hz khi buồn ngủ
            alarm_player.trigger_alarm(freq=2600, duration=300)

            if int(curr_time * 4) % 2 == 0:
                cv2.putText(
                    frame,
                    "!!! WARNING: WAKE UP !!!",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )
        elif status_text in ["NO_FACE_WARNING", "ADJUST_CAMERA_POSITION"]:
            # Còi báo nhè nhẹ 1800Hz yêu cầu chỉnh camera/tư thế
            alarm_player.trigger_alarm(freq=1800, duration=200)

            msg = "!!! PLEASE ADJUST CAMERA OR POSTURE !!!" if status_text == "ADJUST_CAMERA_POSITION" else "!!! NO FACE DETECTED - CHECK CAMERA !!!"
            if int(curr_time * 3) % 2 == 0:
                cv2.putText(
                    frame,
                    msg,
                    (w // 2 - 280, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        cv2.imshow("Driver Drowsiness Monitoring System (YOLO11 + MediaPipe)", frame)

        # Bấm 'q' hoặc 'Esc' để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    feature_extractor.close()
    print("👋 Đã kết thúc thử nghiệm!")


if __name__ == "__main__":
    run_demo(source=0)
