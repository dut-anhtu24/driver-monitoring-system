"""
Logic phân tích trạng thái tài xế (Rule-based & Temporal Filtering)
"""

class DrowsinessAnalyzer:
    """
    Phân tích và theo dõi chuỗi thời gian (Temporal Frames) của các chỉ số EAR, MAR,
    trạng thái mặt lệch vị trí hoặc không có mặt trong camera để đưa ra cảnh báo.
    """

    def __init__(
        self,
        ear_threshold=0.20,
        mar_threshold=0.50,
        drowsy_warning_frames=6,
        drowsy_danger_frames=15,
        yawn_frames=8,
        no_face_frames=20,
        off_center_frames=15
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_warning_frames = drowsy_warning_frames
        self.drowsy_danger_frames = drowsy_danger_frames
        self.yawn_frames = yawn_frames
        self.no_face_frames = no_face_frames
        self.off_center_frames = off_center_frames

        # Bộ đếm khung hình liên tục
        self.consecutive_drowsy_frames = 0
        self.consecutive_yawn_frames = 0
        self.consecutive_no_face_frames = 0
        self.consecutive_off_center_frames = 0

    def evaluate_status(self, ear: float, mar: float, is_off_center: bool = False, face_detected: bool = True) -> dict:
        """
        Đánh giá trạng thái tài xế.
        Trả về dictionary gồm status, color (BGR), và các counters.
        """
        # 1. Trường hợp không tìm thấy mặt trong camera
        if not face_detected or ear is None or mar is None:
            self.consecutive_no_face_frames += 1
            self.consecutive_drowsy_frames = 0
            self.consecutive_yawn_frames = 0
            self.consecutive_off_center_frames = 0

            if self.consecutive_no_face_frames >= self.no_face_frames:
                return {
                    "status": "NO_FACE_WARNING",
                    "color": (255, 0, 255),  # Tím đậm
                    "drowsy_frames": 0,
                    "yawn_frames": 0,
                    "no_face_frames": self.consecutive_no_face_frames,
                    "off_center_frames": 0
                }
            else:
                return {
                    "status": "SEARCHING_FACE",
                    "color": (128, 128, 128),  # Màu xám
                    "drowsy_frames": 0,
                    "yawn_frames": 0,
                    "no_face_frames": self.consecutive_no_face_frames,
                    "off_center_frames": 0
                }

        # Đã phát hiện mặt
        self.consecutive_no_face_frames = 0

        # 2. Kiểm tra mặt bị lệch vị trí camera
        if is_off_center:
            self.consecutive_off_center_frames += 1
        else:
            self.consecutive_off_center_frames = 0

        # 3. Kiểm tra EAR (Mắt nhắm/mở)
        if ear < self.ear_threshold:
            self.consecutive_drowsy_frames += 1
        else:
            self.consecutive_drowsy_frames = 0

        # 4. Kiểm tra MAR (Ngáp)
        if mar > self.mar_threshold:
            self.consecutive_yawn_frames += 1
        else:
            self.consecutive_yawn_frames = 0

        # 5. Đánh giá ưu tiên mức độ cảnh báo
        if self.consecutive_drowsy_frames >= self.drowsy_danger_frames:
            status = "DROWSY_DANGER"
            color = (0, 0, 255)  # Đỏ
        elif self.consecutive_off_center_frames >= self.off_center_frames:
            status = "ADJUST_CAMERA_POSITION"
            color = (255, 0, 180)  # Tím hồng
        elif self.consecutive_drowsy_frames >= self.drowsy_warning_frames:
            status = "DROWSY_WARNING"
            color = (0, 165, 255)  # Cam
        elif self.consecutive_yawn_frames >= self.yawn_frames:
            status = "YAWNING"
            color = (0, 255, 255)  # Vàng
        else:
            status = "NORMAL"
            color = (0, 255, 0)  # Xanh lá

        return {
            "status": status,
            "color": color,
            "drowsy_frames": self.consecutive_drowsy_frames,
            "yawn_frames": self.consecutive_yawn_frames,
            "no_face_frames": 0,
            "off_center_frames": self.consecutive_off_center_frames
        }

    def reset(self):
        """Reset toàn bộ bộ đếm."""
        self.consecutive_drowsy_frames = 0
        self.consecutive_yawn_frames = 0
        self.consecutive_no_face_frames = 0
        self.consecutive_off_center_frames = 0

