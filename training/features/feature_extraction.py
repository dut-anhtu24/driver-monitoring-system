import cv2
import numpy as np
import mediapipe as mp


class FacialFeatureExtractor:
    """
    Trích xuất 468 điểm mốc khuôn mặt bằng MediaPipe Face Mesh 
    và tính toán các chỉ số EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio).
    """

    # Chỉ số điểm mốc Mắt Trái (Left Eye)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Chỉ số điểm mốc Mắt Phải (Right Eye)
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Chỉ số điểm mốc Miệng (Mouth)
    MOUTH_INDICES = [61, 291, 13, 14, 81, 178, 311, 402]

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    @staticmethod
    def _euclidean_distance(point1, point2):
        """Tính khoảng cách Euclidean giữa 2 điểm 2D/3D."""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def calculate_ear(self, landmarks, eye_indices):
        """
        Tính chỉ số EAR (Eye Aspect Ratio) cho 1 mắt.
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vert_dist1 = self._euclidean_distance(p2, p6)
        vert_dist2 = self._euclidean_distance(p3, p5)
        horiz_dist = self._euclidean_distance(p1, p4)

        if horiz_dist == 0:
            return 0.0

        ear = (vert_dist1 + vert_dist2) / (2.0 * horiz_dist)
        return float(ear)

    def calculate_mar(self, landmarks):
        """
        Tính chỉ số MAR (Mouth Aspect Ratio) cho miệng.
        MAR = (||p13 - p14|| + ||p81 - p178|| + ||p311 - p402||) / (2 * ||p61 - p291||)
        """
        p_left = landmarks[61]
        p_right = landmarks[291]
        p_top_inner = landmarks[13]
        p_bottom_inner = landmarks[14]
        p_top_l = landmarks[81]
        p_bottom_l = landmarks[178]
        p_top_r = landmarks[311]
        p_bottom_r = landmarks[402]

        v1 = self._euclidean_distance(p_top_inner, p_bottom_inner)
        v2 = self._euclidean_distance(p_top_l, p_bottom_l)
        v3 = self._euclidean_distance(p_top_r, p_bottom_r)
        h = self._euclidean_distance(p_left, p_right)

        if h == 0:
            return 0.0

        mar = (v1 + v2 + v3) / (2.0 * h)
        return float(mar)

    def extract_features(self, image_rgb):
        """
        Trích xuất điểm mốc và tính chỉ số EAR/MAR từ khung ảnh RGB.
        Trả về (avg_ear, mar, landmarks_list).
        """
        h, w, _ = image_rgb.shape
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        # Lấy khuôn mặt đầu tiên
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_pixel = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        left_ear = self.calculate_ear(landmarks_pixel, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_ear(landmarks_pixel, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        mar = self.calculate_mar(landmarks_pixel)

        return avg_ear, mar, landmarks_pixel

    def close(self):
        self.face_mesh.close()
