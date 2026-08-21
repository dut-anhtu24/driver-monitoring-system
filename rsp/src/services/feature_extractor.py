import cv2
import numpy as np

try:
    import mediapipe as mp
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except ImportError:
        try:
            import mediapipe.solutions.face_mesh as mp_face_mesh
        except ImportError:
            mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

class FacialFeatureExtractor:
    """
    Trích xuất 468 điểm mốc khuôn mặt bằng MediaPipe Face Mesh
    và tính toán các chỉ số EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio).
    """

    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [61, 291, 13, 14, 81, 178, 311, 402]

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.has_mp = HAS_MEDIAPIPE
        if self.has_mp:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            except Exception as e:
                print(f"⚠️ Lỗi khởi tạo MediaPipe FaceMesh: {e}")
                self.has_mp = False

    @staticmethod
    def _euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def calculate_ear(self, landmarks, eye_indices):
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
        if not self.has_mp:
            return None, None, None

        h, w, _ = image_rgb.shape
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks_pixel = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        left_ear = self.calculate_ear(landmarks_pixel, self.LEFT_EYE_INDICES)
        right_ear = self.calculate_ear(landmarks_pixel, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        mar = self.calculate_mar(landmarks_pixel)

        return avg_ear, mar, landmarks_pixel

    def close(self):
        if self.has_mp:
            self.face_mesh.close()
