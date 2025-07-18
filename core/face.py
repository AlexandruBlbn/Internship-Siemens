import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def detect_person(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_detection.process(rgb_frame)
            if results.detections:
                return True, len(results.detections)
            return False, 0
        except Exception:
            return False, 0

    def draw_detections(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_detection.process(rgb_frame)
            rgb_frame.flags.writeable = True
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    self.mp_drawing.draw_detection(bgr_frame, detection)
            return bgr_frame
        except Exception:
            return frame