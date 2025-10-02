import cv2
import numpy as np
from face_database import FaceDatabase
from face_model import FaceModel
from face_detector import FaceDetector


class FaceRecognitionSystem:
    def __init__(self, threshold=0.6, low_thr=0.8, high_thr=0.95):
        self.db = FaceDatabase()
        self.model = FaceModel()
        self.detector = FaceDetector()
        self.threshold = threshold
        self.low_thr = low_thr
        self.high_thr = high_thr

    def _cosine_similarity(self, f1, f2):
        return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))

    def recognize(self):
        if not self.db.has_faces():
            print("❌ База пуста!")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        print("🔍 Запуск распознавания (Q/Esc для выхода)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face, box = self.detector.detect(frame)
            if face is not None:
                features = self.model.extract_features(face)
                best_name, best_sim = "Unknown", 0.0

                for db_name, db_features in self.db.all_faces().items():
                    sim = self._cosine_similarity(features, db_features)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = db_name

                (x1, y1, x2, y2) = box
                if best_sim >= self.high_thr:
                    color = (0, 255, 0)
                    label = f"{best_name} ({best_sim:.2f})"
                elif best_sim >= self.low_thr:
                    color = (0, 255, 255)
                    label = f"{best_name}? ({best_sim:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({best_sim:.2f})"
                    best_name = "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Распознавание", frame)
            key = cv2.waitKey(30)
            if key in [27, ord('Q')]:
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        face, box = self.detector.detect(frame)
        name = None
        if face is not None:
            features = self.model.extract_features(face)
            best_name, best_sim = "Unknown", 0.0
            for db_name, db_features in self.db.all_faces().items():
                s = self._cosine_similarity(features, db_features)
                if s > best_sim and s > self.threshold:
                    best_name, best_sim = db_name, s

            (x1, y1, x2, y2) = box
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{best_name} ({best_sim:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            name = best_name
        return frame, name
