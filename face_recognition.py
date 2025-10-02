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

    def process_frame(self, frame):
        """Обработка одного кадра для встроенного отображения"""
        face, box = self.detector.detect(frame)
        name = None
        confidence = 0.0

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
                name = best_name
            elif best_sim >= self.low_thr:
                color = (0, 255, 255)
                label = f"{best_name}? ({best_sim:.2f})"
                name = best_name
            else:
                color = (0, 0, 255)
                label = f"Unknown ({best_sim:.2f})"
                name = "Unknown"

            # Рисуем рамку и текст
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            confidence = best_sim

        return frame, name

    # Старые методы для совместимости
    def recognize(self):
        """Устаревший метод - теперь не используется"""
        print("⚠ Этот метод устарел. Используйте process_frame()")

    def stop(self):
        """Заглушка для совместимости"""
        pass