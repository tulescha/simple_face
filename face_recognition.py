import cv2
import numpy as np
import time
from face_database import FaceDatabase
from face_model import FaceModel
from face_detector import FaceDetector

class FaceRecognitionSystem:
    def __init__(self, threshold=0.6):
        self.db = FaceDatabase()
        self.model = FaceModel()
        self.detector = FaceDetector()
        self.threshold = threshold

    def _cosine_similarity(self, f1, f2):
        return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))

    def _setup_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap if cap.isOpened() else None

    def register_face(self, name):
        print(f"\n📷 Регистрация: {name}")
        cap = self._setup_camera()
        if not cap:
            print("❌ Камера не найдена!")
            return

        features_list = []
        while len(features_list) < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            face, box = self.detector.detect(frame)
            if face is not None:
                features = self.model.extract_features(face)
                features_list.append(features)
                (x1, y1, x2, y2) = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Регистрация", frame)
            cv2.waitKey(500)

        cap.release()
        cv2.destroyAllWindows()
        avg_features = np.mean(features_list, axis=0)
        self.db.add_face(name, avg_features)
        print(f"✅ Пользователь {name} добавлен в базу")

    def recognize(self):
        if not self.db.has_faces():
            print("❌ База пуста!")
            return

        cap = self._setup_camera()
        if not cap:
            return

        print("🔍 Запуск распознавания (Q/Esc для выхода)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face, box = self.detector.detect(frame)
            if face is not None:
                features = self.model.extract_features(face)
                name, sim = "Unknown", 0.0
                for db_name, db_features in self.db.all_faces().items():
                    s = self._cosine_similarity(features, db_features)
                    if s > sim and s > self.threshold:
                        name, sim = db_name, s

                (x1, y1, x2, y2) = box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{name} ({sim:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Распознавание", frame)
            key = cv2.waitKey(30)
            if key in [27, ord('q'), ord('Q')]:
                break

        cap.release()
        cv2.destroyAllWindows()
