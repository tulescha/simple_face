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

    def recognize(self, high_thr=0.95, low_thr=0.80):
        """
        Распознавание лиц с 3 уровнями:
        - >= high_thr: уверенный матч (зелёный)
        - >= low_thr: сомнительный матч (жёлтый)
        - ниже: Unknown (красный)
        """
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
                best_name, best_sim = "Unknown", 0.0

                # ищем наиболее похожее лицо
                for db_name, db_features in self.db.all_faces().items():
                    sim = self._cosine_similarity(features, db_features)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = db_name

                (x1, y1, x2, y2) = box

                # логика порогов
                if best_sim >= high_thr:
                    color = (0, 255, 0)  # зелёный
                    label = f"{best_name} ({best_sim:.2f})"
                elif best_sim >= low_thr:
                    color = (0, 255, 255)  # жёлтый
                    label = f"{best_name}? ({best_sim:.2f})"
                else:
                    color = (0, 0, 255)  # красный
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
