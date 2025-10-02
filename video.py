import cv2
import numpy as np
import time
from PyQt5 import QtCore

class VideoThread(QtCore.QThread):
    """Поток для захвата видео с камеры"""
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)
    recognition_result = QtCore.pyqtSignal(str, float)

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.running = True
        self.recognition_enabled = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Не удалось открыть камеру")
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Отражаем изображение по горизонтали для естественного отображения
                frame = cv2.flip(frame, 1)

                if self.recognition_enabled:
                    # Выполняем распознавание
                    processed_frame, name = self.recognizer.process_frame(frame)
                    self.change_pixmap_signal.emit(processed_frame)

                    # Если обнаружено лицо, отправляем результат
                    if name and name != "Unknown":
                        # Находим максимальное сходство для отображения
                        face, box = self.recognizer.detector.detect(frame)
                        if face is not None:
                            features = self.recognizer.model.extract_features(face)
                            best_sim = 0.0
                            for db_features in self.recognizer.db.all_faces().values():
                                sim = self.recognizer._cosine_similarity(features, db_features)
                                if sim > best_sim:
                                    best_sim = sim
                            self.recognition_result.emit(name, best_sim)
                else:
                    # Просто показываем видео без распознавания
                    self.change_pixmap_signal.emit(frame)

            time.sleep(0.03)  # ~30 FPS

        cap.release()

    def stop(self):
        self.running = False
        self.recognition_enabled = False