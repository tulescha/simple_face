import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from ui_main import Ui_MainWindow
from face_recognition import FaceRecognitionSystem
import threading
import time


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


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.recognizer = FaceRecognitionSystem()

        # Создаем QLabel для отображения видео, если его нет в UI
        if not hasattr(self.ui, 'video_label'):
            self.video_label = QtWidgets.QLabel(self.ui.centralwidget)
            self.video_label.setGeometry(QtCore.QRect(300, 50, 480, 360))
            self.video_label.setStyleSheet("background-color: black; border: 2px solid gray;")
            self.video_label.setAlignment(QtCore.Qt.AlignCenter)
            self.video_label.setText("Камера неактивна")
            self.video_label.setMinimumSize(480, 360)

        # Поток для видео
        self.video_thread = VideoThread(self.recognizer)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.recognition_result.connect(self.on_recognition_result)

        # Подключаем кнопки (без кнопки остановки камеры)
        self.ui.btnRegister.clicked.connect(self.register_user)
        self.ui.btnRecognize.clicked.connect(self.toggle_recognition)
        self.ui.btnShowDB.clicked.connect(self.show_db)
        self.ui.btnThresholds.clicked.connect(self.set_thresholds)
        self.ui.btnExit.clicked.connect(self.close)

        # Лог-виджет
        self.log_widget = self.ui.textBrowser

        # Статус распознавания
        self.recognition_active = False

        # Автоматически запускаем камеру при старте
        self.start_camera()

        self.log("🚀 Система распознавания лиц запущена")
        self.log("👉 Нажмите 'Запуск распознавания' для начала")

    def start_camera(self):
        """Автоматический запуск камеры при старте"""
        if not self.video_thread.isRunning():
            self.video_thread.start()
            self.log("📹 Камера запущена")

    def update_image(self, cv_img):
        """Обновление изображения в QLabel"""
        qt_img = self.convert_cv_qt(cv_img)
        if hasattr(self.ui, 'video_label'):
            self.ui.video_label.setPixmap(qt_img)
        else:
            self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Конвертация OpenCV image в QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(480, 360, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def on_recognition_result(self, name, confidence):
        """Обработка результата распознавания"""
        if confidence >= self.recognizer.high_thr:
            self.log(f"✅ Распознан: {name} ({confidence:.2f})")
        elif confidence >= self.recognizer.low_thr:
            self.log(f"⚠ Возможно: {name}? ({confidence:.2f})")
        else:
            self.log(f"❌ Неизвестный ({confidence:.2f})")

    def log(self, text):
        """Добавление сообщения в лог"""
        if self.log_widget:
            self.log_widget.append(text)
            scrollbar = self.log_widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        print(f"[LOG] {text}")

    def register_user(self):
        """Регистрация нового пользователя"""
        if self.recognition_active:
            self.log("❌ Сначала остановите распознавание!")
            return

        name, ok = QtWidgets.QInputDialog.getText(self, "Регистрация", "Введите имя:")
        if ok and name:
            if name in self.recognizer.db.all_faces():
                self.log(f"⚠ Пользователь {name} уже существует!")
                return

            self.log(f"🔍 Регистрация {name}... Смотрите в камеру!")

            # Ждем обнаружения лица
            face_detected = False
            for attempt in range(50):
                # В реальном приложении здесь должен быть код для захвата кадра и детекции лица
                time.sleep(0.1)

                # Для демонстрации имитируем обнаружение лица
                if attempt == 25:
                    face_detected = True
                    break

            if face_detected:
                # Создаем фиктивные features (в реальном приложении извлекаем из лица)
                dummy_features = np.random.rand(512).astype(np.float32)
                self.recognizer.db.add_face(name, dummy_features)
                self.log(f"🎉 Пользователь {name} успешно зарегистрирован!")
            else:
                self.log("❌ Не удалось обнаружить лицо для регистрации")

    def toggle_recognition(self):
        """Включение/выключение распознавания"""
        if not self.recognizer.db.has_faces():
            self.log("❌ База пуста! Сначала зарегистрируйте пользователей.")
            return

        if not self.recognition_active:
            # Запускаем распознавание
            self.video_thread.recognition_enabled = True
            self.recognition_active = True
            self.ui.btnRecognize.setText("Остановить распознавание")
            self.ui.btnRecognize.setStyleSheet("background-color: orange;")
            self.log("🔍 Распознавание запущено...")
        else:
            # Останавливаем распознавание
            self.video_thread.recognition_enabled = False
            self.recognition_active = False
            self.ui.btnRecognize.setText("Запуск распознавания")
            self.ui.btnRecognize.setStyleSheet("")
            self.log("🛑 Распознавание остановлено")

    def show_db(self):
        """Показать список зарегистрированных пользователей"""
        faces = self.recognizer.db.all_faces()
        if faces:
            user_list = "\n".join([f"• {name}" for name in faces.keys()])
            self.log(f"👥 Зарегистрированные пользователи ({len(faces)}):\n{user_list}")
        else:
            self.log("❌ База данных пуста")

    def set_thresholds(self):
        """Установка порогов распознавания"""
        current_low = self.recognizer.low_thr
        current_high = self.recognizer.high_thr

        low, ok1 = QtWidgets.QInputDialog.getDouble(
            self, "Нижний порог",
            "Порог для 'сомнительного' распознавания:",
            current_low, 0.1, 0.9, 2
        )
        if not ok1:
            return

        high, ok2 = QtWidgets.QInputDialog.getDouble(
            self, "Верхний порог",
            "Порог для 'уверенного' распознавания:",
            current_high, low, 1.0, 2
        )
        if not ok2:
            return

        self.recognizer.low_thr = low
        self.recognizer.high_thr = high
        self.log(f"✅ Пороги обновлены:")
        self.log(f"   • Сомнительное распознавание: ≥ {low:.2f}")
        self.log(f"   • Уверенное распознавание: ≥ {high:.2f}")

    def closeEvent(self, event):
        """Обработчик закрытия окна"""
        self.video_thread.stop()
        if self.video_thread.isRunning():
            self.video_thread.wait(2000)  # Ждем до 2 секунд
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = MainApp()
    window.setWindowTitle("Система распознавания лиц")
    window.show()

    sys.exit(app.exec_())