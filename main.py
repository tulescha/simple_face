import cv2
import torch
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class WorkingFaceRecognition:
    def __init__(self, threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используемое устройство: {self.device}")

        # Простой детектор лиц на основе OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.threshold = threshold
        self.embeddings_db = {}
        self.load_database()

    def load_database(self):
        """Загрузка базы данных эмбеддингов"""
        if os.path.exists('face_embeddings.pkl'):
            with open('face_embeddings.pkl', 'rb') as f:
                self.embeddings_db = pickle.load(f)
            print(f"Загружено {len(self.embeddings_db)} эмбеддингов из базы данных")
        else:
            print("База данных не найдена. Сначала зарегистрируйте свое лицо.")

    def save_database(self):
        """Сохранение базы данных"""
        with open('face_embeddings.pkl', 'wb') as f:
            pickle.dump(self.embeddings_db, f)
        print("База данных сохранена")

    def setup_camera(self):
        """Настройка камеры с обработкой ошибок"""
        # Пробуем разные индексы камер
        for camera_index in [0, 1]:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Камера {camera_index} успешно подключена")
                    return cap
            cap.release()

        print("❌ Не удалось найти работающую камеру!")
        return None

    def extract_face_features(self, face_image):
        """
        Упрощенное извлечение признаков из лица
        В реальном проекте здесь должна быть нейросеть
        """
        try:
            # Преобразуем в grayscale и изменяем размер
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (64, 64))

            # Нормализуем
            normalized_face = resized_face / 255.0

            # Вычисляем гистограмму ориентированных градиентов (HOG) как простой признак
            hog_features = self.calculate_hog_features(normalized_face)

            # Добавляем другие простые признаки
            hist_features = cv2.calcHist([gray_face], [0], None, [16], [0, 256]).flatten()
            hist_features = hist_features / np.sum(hist_features)  # Нормализуем

            # Объединяем все признаки
            combined_features = np.concatenate([hog_features, hist_features])

            # Добиваем до 512 размерности (как в FaceNet)
            if len(combined_features) < 512:
                padding = np.zeros(512 - len(combined_features))
                combined_features = np.concatenate([combined_features, padding])
            else:
                combined_features = combined_features[:512]

            return combined_features.astype(np.float32)

        except Exception as e:
            print(f"Ошибка при извлечении признаков: {e}")
            return np.random.randn(512).astype(np.float32)

    def calculate_hog_features(self, image):
        """Вычисление HOG признаков"""
        try:
            # Простая реализация HOG
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

            mag, ang = cv2.cartToPolar(gx, gy)

            # Квантование гистограммы
            bins = np.int32(8 * ang / (2 * np.pi))

            # Разбиваем на ячейки 8x8
            cell_size = 8
            h, w = image.shape
            hog_features = []

            for i in range(0, h - cell_size, cell_size):
                for j in range(0, w - cell_size, cell_size):
                    cell_mag = mag[i:i + cell_size, j:j + cell_size]
                    cell_bins = bins[i:i + cell_size, j:j + cell_size]

                    hist = np.zeros(8)
                    for bin_val in range(8):
                        hist[bin_val] = np.sum(cell_mag[cell_bins == bin_val])

                    # L2 нормализация
                    hist = hist / (np.linalg.norm(hist) + 1e-6)
                    hog_features.extend(hist)

            return np.array(hog_features).astype(np.float32)

        except Exception as e:
            print(f"Ошибка HOG: {e}")
            return np.zeros(128).astype(np.float32)

    def detect_face(self, frame):
        """Детекция лица с помощью OpenCV"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),  # Увеличиваем минимальный размер для лучшего качества
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Возвращаем координаты и обрезанное лицо
                face_img = frame[y:y + h, x:x + w]
                return face_img, [x, y, x + w, y + h]

        except Exception as e:
            print(f"Ошибка детекции лица: {e}")

        return None, None

    def register_face(self, name, num_samples=5):
        """Регистрация нового лица"""
        print(f"Регистрация пользователя: {name}")

        cap = self.setup_camera()
        if cap is None:
            return False

        print("✅ Камера подключена! Смотрите в камеру...")

        embeddings = []
        sample_count = 0

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Не удалось получить кадр")
                continue

            # Детекция лица
            face_img, box = self.detect_face(frame)

            display_frame = frame.copy()

            if face_img is not None and box is not None:
                # Извлекаем признаки
                embedding = self.extract_face_features(face_img)
                embeddings.append(embedding)
                sample_count += 1

                print(f"✅ Образец {sample_count}/{num_samples} получен")

                # Визуализация
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Sample: {sample_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Сохраняем пример лица для отладки
                if sample_count == 1:
                    cv2.imwrite(f"face_sample_{name}.jpg", face_img)
            else:
                cv2.putText(display_frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Face Registration', display_frame)

            # Пауза между образцами
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if embeddings:
            # Усредняем эмбеддинги
            avg_embedding = np.mean(embeddings, axis=0)
            self.embeddings_db[name] = avg_embedding
            self.save_database()
            print(f"✅ Пользователь {name} успешно зарегистрирован!")
            return True
        else:
            print("❌ Не удалось зарегистрировать пользователя")
            return False

    def recognize_face(self, embedding):
        """Распознавание лица по эмбеддингу"""
        if not self.embeddings_db:
            return "No database", 0.0

        best_similarity = 0.0
        best_name = "Unknown"

        for name, db_embedding in self.embeddings_db.items():
            # Используем косинусное сходство
            similarity = cosine_similarity([embedding], [db_embedding])[0][0]

            if similarity > best_similarity and similarity > self.threshold:
                best_similarity = similarity
                best_name = name

        return best_name, best_similarity

    def real_time_recognition(self):
        """Режим реального времени распознавания"""
        if not self.embeddings_db:
            print("Сначала зарегистрируйте хотя бы одно лицо!")
            return

        cap = self.setup_camera()
        if cap is None:
            return

        print("🔍 Запуск распознавания... Нажмите 'q' для выхода")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция и распознавание
            face_img, box = self.detect_face(frame)

            display_frame = frame.copy()

            if face_img is not None and box is not None:
                # Извлекаем признаки и распознаем
                embedding = self.extract_face_features(face_img)
                name, confidence = self.recognize_face(embedding)

                x1, y1, x2, y2 = box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # Рисуем рамку и текст
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(display_frame, f"{name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Статус
                status = "✅ VERIFIED" if name != "Unknown" else "❌ UNKNOWN"
                cv2.putText(display_frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                cv2.putText(display_frame, "🔍 Searching for face...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Face Recognition - Press Q to quit', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def test_camera_simple():
    """Простой тест камеры"""
    print("🔍 Тестируем камеру...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Камера 0 не доступна, пробуем камеру 1...")
        cap = cv2.VideoCapture(1)

    if cap.isOpened():
        print("✅ Камера найдена! Покажите лицо для теста...")

        for i in range(50):  # 50 кадров
            ret, frame = cap.read()
            if ret:
                # Простая детекция лица для теста
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Camera Test - Press any key to close', frame)

                if cv2.waitKey(1) & 0xFF != 255:
                    break
            else:
                print("⚠️ Не удалось получить кадр")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Тест камеры завершен")
    else:
        print("❌ Не удалось подключиться к камере")


def main():
    print("=" * 60)
    print("🔐 СИСТЕМА БИОМЕТРИЧЕСКОЙ ИДЕНТИФИКАЦИИ")
    print("=" * 60)

    # Сначала тестируем камеру
    test_camera_simple()

    recognizer = WorkingFaceRecognition(threshold=0.6)

    while True:
        print("\n" + "=" * 50)
        print("МЕНЮ:")
        print("1. 📷 Зарегистрировать новое лицо")
        print("2. 🔍 Запуск распознавания")
        print("3. 👥 Показать базу данных")
        print("4. 🧪 Тест камеры")
        print("5. 🚪 Выход")
        print("=" * 50)

        choice = input("Выберите действие (1-5): ").strip()

        if choice == '1':
            name = input("Введите ваше имя: ").strip()
            if name:
                recognizer.register_face(name)
            else:
                print("❌ Имя не может быть пустым!")

        elif choice == '2':
            recognizer.real_time_recognition()

        elif choice == '3':
            if recognizer.embeddings_db:
                print(f"\n📊 Зарегистрированные пользователи: {list(recognizer.embeddings_db.keys())}")
            else:
                print("\n📊 База данных пуста")

        elif choice == '4':
            test_camera_simple()

        elif choice == '5':
            print("👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор!")


if __name__ == "__main__":
    main()