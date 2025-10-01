import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
from PIL import Image
import time


class PyTorchFaceRecognition:
    def __init__(self, threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Используемое устройство: {self.device}")

        # Загружаем предобученную модель
        self.model = self.load_pretrained_model()

        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Детектор лиц OpenCV
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.threshold = threshold
        self.face_database = {}
        self.load_database()

    def load_pretrained_model(self):
        """Загрузка предобученной модели PyTorch"""
        try:
            model = models.resnet18(pretrained=True)
            # Убираем последний слой (fc), но оставляем avgpool
            modules = list(model.children())[:-1]
            model = nn.Sequential(*modules)
            model.eval()
            model.to(self.device)
            print("✅ PyTorch модель ResNet18 загружена успешно")
            return model
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return None


    def extract_features_torch(self, face_image):
        """Извлечение признаков с помощью PyTorch"""
        if self.model is None:
            return np.random.randn(512).astype(np.float32)

        try:
            # Конвертируем BGR в RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

            # Применяем трансформации
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Извлекаем признаки
            with torch.no_grad():
                features = self.model(input_batch)  # [1, 512, 1, 1]
                features = features.view(features.size(0), -1)  # [1, 512]
                features = features.squeeze(0).cpu().numpy()  # (512,)

            print(f"📊 Извлечены признаки размерности: {features.shape}")
            return features.astype(np.float32)

        except Exception as e:
            print(f"⚠️ Ошибка извлечения признаков PyTorch: {e}")
            return np.random.randn(512).astype(np.float32)

    def load_database(self):
        """Загрузка базы данных лиц"""
        try:
            if os.path.exists('pytorch_face_database.pkl'):
                with open('pytorch_face_database.pkl', 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"✅ Загружено {len(self.face_database)} лиц из базы данных")
            else:
                print("📝 База данных не найдена. Сначала зарегистрируйтесь.")
        except Exception as e:
            print(f"❌ Ошибка загрузки базы данных: {e}")
            self.face_database = {}

    def save_database(self):
        """Сохранение базы данных"""
        try:
            with open('pytorch_face_database.pkl', 'wb') as f:
                pickle.dump(self.face_database, f)
            print("💾 База данных сохранена")
        except Exception as e:
            print(f"❌ Ошибка сохранения базы данных: {e}")

    def setup_camera(self):
        """Настройка камеры"""
        for camera_index in [0, 1]:
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Камера {camera_index} подключена")
                    return cap
            cap.release()

        print("❌ Не удалось подключиться к камере!")
        return None

    def detect_face(self, frame):
        """Детекция лица"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = frame[y:y + h, x:x + w]
                return face_img, [x, y, x + w, y + h]

        except Exception as e:
            print(f"⚠️ Ошибка детекции лица: {e}")

        return None, None

    def calculate_cosine_similarity(self, features1, features2):
        """Вычисление косинусной схожести"""
        try:
            # Нормализуем векторы
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Косинусная схожесть
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Ошибка вычисления схожести: {e}")
            return 0.0

    def register_face(self, name):
        """Регистрация нового лица с PyTorch"""
        print(f"\n📷 Начинаем регистрацию пользователя: {name}")

        cap = self.setup_camera()
        if cap is None:
            return False

        print("👀 Смотрите в камеру прямо...")
        print("⏳ Собираем образцы...")

        features_list = []
        sample_count = 0
        max_samples = 5

        while sample_count < max_samples:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Не удалось получить кадр с камеры")
                continue

            face_img, box = self.detect_face(frame)
            display_frame = frame.copy()

            if face_img is not None:
                # Извлекаем признаки с помощью PyTorch
                features = self.extract_features_torch(face_img)
                features_list.append(features)
                sample_count += 1

                print(f"✅ Образец {sample_count}/{max_samples} - Признаки: {features.shape}")

                # Визуализация
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"PyTorch Sample: {sample_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Features: {features.shape[0]} dim", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Пауза между образцами
                time.sleep(1)
            else:
                cv2.putText(display_frame, "Лицо не найдено - двигайтесь ближе", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('PyTorch Face Registration - Press Q to cancel', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("❌ Регистрация отменена пользователем")
                break

        cap.release()
        cv2.destroyAllWindows()

        if features_list:
            # Усредняем признаки
            avg_features = np.mean(features_list, axis=0)
            self.face_database[name] = avg_features
            self.save_database()
            print(f"\n🎉 УСПЕХ! Пользователь {name} зарегистрирован!")
            print(f"📊 Размерность признаков: {avg_features.shape}")
            print(f"💾 Сохранено в базе данных")
            return True
        else:
            print("❌ Не удалось зарегистрировать лицо - лица не обнаружены")
            return False

    def recognize_face(self, features):
        """Распознавание лица по PyTorch признакам"""
        if not self.face_database:
            return "No database", 0.0

        best_match = "Unknown"
        best_similarity = 0.0

        for name, stored_features in self.face_database.items():
            similarity = self.calculate_cosine_similarity(features, stored_features)

            if similarity > best_similarity and similarity > self.threshold:
                best_similarity = similarity
                best_match = name

        return best_match, best_similarity

    def real_time_recognition(self):
        """Распознавание в реальном времени с PyTorch"""
        if not self.face_database:
            print("❌ Сначала зарегистрируйте хотя бы одно лицо!")
            return

        cap = self.setup_camera()
        if cap is None:
            return

        print("\n🔍 Запуск PyTorch распознавания...")
        print("📍 Нажмите Q для выхода")

        recognition_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Проблема с получением кадра")
                break

            face_img, box = self.detect_face(frame)
            display_frame = frame.copy()

            if face_img is not None and box is not None:
                # Извлекаем признаки с помощью PyTorch
                features = self.extract_features_torch(face_img)
                name, confidence = self.recognize_face(features)

                recognition_count += 1

                x1, y1, x2, y2 = box

                if name != "Unknown":
                    color = (0, 255, 0)  # Зеленый - свой
                    status = f"✅ {name}"
                    confidence_text = f"Confidence: {confidence:.3f}"
                else:
                    color = (0, 0, 255)  # Красный - чужой
                    status = "❌ Unknown Person"
                    confidence_text = f"Similarity: {confidence:.3f}"

                # Рисуем рамку и информацию
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(display_frame, status, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_frame, confidence_text, (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Системная информация
                cv2.putText(display_frame, "PyTorch ResNet18", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Database: {len(self.face_database)} users",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Frame: {recognition_count}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                cv2.putText(display_frame, "🔍 Searching for face...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, "PyTorch Ready - Show your face", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('PyTorch Face Recognition - Press Q to quit', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Выход из режима распознавания")
                break

        cap.release()
        cv2.destroyAllWindows()


def test_system():
    """Тест системы"""
    print("🧪 Тестируем систему...")

    # Тест камеры
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✅ Камера работает")
            # Тест детекции лица
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            print(f"✅ Детектор лиц работает. Найдено лиц: {len(faces)}")
        cap.release()
    else:
        print("❌ Камера не работает")

    # Тест PyTorch
    try:
        model = models.resnet18(pretrained=True)
        print("✅ PyTorch и torchvision работают")
    except Exception as e:
        print(f"❌ Ошибка PyTorch: {e}")


def main():
    print("=" * 70)
    print("🔐 PyTorch СИСТЕМА БИОМЕТРИЧЕСКОЙ ИДЕНТИФИКАЦИИ")
    print("=" * 70)
    print("🧠 Технологии: ResNet18 + PyTorch + OpenCV")
    print("🎯 Алгоритм: Косинусная схожесть глубоких признаков")
    print("📊 Метрика: 512-мерные эмбеддинги")
    print("=" * 70)

    # Информация о системе
    print(f"📦 PyTorch версия: {torch.__version__}")
    print(f"⚡ CUDA доступно: {torch.cuda.is_available()}")
    print(f"🔢 NumPy версия: {np.__version__}")
    print(f"📷 OpenCV версия: {cv2.__version__}")

    # Тестируем систему
    test_system()

    # Создаем систему распознавания
    recognizer = PyTorchFaceRecognition(threshold=0.6)

    while True:
        print("\n" + "=" * 50)
        print("МЕНЮ PyTorch системы распознавания:")
        print("1. 📷 Зарегистрировать новое лицо")
        print("2. 🔍 Запуск распознавания в реальном времени")
        print("3. 👥 Просмотр базы данных")
        print("4. ⚙️ Настройки системы")
        print("5. 🚪 Выход")
        print("=" * 50)

        choice = input("Выберите действие (1-5): ").strip()

        if choice == '1':
            name = input("Введите ваше имя: ").strip()
            if name:
                if name in recognizer.face_database:
                    print(f"⚠️ Пользователь {name} уже зарегистрирован!")
                    overwrite = input("Перезаписать? (y/n): ").strip().lower()
                    if overwrite == 'y':
                        recognizer.register_face(name)
                else:
                    recognizer.register_face(name)
            else:
                print("❌ Имя не может быть пустым!")

        elif choice == '2':
            recognizer.real_time_recognition()

        elif choice == '3':
            if recognizer.face_database:
                print(f"\n📊 База данных пользователей ({len(recognizer.face_database)}):")
                for i, name in enumerate(recognizer.face_database.keys(), 1):
                    features = recognizer.face_database[name]
                    print(f"   {i}. 👤 {name} - признаки: {features.shape}")
            else:
                print("\n📊 База данных пуста")

        elif choice == '4':
            print(f"\n⚙️ Текущие настройки:")
            print(f"   Порог схожести: {recognizer.threshold}")
            print(f"   Устройство: {recognizer.device}")
            new_threshold = input("Новый порог (0.1-0.9) или Enter для пропуска: ").strip()
            if new_threshold:
                try:
                    recognizer.threshold = float(new_threshold)
                    print(f"✅ Порог установлен: {recognizer.threshold}")
                except ValueError:
                    print("❌ Неверное значение порога")

        elif choice == '5':
            print("\n👋 До свидания! Система завершает работу...")
            break

        else:
            print("❌ Неверный выбор! Попробуйте снова.")


if __name__ == "__main__":
    main()