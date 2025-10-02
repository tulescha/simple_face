from face_recognition import FaceRecognitionSystem
import tkinter as tk
from gui import FaceRecognitionApp

def main():
    print("=" * 60)
    print("🔐 СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ (PyTorch + OpenCV)")
    print("=" * 60)

    recognizer = FaceRecognitionSystem()

    while True:
        print("\n📋 МЕНЮ:")
        print("1. Зарегистрировать пользователя")
        print("2. Запуск распознавания")
        print("3. Просмотр базы")
        print("4. Изменить пороги распознавания")
        print("5. Выход")

        choice = input("👉 Выберите действие: ").strip()

        if choice == '1':
            name = input("Введите имя: ").strip()
            recognizer.register_face(name)

        elif choice == '2':
            print("\n🚀 Запуск распознавания...")
            recognizer.recognize()

        elif choice == '3':
            if recognizer.db.has_faces():
                print("\n👥 Пользователи в базе:")
                for i, name in enumerate(recognizer.db.all_faces().keys(), 1):
                    print(f"{i}. {name}")
            else:
                print("❌ База пуста!")

        elif choice == '4':
            try:
                low = float(input("Введите нижний порог (например 0.75): "))
                high = float(input("Введите верхний порог (например 0.85): "))
                print(f"✅ Установлены пороги: сомнительное ≥ {low}, уверенное ≥ {high}")
                # сохраняем пороги прямо в recognizer
                recognizer.low_thr = low
                recognizer.high_thr = high
            except:
                print("⚠️ Ошибка: введите число через точку")

        elif choice == '5':
            print("👋 Выход из программы.")
            break

        else:
            print("⚠️ Неверный выбор!")

if __name__ == "__main__":
    # main()
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
