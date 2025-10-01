from face_recognition import FaceRecognitionSystem

def main():
    system = FaceRecognitionSystem()

    while True:
        print("\nМеню:")
        print("1. Зарегистрировать лицо")
        print("2. Запуск распознавания")
        print("3. Показать базу")
        print("4. Удалить пользователя")
        print("5. Выход")

        choice = input("Ваш выбор: ").strip()
        if choice == "1":
            name = input("Введите имя: ")
            system.register_face(name)
        elif choice == "2":
            system.recognize()
        elif choice == "3":
            for i, name in enumerate(system.db.all_faces().keys(), 1):
                print(f"{i}. {name}")
        elif choice == "4":
            name = input("Имя для удаления: ")
            system.db.delete_face(name)
        elif choice == "5":
            break
        else:
            print("❌ Неверный ввод!")

if __name__ == "__main__":
    main()
