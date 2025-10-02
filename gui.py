import tkinter as tk
from tkinter import messagebox
from face_recognition import FaceRecognitionSystem

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("400x300")
        self.system = FaceRecognitionSystem(threshold=0.65)  # порог можно менять

        # Заголовок
        self.label = tk.Label(root, text="Система распознавания лиц", font=("Arial", 14))
        self.label.pack(pady=20)

        # Поле для имени
        self.name_label = tk.Label(root, text="Введите имя:")
        self.name_label.pack()
        self.name_entry = tk.Entry(root)
        self.name_entry.pack(pady=5)

        # Кнопки
        self.register_btn = tk.Button(root, text="Регистрация", command=self.register_face, width=20, height=2, bg="lightblue")
        self.register_btn.pack(pady=10)

        self.recognize_btn = tk.Button(root, text="Распознавание", command=self.recognize_face, width=20, height=2, bg="lightgreen")
        self.recognize_btn.pack(pady=10)

        self.quit_btn = tk.Button(root, text="Выход", command=root.quit, width=20, height=2, bg="lightcoral")
        self.quit_btn.pack(pady=10)

    def register_face(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Ошибка", "Введите имя перед регистрацией!")
            return
        self.system.register_face(name)

    def recognize_face(self):
        self.system.recognize()
