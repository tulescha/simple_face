import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import threading
from face_recognition import FaceRecognitionSystem


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        self.system = FaceRecognitionSystem(threshold=0.65)

        # Заголовок
        self.label = tk.Label(root, text="Система распознавания лиц", font=("Arial", 16))
        self.label.pack(pady=10)

        # Поле для имени
        self.name_label = tk.Label(root, text="Введите имя:")
        self.name_label.pack()
        self.name_entry = tk.Entry(root)
        self.name_entry.pack(pady=5)

        # Кнопки
        self.register_btn = tk.Button(root, text="Регистрация",
                                      command=self.register_face,
                                      width=20, height=2, bg="lightblue")
        self.register_btn.pack(pady=10)

        self.recognize_btn = tk.Button(root, text="Запуск распознавания",
                                       command=self.start_recognition,
                                       width=20, height=2, bg="lightgreen")
        self.recognize_btn.pack(pady=10)

        self.stop_btn = tk.Button(root, text="Остановить распознавание",
                                  command=self.stop_recognition,
                                  width=20, height=2, bg="orange")
        self.stop_btn.pack(pady=10)

        self.show_db_btn = tk.Button(root, text="Просмотр базы",
                                     command=self.show_database,
                                     width=20, height=2, bg="khaki")
        self.show_db_btn.pack(pady=10)

        self.set_thr_btn = tk.Button(root, text="Изменить пороги",
                                     command=self.set_thresholds,
                                     width=20, height=2, bg="lightyellow")
        self.set_thr_btn.pack(pady=10)

        self.quit_btn = tk.Button(root, text="Выход",
                                  command=root.quit,
                                  width=20, height=2, bg="lightcoral")
        self.quit_btn.pack(pady=10)

        # Видео окно
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Лог
        self.log = tk.Text(root, height=6, state="disabled")
        self.log.pack(fill="x", padx=10, pady=5)

        self.cap = None
        self.running = False

    def log_message(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def register_face(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Ошибка", "Введите имя перед регистрацией!")
            return
        self.log_message(f"✅ (демо) Пользователь {name} зарегистрирован.")
        # тут можно реализовать захват кадров с камеры и сохранение в базу

    def start_recognition(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.log_message("🔍 Запуск распознавания...")
        threading.Thread(target=self.update_frame, daemon=True).start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame, name = self.system.process_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            if name:
                self.log_message(f"Обнаружен: {name}")
            self.root.update_idletasks()
        self.cap.release()

    def stop_recognition(self):
        if self.running:
            self.running = False
            self.log_message("🛑 Распознавание остановлено")

        # очистка окна видео
        self.video_label.config(image="", text="Камера остановлена", font=("Arial", 12), fg="gray")
        self.video_label.imgtk = None

    def show_database(self):
        if self.system.db.has_faces():
            users = "\n".join(self.system.db.all_faces().keys())
            self.log_message("👥 В базе:\n" + users)
        else:
            self.log_message("❌ База пуста")

    def set_thresholds(self):
        try:
            low = float(simpledialog.askstring("Порог", "Введите нижний порог (например 0.75):"))
            high = float(simpledialog.askstring("Порог", "Введите верхний порог (например 0.85):"))
            self.system.low_thr = low
            self.system.high_thr = high
            self.log_message(f"✅ Пороги обновлены: сомнительное ≥ {low}, уверенное ≥ {high}")
        except Exception:
            messagebox.showerror("Ошибка", "Введите корректные числа!")
