import tkinter as tk
from gui import FaceRecognitionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_recognition)  # корректное закрытие камеры
    root.mainloop()
