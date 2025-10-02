import sys
from PyQt5 import QtWidgets
from ui_main import Ui_MainWindow
from face_recognition import FaceRecognitionSystem


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.recognizer = FaceRecognitionSystem()

        # Привязка кнопок
        self.ui.btnRegister.clicked.connect(self.register_user)
        self.ui.btnRecognize.clicked.connect(self.recognize)
        self.ui.btnShowDB.clicked.connect(self.show_db)
        self.ui.btnDelete.clicked.connect(self.delete_user)
        self.ui.btnExit.clicked.connect(self.close)

    def log(self, text):
        """Вывод текста в окно статуса"""
        self.ui.textStatus.append(text)

    def register_user(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Регистрация", "Введите имя:")
        if ok and name:
            self.recognizer.register_face(name)
            self.log(f"✅ Пользователь {name} зарегистрирован")

    def recognize(self):
        self.log("🚀 Запуск распознавания...")
        self.recognizer.recognize()
        self.log("🛑 Распознавание завершено")

    def show_db(self):
        if self.recognizer.db.has_faces():
            users = "\n".join(self.recognizer.db.all_faces().keys())
            self.log("👥 В базе:\n" + users)
        else:
            self.log("❌ База пуста")

    def delete_user(self):
        if not self.recognizer.db.has_faces():
            self.log("❌ База пуста")
            return

        names = list(self.recognizer.db.all_faces().keys())
        name, ok = QtWidgets.QInputDialog.getItem(self, "Удаление", "Выберите пользователя:", names, 0, False)
        if ok:
            self.recognizer.db.remove_face(name)
            self.log(f"🗑 Пользователь {name} удалён")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
