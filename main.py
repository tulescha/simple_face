import sys
from ui_main import *

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = MainApp()
    window.setWindowTitle("Система распознавания лиц")
    window.show()

    sys.exit(app.exec_())