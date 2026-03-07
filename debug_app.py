import sys
import os
import traceback
from PyQt6.QtWidgets import QApplication
from ui_main import MainWindow

def log_error(msg):
    with open("crash_log.txt", "a") as f:
        f.write(msg + "\n")

def exception_hook(exctype, value, tb):
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    print(error_msg)
    log_error(error_msg)
    sys.exit(1)

sys.excepthook = exception_hook

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
