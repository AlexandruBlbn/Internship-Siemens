import json
import traceback

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def save_json(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            return True
    except:
        traceback.print_exc()
        return False
    
import sys
from PyQt5.QtWidgets import QApplication
from core.recognizer import RecognizerWorker
from core.network_receiver import NetworkReceiver
from core.robot_controller import RobotController
from core.lcd import send_lcd_message
from core.animated_ripple import AnimatedRipple  # dacă îl muţi în core
from wake_app.main_window import MainWindow      # Clasa GUI pe care o păstrezi

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()