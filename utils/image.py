import cv2

def draw_bboxes(frame, regions, last_point):
    for r in regions:
        x, y, w, h = r['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{r['confidence']:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if last_point:
        cv2.circle(frame, (int(last_point[0]), int(last_point[1])), 10, (255, 0, 0), 2)

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