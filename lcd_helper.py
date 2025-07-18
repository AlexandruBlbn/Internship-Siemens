# lcd_helper.py
import serial
import time

class LCDSerial:
    def __init__(self, port="/dev/ttyACM0", baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Așteaptă deschiderea portului
        except Exception as e:
            print(f"[LCD] Eroare la deschiderea portului serial: {e}")
            self.ser = None

    def send(self, msg):
        if not self.ser:
            print("[LCD] Port serial indisponibil!")
            return
        try:
            # Trimitere mesaj și newline
            self.ser.write((msg.strip() + '\n').encode('utf-8'))
        except Exception as e:
            print(f"[LCD] Eroare la trimiterea mesajului: {e}")
