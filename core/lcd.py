import time
import serial
from .config import LCD_PORT, LCD_BAUDRATE

class LCDSerial:
    def __init__(self):
        try:
            time.sleep(1.0)
            self.ser = serial.Serial(LCD_PORT, LCD_BAUDRATE, timeout=1)
            time.sleep(2.0)
            self.ser.reset_input_buffer(); self.ser.reset_output_buffer()
            self.ser.write(b"\n"); self.ser.flush()
            time.sleep(1.0)
            self.connected = True
        except:
            self.connected = False
            self.ser = None

    def send(self, message):
        if not self.connected:
            return False
        try:
            self.ser.reset_input_buffer(); self.ser.reset_output_buffer()
            self.ser.write(f"{message}\n".encode('utf-8'))
            self.ser.flush()
            time.sleep(0.3)
            return True
        except:
            return False

lcd = LCDSerial()

def send_lcd_message(msg):
    if lcd.connected:
        time.sleep(0.5)
        lcd.send(msg)