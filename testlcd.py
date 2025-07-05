import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Așteaptă conexiunea

ser.write(b'Cautare...\n')
time.sleep(6)
ser.write(b'Astept...\n')
time.sleep(6)
ser.write(b'Medicamentul...\n')
time.sleep(6)
ser.write(b'Gata!\n')
