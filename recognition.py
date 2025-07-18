#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Curabot complet Raspberry Pi, Camera Pi și LCD 100% sincronizat """

import sys, threading, os, json, time, subprocess
from datetime import datetime

import cv2
import easyocr
import speech_recognition as sr
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QMessageBox, QPushButton, QHBoxLayout, QSizePolicy
)

from flask import Flask, request
from db_medicamente import MedDatabase, strip_accents, extract_base_name 
import json
import serial
# root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if root not in sys.path:
#     sys.path.insert(0, root)
    
# from lcd_messages import *

# LCD_MESSAGE_FILE = '/home/user/Desktop/Internship-Siemens/lcd_message.json'


# Apoi înlocuiește definițiile existente cu:
LCD_PORT = '/dev/ttyACM0' 
LCD_BAUDRATE = 9600

# Definește direct mesajele LCD utilizate
LCD_SPEAK_MED = "Spune medicament"
LCD_WAITING = "Astept..."
LCD_DETECTING = "Cautare..."
LCD_EXECUTING = "Medicamentul..."


# Modifică clasa LCDSerial pentru a trimite direct mesajele formatate corect
class LCDSerial:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        try:
            time.sleep(1.0)
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Așteaptă stabilizarea conexiunii
            print(f"[LCD] Conexiune stabilită pe {port}")
            self.connected = True
            
            # Resetează buffer-ul înainte de primul mesaj
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            # Trimite un newline gol pentru resetare
            self.ser.write(b"\n")
            self.ser.flush()
            time.sleep(1.0)

            # Mesaj inițial
            self.send("Initializare...")
            time.sleep(5.0)  # Așteaptă procesarea primului mesaj
            
            # Al doilea mesaj pentru confirmare
            print("[LCD] Trimit al doilea mesaj: 'Astept...'")
            self.send("Astept...")

        except Exception as e:
            print(f"[LCD] Eroare conectare: {e}")
            self.connected = False
            self.ser = None

    def send(self, message):
        if not self.connected or self.ser is None:
            print(f"[LCD] Nu pot trimite: {message}")
            return False
        try:
            # Curăță buffer-ul
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Formatare simplă: doar textul urmat de newline
            # Arduino așteaptă exact acest format
            cmd = f"{message}\n".encode('utf-8')
            self.ser.write(cmd)
            self.ser.flush()
            
            # Delay mai mare pentru procesare
            time.sleep(0.3)  # Crescut de la 0.5 la 1.0
            
            print(f"[LCD] Trimis: {message}")
            return True
        except Exception as e:
            print(f"[LCD] Eroare: {e}")
            return False

# Inițializează LCD-ul
lcd = LCDSerial()

# Funcție simplificată pentru trimiterea mesajelor
def send_lcd_message(msg):
    """Trimite mesaj direct la Arduino"""
    if lcd and lcd.connected:
        # Delay pentru a evita comenzi rapide consecutive
        time.sleep(0.5)
        lcd.send(msg)
    print(f"[LCD] Mesaj trimis: {msg}")

sys.path.append('/home/user/Desktop/Internship-Siemens')

MANEVRE_SCRIPT = '/home/user/Desktop/Internship-Siemens/manevre.py'
PICKUP_SIGNAL_FILE = '/home/user/Desktop/Internship-Siemens/pickup_signal.json'
CAMERA_REQUEST_FILE = '/home/user/Desktop/Internship-Siemens/camera_request.json'
CAMERA_RELEASE_FILE = '/home/user/Desktop/Internship-Siemens/camera_released.json'



# ---------------------- FOTO CU PI CAMERA ----------------------

def convert_coordinates_back(bbox, rotation_name, rotated_shape):
    """Convertește coordonatele de la imaginea rotită înapoi la imaginea originală"""
    h_rot, w_rot = rotated_shape
    
    converted_bbox = []
    for point in bbox:
        x, y = point
        
        if rotation_name == 'original':
            # Nicio modificare
            new_x, new_y = x, y
        elif rotation_name == 'rot_90':
            # Rotit 90° în sens orar -> convertim înapoi
            new_x = y
            new_y = w_rot - x
        elif rotation_name == 'rot_neg90':
            # Rotit 90° în sens anti-orar -> convertim înapoi  
            new_x = h_rot - y
            new_y = x
        elif rotation_name == 'rot_180':
            # Rotit 180° -> convertim înapoi
            new_x = w_rot - x
            new_y = h_rot - y
        else:
            new_x, new_y = x, y
            
        converted_bbox.append([new_x, new_y])
    
    return converted_bbox

def capture_picamera(photo_path="utils/meds.png"):
    os.makedirs(os.path.dirname(photo_path), exist_ok=True)
    with open(CAMERA_REQUEST_FILE, 'w') as f:
        json.dump({"action": "release_camera", "timestamp": datetime.now().isoformat()}, f)
    print("[Curabot] Solicitare eliberare camera trimisă către manevre.py...")
    start = time.time()
    while time.time() - start < 7:
        if os.path.exists(CAMERA_RELEASE_FILE):
            with open(CAMERA_RELEASE_FILE, 'r') as f:
                status = json.load(f).get('status')
            if status == 'released':
                print("[Curabot] Camera a fost eliberată de manevre.py!")
                os.remove(CAMERA_RELEASE_FILE)
                break
        time.sleep(0.1)
    else:
        raise RuntimeError("Timeout: Nu am primit eliberarea camerei de la manevre.py!")
    
    # Capturăm inițial un JPEG temporar pentru compatibilitate
    temp_jpg = os.path.splitext(photo_path)[0] + ".jpg"
    cmd = [
        "libcamera-still",
        "-t", "800",
        "-o", temp_jpg,
        "--width", "1440",
        "--height", "1080",
        "--nopreview",
        "--autofocus-mode", "auto"
    ]
    print(f"[Curabot] Capturez imaginea cu Pi Camera: {temp_jpg}")
    subprocess.run(cmd, check=True)
    # Verificăm JPEG-ul și convertim la PNG
    if not os.path.exists(temp_jpg) or os.path.getsize(temp_jpg) < 10000:
        raise RuntimeError("Camera Pi nu a capturat imaginea corect!")
    img = cv2.imread(temp_jpg)
    cv2.imwrite(photo_path, img)
    os.remove(temp_jpg)
    print("[Curabot] Imagine capturată cu succes!")
    os.remove(CAMERA_REQUEST_FILE)
    print("[Curabot] Camera_request.json eliminat, manevre.py poate relua.")

# ---------------------- EasyOCR helper ----------------------
def easyocr_detect_medicine(med_name, image_path, output_path=None):
    import difflib
    reader = easyocr.Reader(['ro', 'en'], gpu=False, verbose=False)
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise FileNotFoundError(f"Nu pot încărca imaginea: {image_path}")
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = img_gray.shape[:2]
    img_for_draw = img_original.copy()

    # Threshold minim pentru dimensiuni bounding box-uri
    MIN_BBOX_WIDTH = 50   # Lățime minimă în pixeli
    MIN_BBOX_HEIGHT = 180  # Înălțime minimă în pixeli
    MIN_BBOX_AREA = 17000   # Aria minimă în pixeli pătrați

    # Detectare bounding box-uri cu OpenCV (contururi)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_regions = []
    total_contours = len(contours)
    filtered_count = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # Aplicăm threshold-urile de dimensiune ÎNAINTE de alte verificări
        if (w >= MIN_BBOX_WIDTH and h >= MIN_BBOX_HEIGHT and area >= MIN_BBOX_AREA and
            0.3 < aspect_ratio < 20 and area < 80000 and 
            w < img_gray.shape[1] * 0.9 and h < img_gray.shape[0] * 0.5 and 
            x > 5 and x + w < img_gray.shape[1] - 5 and y > 5 and y + h < img_gray.shape[0] - 5):
            center_x = x + w // 2
            center_y = y + h // 2
            text_regions.append({'bbox': (x, y, w, h), 'center': (center_x, center_y)})
            filtered_count += 1

    print(f"[OCR] Threshold aplicat: min {MIN_BBOX_WIDTH}x{MIN_BBOX_HEIGHT}px, aria ≥{MIN_BBOX_AREA}px²")
    print(f"[OCR] Regiuni detectate: {filtered_count}/{total_contours} (filtrate după dimensiune)")
    print(f"[OCR] Începe analiza cu EARLY STOPPING pentru medicamentul: '{med_name}'")

    matches = []
    selected_idx = None
    med_norm = strip_accents(med_name).replace(" ", "")
    keywords = [strip_accents(word) for word in med_name.split() if len(word) > 3]
    early_stop = False  # Flag pentru early stopping

    # OCR pe fiecare crop (doar pe regiuni care au trecut threshold-ul) cu early stopping
    for idx, region in enumerate(text_regions):
        if early_stop:
            print(f"[OCR-EARLY-STOP] Opresc analiza la regiunea {idx+1} - medicament găsit în regiunea {selected_idx+1}")
            break
            
        x, y, w, h = region['bbox']
        print(f"[OCR] Analizez regiunea {idx+1}/{len(text_regions)}: {w}x{h}px (aria: {w*h}px²)")
        
        crop = img_gray[y:y+h, x:x+w]
        
        # Salvează cropul pentru debugging
        crops_dir = os.path.join(os.path.dirname(__file__), "utils", "crops")
        os.makedirs(crops_dir, exist_ok=True)
        crop_filename = f"{med_name.replace(' ', '_')}_box{idx}_{w}x{h}.png"
        crop_path = os.path.join(crops_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        print(f"[OCR] Crop salvat: {crop_path}")
        
        # Rotește cropul cu 90 de grade pentru OCR
        crop_rot90 = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        ocr_results = reader.readtext(crop_rot90)
        
        # Analizează fiecare text detectat în această regiune
        for bbox, text, conf in ocr_results:
            text_norm = strip_accents(text).replace(" ", "")
            ratio = difflib.SequenceMatcher(None, med_norm, text_norm).ratio()
            found = False
            
            # Verifică dacă textul match-uiește medicamentul țintă
            if len(text_norm) >= 4 and ((med_norm in text_norm or text_norm in med_norm) or ratio > 0.75):
                found = True
            else:
                for kw in keywords:
                    if kw in text_norm or text_norm in kw:
                        found = True
                        break
                    if len(text_norm) >= 4 and difflib.SequenceMatcher(None, kw, text_norm).ratio() > 0.85:
                        found = True
                        break
            
            if found:
                matches.append({'bbox': region['bbox'], 'text': text, 'confidence': conf, 'center': region['center']})
                selected_idx = idx
                early_stop = True  # Activează early stopping
                print(f"[OCR-EARLY-STOP] Medicament '{med_name}' găsit în regiunea {idx+1}! Text detectat: '{text}' (conf: {conf:.2f})")
                print(f"[OCR-EARLY-STOP] Opresc analiza - nu mai verific restul de {len(text_regions) - idx - 1} regiuni")
                break  # Oprește analiza acestei regiuni

    # Desenare: bbox selectat roșu, restul verzi (fără punct OCR)
    for idx, region in enumerate(text_regions):
        x, y, w, h = region['bbox']
        color = (0, 255, 0)  # verde
        thickness = 2
        if selected_idx is not None and idx == selected_idx:
            color = (0, 0, 255)  # roșu
            thickness = 3
        cv2.rectangle(img_for_draw, (x, y), (x + w, y + h), color, thickness)
        # Eliminăm desenarea punctului OCR, doar bbox și eticheta SELECTED
        if selected_idx is not None and idx == selected_idx:
            cv2.putText(img_for_draw, "SELECTED", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Pregătește datele pentru robot: centrul bbox-ului selectat
    center_final = None
    camera_center = None
    if selected_idx is not None:
        # Calculează punctul median al bounding box-ului selectat
        x, y, w, h = text_regions[selected_idx]['bbox']
        center_final = (x + w // 2, y + h // 2)
        print(f"[OCR] Bbox selectat: x={x}, y={y}, w={w}, h={h}")
        print(f"[OCR] Punct median bbox: ({center_final[0]}, {center_final[1]})")
        
        # Scalează la rezoluția camerei (1440x1080) pentru manevre.py
        camera_scale_x = 1440 / w_orig
        camera_scale_y = 1080 / h_orig
        camera_x = int(center_final[0] * camera_scale_x)
        camera_y = int(center_final[1] * camera_scale_y)
        camera_center = (camera_x, camera_y)
        print(f"[OCR] Scale factors: x={camera_scale_x:.3f}, y={camera_scale_y:.3f}")
        print(f"[OCR] Punct median scalat pentru camera: ({camera_x}, {camera_y})")
        
        # Salvează coordonatele pentru robot (pentru mișcarea brațului)
        coords_file = os.path.join(os.path.dirname(__file__), "coords_for_robot.json")
        robot_data = {
            'medicament': med_name,
            'center': [camera_x, camera_y],
            'original_center': [center_final[0], center_final[1]],
            'corners': [],
            'processing_mode': 'bbox_ocr',
            'bbox_dimensions': [w, h],
            'timestamp': datetime.now().isoformat()
        }
        with open(coords_file, "w", encoding="utf-8") as f:
            json.dump(robot_data, f, indent=2)
        
        # Actualizează DIRECT punctul OCR în manevre.py (fără legile de mișcare)
        # Salvează punctul median al bbox-ului ca punct OCR fix
        ocr_update_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_point_update.json")
        ocr_data = {
            'action': 'update_ocr_point',
            'ocr_point': [camera_x, camera_y],  # Coordonate scalate pentru camera
            'original_point': [center_final[0], center_final[1]],  # Coordonate originale
            'medicament': med_name,
            'bbox': [x, y, w, h],
            'confidence': matches[0]['confidence'] if matches else 1.0,
            'bypass_movement_laws': True,  # Flag pentru a ignora legile de mișcare
            'source': 'bbox_detection',
            'timestamp': datetime.now().isoformat()
        }
        with open(ocr_update_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, indent=2)
        
        print(f"[OCR] Coordonate salvate în: {coords_file}")
        print(f"[OCR] Punct OCR actualizat DIRECT în: {ocr_update_file}")
        print(f"[OCR] Medicament: {med_name}, Punct BBOX: {center_final} -> Camera: {camera_center}")
        print(f"[OCR] Punctul OCR va fi setat fix la centrul bbox-ului (fără compensări de mișcare)")
        print(f"[OCR] EARLY STOPPING - Analiza completă după găsirea medicamentului în regiunea {selected_idx+1}")
        # Auto-trigger robot arm after OCR analysis
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            # If running in the main app, trigger robot pickup automatically
            if app is not None:
                for widget in app.topLevelWidgets():
                    if hasattr(widget, 'robot_controller'):
                        widget.robot_controller.send_pickup_signal()
                        print("[OCR] Semnal automat trimis pentru preluarea punctului OCR.")
                        break
        except Exception as e:
            print(f"[OCR] Eroare la trimiterea semnalului automat către robot: {e}")
    else:
        coords_file = os.path.join(os.path.dirname(__file__), "coords_for_robot.json")
        if os.path.exists(coords_file):
            os.remove(coords_file)
        print(f"[OCR] Nu s-a găsit niciun match pentru '{med_name}' în {len(text_regions)} regiuni analizate")
        print("[OCR] Fișierul coords_for_robot.json a fost șters")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "utils", "meds_marked.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_for_draw)
    print(f"[OCR] Imagine marcată salvată ca PNG: {output_path}")

    return {
        "medicine_name": med_name,
        "matches_found": len(matches),
        "matches": matches,
        "center_for_robot": center_final,
        "camera_center": camera_center,
        "marked_image_path": output_path,
        "timestamp": datetime.now().isoformat()
    }

# ---------------------- Gestionarea procesului manevre.py ----------------------
class RobotController:
    def __init__(self, parent=None):
        self.parent = parent
        self.process = None
        self.running = False

    def start_robot_process(self):
        if self.running:
            return
        self.process = subprocess.Popen(
            [sys.executable, MANEVRE_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.running = True
        threading.Thread(target=self._read_process_output, daemon=True).start()
        return True

    def _read_process_output(self):
        while self.running and self.process:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.encode('utf-8').strip()
            if line and self.parent:
                self.parent._append_log(f"[ROBOT] {line}")

    def send_pickup_signal(self):
        if not self.running:
            if self.parent:
                self.parent._append_log("Procesul manevre.py nu rulează")
            return False
        
        try:
            # Creează fișierul de semnal pentru pickup automat
            with open(PICKUP_SIGNAL_FILE, 'w') as f:
                json.dump({
                    "action": "pickup_ocr_point",
                    "auto_pickup": True,
                    "timestamp": datetime.now().isoformat()
                }, f)
            
            if self.parent:
                self.parent._append_log("Semnal automat trimis pentru preluarea punctului OCR")
            print("[ROBOT] Semnal automat trimis către manevre.py")
            return True
        except Exception as e:
            if self.parent:
                self.parent._append_log(f"Eroare la trimiterea semnalului: {e}")
            return False

    def stop_robot_process(self):
        if not self.running:
            return
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.running = False
            if self.parent:
                self.parent._append_log("Proces manevre.py oprit")

# ---------------------- Flask Network Receiver ----------------------
class NetworkReceiver(QObject):
    text_received = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.app = Flask("CurabotReceiver")
        self.app.add_url_rule('/speech', view_func=self.receive_text, methods=['POST'])

    def receive_text(self):
        data = request.json
        text = data.get("text")
        print(f"[REȚEA] Primit text: {text}")
        if text:
            self.text_received.emit(text)
            return {'status': 'ok'}
        return {'status': 'missing text'}

    def start(self):
        thread = threading.Thread(
            target=lambda: self.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
        )
        thread.daemon = True
        thread.start()

# ---------------------- AnimatedRipple ----------------------
class AnimatedRipple(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self._ripples = []
        self._listening = False
        self._spawn_timer = QTimer(self)
        self._spawn_timer.setInterval(400)
        self._spawn_timer.timeout.connect(self._add_ripple)
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(30)
        self._update_timer.timeout.connect(self._update_ripples)
        self._update_timer.start()
        # Turquoise/teal color palette 
        self._base_color = QColor(0, 153, 153)  # #009999
        self._min_radius = 30
        self._max_radius = 110
        # Optionally, add a subtle glow or gradient effect for a modern look

    def start(self):
        if not self._listening:
            self._listening = True
            self._spawn_timer.start()

    def stop(self):
        self._listening = False
        self._spawn_timer.stop()

    def _add_ripple(self):
        self._ripples.append({"radius": self._min_radius, "alpha": 170})

    def _update_ripples(self):
        new_list = []
        for r in self._ripples:
            r["radius"] += 3
            r["alpha"] = max(0, r["alpha"] - 4)
            if r["alpha"] > 0 and r["radius"] <= self._max_radius:
                new_list.append(r)
        self._ripples = new_list
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        center = QPointF(w / 2, h / 2)
        if not self._listening:
            brush = QBrush(self._base_color)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, self._min_radius, self._min_radius)
            painter.end()
            return
        # Ripple effect with color gradient for a modern look
        for r in self._ripples:
            color = QColor(0, 153, 153)  # #009999
            color.setAlpha(r["alpha"])
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, r["radius"], r["radius"])
        # Center solid circle with a lighter accent
        center_color = QColor(102, 204, 204)  # #66cccc
        painter.setBrush(QBrush(center_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, self._min_radius, self._min_radius)
        painter.end()


# ---------------------- MainWindow + Logic ----------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Curabot")
        self.setFixedSize(650, 950) 
        self.setStyleSheet(self._get_style())
        self.current_results = None
        self.awaiting_medicine_name = False

        # Inițializare controller robot
        self.robot_controller = RobotController(self)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        title = QLabel("Curabot")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("asistentul tău personal")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        self.circle = AnimatedRipple()
        self.circle.setFixedSize(220, 220)
        main_layout.addWidget(self.circle, alignment=Qt.AlignHCenter)

        self.instructions = QLabel("Inițializare…")
        self.instructions.setObjectName("InstrLabel")
        self.instructions.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.instructions)

        # Container pentru loguri cu border stilizat
        self.log_container = QWidget()
        self.log_container.setObjectName("LogContainer")
        self.log_container.setFixedHeight(120)
        
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)
        
        self.log_text = QTextEdit()
        self.log_text.setObjectName("LogText")
        self.log_text.setReadOnly(True)
        self.log_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(self.log_container)
        # Ascund log-urile în interfață (doar afișare în terminal)
        self.log_container.hide()

        # Container pentru imagine cu border stilizat
        self.img_container = QWidget()
        self.img_container.setObjectName("ImgContainer")
        self.img_container.setFixedHeight(320)
        
        img_layout = QVBoxLayout(self.img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        img_layout.setSpacing(0)

        self.img_label = QLabel("Imaginea va apărea aici…")
        self.img_label.setObjectName("ImgLabel")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout.addWidget(self.img_label)
        
        main_layout.addWidget(self.img_container, stretch=1)

        self.setLayout(main_layout)

        self.worker = RecognizerWorker(self)
        self.worker.sig_log.connect(self._append_log)
        self.worker.sig_phase.connect(self._on_phase_changed)
        self.worker.sig_error.connect(self._on_error)
        self.worker.sig_results.connect(self._on_results_ready)
        self.worker.sig_finished.connect(lambda: None)

        self.net_receiver = NetworkReceiver()
        self.net_receiver.text_received.connect(self.on_network_text)
        self.net_receiver.start()

        self.robot_controller.start_robot_process()

    def on_network_text(self, text):
        self._append_log(f"Am primit textul din rețea: {text}")
        if not self.awaiting_medicine_name:
            if strip_accents(text).strip() == "medicament":
                self.awaiting_medicine_name = True
                self.instructions.setText("Spune numele medicamentului…")
                self._append_log("Trigger recunoscut! Aștept numele medicamentului...")
                self.circle.start()
            else:
                self._append_log("Aștept să aud ca trigger!")
        else:
            self.awaiting_medicine_name = False
            self.instructions.setText("Spune „medicament”")
            self.circle.stop()
            self.worker.process_medicine(text)

    def _get_style(self) -> str:
        return '''
        QWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #009999, 
                stop:0.3 #006666, 
                stop:0.7 #004444, 
                stop:1 #002222);
            color: #ffffff;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 16px;
        }
        #TitleLabel {
            color: #ffffff;
            font-size: 48px;
            font-weight: 900;
            letter-spacing: 1.5px;
            margin: 14px 0 0 0;
            background: transparent;
            border: none;
            text-align: center;
            text-shadow: 0 4px 18px rgba(0, 153, 153, 0.3);
        }
        #SubtitleLabel {
            color: #66cccc;
            font-size: 22px;
            font-weight: 500;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
            border-radius: 15px;
        }
        #InstrLabel {
            color: #ffffff;
            font-size: 23px;
            font-weight: 600;
            margin: 8px 0 18px 0;
            background: rgba(0, 153, 153, 0.2);
            border-radius: 15px;
            padding: 16px 10px;
            border: 2px solid #009999;
        }
        #LogContainer {
            background: rgba(0, 153, 153, 0.1);
            border: 2px solid #009999;
            border-radius: 15px;
            padding: 0px;
        }
        #LogText {
            background: transparent;
            border: none;
            color: #cccccc;
            font-size: 14px;
            padding: 10px;
            border-radius: 13px;
        }
        #ImgContainer {
            background: rgba(0, 153, 153, 0.1);
            border: 2px solid #009999;
            border-radius: 20px;
            padding: 0px;
            
        }
        #ImgLabel {
            background: transparent;
            border: none;
            color: #cccccc;
            font-size: 18px;
            font-style: italic;
            border-radius: 18px;
        }
        '''

    def _append_log(self, text: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_text = f"[{timestamp}] {text}"
        print(formatted_text)

    def _on_phase_changed(self, phase: str):
        if getattr(self, "in_init_phase", False):
            return
        if phase == 'trigger':
            self.instructions.setText("Spune „medicament”")
            self.circle.stop()
            send_lcd_message(LCD_SPEAK_MED)
        elif phase == 'name':
            self.instructions.setText("Spune numele medicamentului…")
            self.circle.start()
            # queue_lcd_message(LCD_WAITING)
        elif phase == 'processing':
            self.instructions.setText("Procesez imaginea cu EasyOCR…")
            self.circle.stop()
            send_lcd_message(LCD_DETECTING)
            time.sleep(1) 
            
        elif phase == 'completed':
            self.instructions.setText("Procesare completă!")
            self.circle.stop()
            # send_lcd_message(LCD_WAITING)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Eroare", msg)
        self.worker.stop()
        self.close()

    def _on_results_ready(self, results):
        self.current_results = results
        if os.path.exists(results['marked_image_path']):
            pix = QPixmap(results['marked_image_path'])
            # Scalează imaginea pentru a umple complet containerul (ignora raportul de aspect)
            w = self.img_container.width() - 4  # -4 pentru border
            h = self.img_container.height() - 4  # -4 pentru border
            # Redimensionează
            scaled_pix = pix.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            # Crează pixmap cu colțuri rotunjite
            from PyQt5.QtGui import QPainterPath
            rounded = QPixmap(w, h)
            rounded.fill(Qt.transparent)
            painter = QPainter(rounded)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            radius = 18.0
            path.addRoundedRect(0.0, 0.0, float(w), float(h), radius, radius)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, scaled_pix)
            painter.end()
            self.img_label.setPixmap(rounded)
        self._append_log("Rezultate complete disponibile pentru salvare")

    def closeEvent(self, event):
        self.worker.stop()
        self.robot_controller.stop_robot_process()
        event.accept()

# ---------------------- RecognizerWorker ----------------------
class RecognizerWorker(QObject):
    sig_log = pyqtSignal(str)
    sig_phase = pyqtSignal(str)
    sig_error = pyqtSignal(str)
    sig_finished = pyqtSignal()
    sig_results = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self.recognizer = sr.Recognizer()
        self.trigger_word = "medicament"
        self.last_results = None
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        base = os.path.dirname(__file__)
        excel_path = os.path.join(base, 'medicamente.xlsx')
        self.db = MedDatabase(excel_path=excel_path)
        
        # Șterge coordonatele vechi la pornire
        self._clear_old_coordinates()
        
        self.source = sr.Microphone()
        with self.source as s:
            self.recognizer.adjust_for_ambient_noise(s, duration=2)
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _clear_old_coordinates(self):
        """Șterge coordonatele vechi la pornirea aplicației"""
        coords_file = os.path.join(os.path.dirname(__file__), "coords_for_robot.json")
        
        try:
            if os.path.exists(coords_file):
                os.remove(coords_file)
                print("[CLEANUP] Coordonate vechi șterse: coords_for_robot.json")
        except Exception as e:
            print(f"[CLEANUP] Eroare la ștergerea coordonatelor vechi: {e}")

    def _load_medicines_cache(self):
        """Funcție eliminată - nu mai folosim cache"""
        return False

    def stop(self):
        self._running = False
        self.sig_finished.emit()

    def get_last_results(self):
        return self.last_results

    def _run(self):
        if self.source is None:
            self.sig_log.emit("Nu am microfon disponibil, ascult doar comenzi din rețea!")
            return
        self.sig_phase.emit('trigger')
        with self.source as s:
            while self._running:
                try:
                    self.sig_phase.emit('trigger')
                    audio = self.recognizer.listen(s, timeout=5, phrase_time_limit=5)
                    try:
                        text = self.recognizer.recognize_google(audio, language='ro-RO').lower()
                        self.sig_log.emit(f"Auzit: '{text}'")
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        self.sig_log.emit(f"Eroare serviciu recunoaștere: {e}")
                        continue
                    if strip_accents(self.trigger_word) not in strip_accents(text):
                        continue
                    self.sig_phase.emit('name')
                    self.sig_log.emit("Trigger detectat! Spune numele medicamentului…")
                    audio2 = self.recognizer.listen(s, timeout=10, phrase_time_limit=8)
                    try:
                        name = self.recognizer.recognize_google(audio2, language='ro-RO').lower()
                        self.sig_log.emit(f"Medicament: '{name}'")
                        self.process_medicine(name)
                    except sr.UnknownValueError:
                        self.sig_log.emit("Nu am înțeles numele medicamentului.")
                        self.sig_phase.emit('trigger')
                        continue
                    except sr.RequestError as e:
                        self.sig_log.emit(f"Eroare serviciu: {e}")
                        self.sig_phase.emit('trigger')
                        continue
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.sig_log.emit(f"Eroare generală: {e}")
                    time.sleep(1)

    def process_medicine(self, medicine_name):
        if self.db is None:
            self.sig_log.emit("Baza de date medicamente nu a fost încărcată corect.")
            self.sig_phase.emit('completed')
            return
        self.sig_phase.emit('processing')
        # queue_lcd_message(LCD_DETECTING)
        self.sig_log.emit(f"Medicament cerut: '{medicine_name}'")
        
        norm = strip_accents(medicine_name)
        candidates = self.db.search_by_prefix(norm, limit=5)
        if not candidates:
            for med in self.db.medicamente:
                if extract_base_name(med['denumire']) == norm:
                    candidates = [med]
                    break
        if not candidates:
            self.sig_log.emit(f"'{medicine_name}' nu a fost găsit în baza de date.")
            self.sig_phase.emit('completed')
            return
        
        med = candidates[0]
        base_name = self.db.get_base_name(med)
        
        # Fă întotdeauna OCR pentru medicamentul cerut
        self.sig_log.emit(f"Fac analiza OCR pentru: {base_name}")
        photo_path = os.path.join(os.path.dirname(__file__), 'utils', 'meds.png')
        
        try:
            capture_picamera(photo_path=photo_path)
            send_lcd_message(LCD_EXECUTING)
            self.sig_log.emit("Poză capturată cu Pi Camera.")
            self.sig_log.emit("Procesez imaginea cu EasyOCR...")
            
            # Analizează medicamentul specific
            results = easyocr_detect_medicine(base_name, photo_path)
            
            if results['matches_found'] > 0:
                self.sig_log.emit(f"Găsit '{base_name}' în imagine")
                self.sig_log.emit(f"Coordonate robot: {results['camera_center']}")
                self.sig_log.emit("Trimit semnal automat către robot...")
                time.sleep(2.0)
                send_lcd_message(LCD_EXECUTING)
                
                # Trimite semnal automat către robot
                if self.parent() and hasattr(self.parent(), 'robot_controller'):
                    self.parent().robot_controller.send_pickup_signal()
            else:
                self.sig_log.emit(f"Nu am găsit '{base_name}' în imagine")
        
            
            self.last_results = results
            self.sig_results.emit(results)
            self.sig_phase.emit('completed')
            # queue_lcd_message(LCD_WAITING)
            
        except Exception as e:
            self.sig_log.emit(f"Eroare la procesarea OCR: {e}")
            self.sig_phase.emit('completed')

# ---------------------- Main loop ----------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Curabot")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
