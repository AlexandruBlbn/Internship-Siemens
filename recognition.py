#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Curabot complet Raspberry Pi, Camera Pi și LCD 100% sincronizat """

import sys, threading, os, json, time
from datetime import datetime

import cv2
import easyocr
import speech_recognition as sr
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QBrush, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QMessageBox, QPushButton, QSizePolicy
)

from flask import Flask, request
from db_medicamente import MedDatabase, strip_accents, extract_base_name
from lcd_helper import LCDSerial

lcd = LCDSerial(port="/dev/ttyACM0")  # schimbă dacă e nevoie

# ---------------------- FOTO CU PI CAMERA ----------------------
def capture_picamera(photo_path="utils/meds.jpg"):
    import subprocess
    os.makedirs(os.path.dirname(photo_path), exist_ok=True)
    cmd = [
        "libcamera-still",
        "-t", "800",
        "-o", photo_path,
        "--width", "1280",
        "--height", "960",
        "--nopreview",
        "--autofocus-mode", "auto"
    ]
    subprocess.run(cmd, check=True)
    if not os.path.exists(photo_path) or os.path.getsize(photo_path) < 10000:
        raise RuntimeError("Camera Pi nu a capturat imaginea corect!")

# ---------------------- EasyOCR helper ----------------------
def easyocr_detect_medicine(med_name, image_path, output_path=None):
    reader = easyocr.Reader(['ro', 'en'], gpu=False, verbose=False)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot încărca imaginea: {image_path}")
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1200:
        scale = 1200 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    results = reader.readtext(img)
    matches = []
    med_norm = strip_accents(med_name).replace(" ", "")
    keywords = [strip_accents(word) for word in med_name.split() if len(word) > 3]
    import difflib
    for (bbox, text, conf) in results:
        text_norm = strip_accents(text).replace(" ", "")
        ratio = difflib.SequenceMatcher(None, med_norm, text_norm).ratio()
        found = False
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
            (top_left, top_right, bot_right, bot_left) = bbox
            top_left = tuple(map(int, top_left))
            bot_right = tuple(map(int, bot_right))
            xs = [int(x) for x, y in bbox]
            ys = [int(y) for x, y in bbox]
            center = (int(sum(xs) / 4), int(sum(ys) / 4))
            matches.append({
                'bbox': bbox,
                'text': text,
                'confidence': conf,
                'ratio': ratio,
                'center': center
            })
            cv2.rectangle(img, top_left, bot_right, (0,255,0), 2)
            cv2.putText(img, f"{text} ({int(conf)}%)", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    coords_robot = [m['center'] for m in matches]
    center_final = None
    if coords_robot:
        avg_x = int(sum([c[0] for c in coords_robot]) / len(coords_robot))
        avg_y = int(sum([c[1] for c in coords_robot]) / len(coords_robot))
        center_final = (avg_x, avg_y)
        cv2.circle(img, center_final, 18, (0,255,255), 4)
        cv2.circle(img, center_final, 5, (0,0,255), -1)
        cv2.putText(img, "Pick here", (center_final[0]+12, center_final[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        robot_data = {
            'medicament': med_name,
            'center': [avg_x, avg_y],
            'corners': []
        }
        with open("coords_for_robot.json", "w", encoding="utf-8") as f:
            json.dump(robot_data, f, indent=2)
    else:
        if os.path.exists("coords_for_robot.json"):
            os.remove("coords_for_robot.json")

    if output_path is None:
        output_path = image_path.replace(".jpg", "_marked.jpg").replace(".png", "_marked.jpg")
    cv2.imwrite(output_path, img)
    return {
        "medicine_name": med_name,
        "matches_found": len(matches),
        "matches": matches,
        "center_for_robot": center_final,
        "marked_image_path": output_path,
        "timestamp": datetime.now().isoformat()
    }

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
        self._base_color = QColor(30, 136, 229)
        self._min_radius = 30
        self._max_radius = 110

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
        for r in self._ripples:
            color = QColor(self._base_color)
            color.setAlpha(r["alpha"])
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, r["radius"], r["radius"])
        brush = QBrush(self._base_color)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, self._min_radius, self._min_radius)
        painter.end()

# ---------------------- MainWindow + Logic ----------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🩺 Curabot")
        self.setMinimumSize(750, 950)
        self.setStyleSheet(self._get_style())
        self.current_results = None
        self.awaiting_medicine_name = False

        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(22)

        title = QLabel("Curabot")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("asistentul tău personal")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        self.circle = AnimatedRipple()
        self.circle.setFixedSize(220, 220)
        layout.addWidget(self.circle, alignment=Qt.AlignHCenter)

        self.instructions = QLabel("Initializare…")
        self.instructions.setObjectName("InstrLabel")
        self.instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instructions)

        self.btn_save = QPushButton("💾 Salvează rezultate")
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setObjectName("LogArea")
        self.txt_log.setMinimumHeight(120)
        layout.addWidget(self.txt_log)

        self.img_label = QLabel("Imaginea va apărea aici…")
        self.img_label.setObjectName("ImgPlaceholder")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(520, 340)
        self.img_label.setMaximumHeight(500)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.img_label, stretch=3)

        self.setLayout(layout)

        self.worker = RecognizerWorker(self)
        self.worker.sig_log.connect(self._append_log)
        self.worker.sig_phase.connect(self._on_phase_changed)
        self.worker.sig_error.connect(self._on_error)
        self.worker.sig_results.connect(self._on_results_ready)
        self.worker.sig_finished.connect(lambda: self._append_log("🛑 Worker oprit."))
        self._append_log("🚀 Aplicația a fost inițializată!")
        self.btn_save.clicked.connect(self._save_results)

        # Trigger logic pentru rețea!
        self.net_receiver = NetworkReceiver()
        self.net_receiver.text_received.connect(self.on_network_text)
        self.net_receiver.start()
        self._append_log("🌐 Serverul de rețea pentru comenzi vocale a pornit.")

        # Inițializare LCD la pornire: 30s "Initializare...", apoi "Astept..."
        lcd.send("Initializare...")
        self.in_init_phase = True
        QTimer.singleShot(30000, self._finish_init_phase)

    def _finish_init_phase(self):
        self.in_init_phase = False
        lcd.send("Astept...")
        self._on_phase_changed('trigger')

    def on_network_text(self, text):
        self._append_log(f"🔊 Am primit textul din rețea: {text}")
        if not self.awaiting_medicine_name:
            if strip_accents(text).strip() == "medicament":
                self.awaiting_medicine_name = True
                self.instructions.setText("Spune numele medicamentului…")
                self._append_log("✅ Trigger recunoscut! Aștept numele medicamentului...")
                self.circle.start()
                lcd.send("Astept...")
            else:
                self._append_log("❌ Aștept să aud „medicament” ca trigger!")
        else:
            self.awaiting_medicine_name = False
            self.instructions.setText("Spune „medicament”")
            self.circle.stop()
            self.worker.process_medicine(text)

    def _get_style(self) -> str:
        return '''
        QWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0d164e, stop:1 #203463);
            color: #f6f8fa;
            font-family: 'Segoe UI', 'Montserrat', Arial, sans-serif;
            font-size: 16px;
        }
        #TitleLabel {
            color: #fff;
            font-size: 48px;
            font-weight: 900;
            letter-spacing: 1.5px;
            margin: 14px 0 0 0;
            background: transparent;
            border: none;
            text-align: center;
            text-shadow: 0 4px 18px #5ad0ff33;
        }
        #SubtitleLabel {
            color: #69cfff;
            font-size: 22px;
            font-weight: 500;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
        }
        #InstrLabel {
            color: #e9f6ff;
            font-size: 23px;
            font-weight: 600;
            margin: 8px 0 18px 0;
            background: rgba(17,32,56,0.17);
            border-radius: 18px;
            padding: 16px 10px;
            border: 1px solid #4fa3e8;
        }
        #LogArea {
            background: rgba(27, 41, 65, 0.68);
            color: #dbefff;
            font-family: 'Consolas', 'Fira Mono', monospace;
            font-size: 15px;
            border: 1px solid #4570a1;
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 18px;
        }
        QLabel#ImgPlaceholder {
            color: #b4d0ea;
            font-size: 18px;
            border: 2px dashed #4fa3e8;
            border-radius: 24px;
            background: rgba(33, 51, 80, 0.12);
            font-style: italic;
            margin-top: 10px;
            padding: 4px;
        }
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #78e3fc, stop:1 #3289d6);
            color: #13437c;
            border: none;
            border-radius: 12px;
            padding: 15px 36px;
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b8eafd, stop:1 #68bdfc);
            color: #1e62a5;
        }
        QPushButton:disabled {
            background: #b8c1cc;
            color: #4d6d92;
            opacity: 0.7;
        }
        '''

    def _append_log(self, text: str):
        self.txt_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def _on_phase_changed(self, phase: str):
        if getattr(self, "in_init_phase", False):
            return
        if phase == 'trigger':
            self.instructions.setText("Spune „medicament”")
            self.circle.stop()
            lcd.send("Astept...")
        elif phase == 'name':
            self.instructions.setText("Spune numele medicamentului…")
            self.circle.start()
            lcd.send("Astept...")
        elif phase == 'processing':
            self.instructions.setText("Procesez imaginea cu EasyOCR…")
            self.circle.stop()
            lcd.send("Cautare...")
            QTimer.singleShot(2000, lambda: lcd.send("Medicamentul..."))
        elif phase == 'completed':
            self.instructions.setText("Procesare completă! ✅")
            self.circle.stop()
            lcd.send("Gata...")
            QTimer.singleShot(120000, lambda: lcd.send("Astept..."))

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "❌ Eroare", msg)
        self.worker.stop()
        self.close()

    def _on_results_ready(self, results):
        self.current_results = results
        if os.path.exists(results['marked_image_path']):
            pix = QPixmap(results['marked_image_path']).scaled(
                self.img_label.width(), self.img_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.img_label.setPixmap(pix)
            self.img_label.setStyleSheet("")
        self.btn_save.setEnabled(True)
        self._append_log("📊 Rezultate complete disponibile pentru salvare")

    def _save_results(self):
        if not self.current_results:
            return
        fname = f"curabot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(self.current_results, f, indent=2, ensure_ascii=False)
            self._append_log(f"💾 Rezultate salvate în: {fname}")
            QMessageBox.information(self, "✅ Succes", f"Rezultatele au fost salvate în:\n{fname}")
        except Exception as e:
            self._append_log(f"❌ Eroare salvare: {e}")
            QMessageBox.warning(self, "⚠️ Eroare", f"Nu am putut salva rezultatele:\n{e}")

    def closeEvent(self, event):
        self.worker.stop()
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
        try:
            self.db = MedDatabase(excel_path=excel_path)
        except Exception as e:
            self.sig_error.emit(f"Eroare DB: {e}")
            return
        try:
            self.source = sr.Microphone()
            with self.source as s:
                self.recognizer.adjust_for_ambient_noise(s, duration=2)
        except Exception as e:
            self.sig_error.emit(f"Microfon indisponibil: {e}")
            self.source = None
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False
        self.sig_finished.emit()

    def get_last_results(self):
        return self.last_results

    def _run(self):
        if self.source is None:
            self.sig_log.emit("ℹ️ Nu am microfon disponibil, ascult doar comenzi din rețea!")
            return
        self.sig_phase.emit('trigger')
        with self.source as s:
            while self._running:
                try:
                    self.sig_phase.emit('trigger')
                    audio = self.recognizer.listen(s, timeout=5, phrase_time_limit=5)
                    try:
                        text = self.recognizer.recognize_google(audio, language='ro-RO').lower()
                        self.sig_log.emit(f"👂 Auzit: '{text}'")
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        self.sig_log.emit(f"⚠️ Eroare serviciu recunoaștere: {e}")
                        continue
                    if strip_accents(self.trigger_word) not in strip_accents(text):
                        continue
                    self.sig_phase.emit('name')
                    self.sig_log.emit("✅ Trigger detectat! Spune numele medicamentului…")
                    audio2 = self.recognizer.listen(s, timeout=10, phrase_time_limit=8)
                    try:
                        name = self.recognizer.recognize_google(audio2, language='ro-RO').lower()
                        self.sig_log.emit(f"💊 Medicament: '{name}'")
                        self.process_medicine(name)
                    except sr.UnknownValueError:
                        self.sig_log.emit("❌ Nu am înțeles numele medicamentului.")
                        self.sig_phase.emit('trigger')
                        continue
                    except sr.RequestError as e:
                        self.sig_log.emit(f"⚠️ Eroare serviciu: {e}")
                        self.sig_phase.emit('trigger')
                        continue
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.sig_log.emit(f"⚠️ Eroare generală: {e}")
                    time.sleep(1)

    def process_medicine(self, medicine_name):
        if self.db is None:
            self.sig_log.emit("❌ Baza de date medicamente nu a fost încărcată corect.")
            self.sig_phase.emit('completed')
            return
        self.sig_phase.emit('processing')
        self.sig_log.emit(f"💊 Medicament (REȚEA/LOCAL): '{medicine_name}'")
        norm = strip_accents(medicine_name)
        candidates = self.db.search_by_prefix(norm, limit=5)
        if not candidates:
            for med in self.db.medicamente:
                if extract_base_name(med['denumire']) == norm:
                    candidates = [med]
                    break
        if not candidates:
            self.sig_log.emit(f"❌ '{medicine_name}' nu a fost găsit în baza de date.")
            self.sig_phase.emit('completed')
            return
        med = candidates[0]
        base_name = self.db.get_base_name(med)
        try:
            photo_path = os.path.join(os.path.dirname(__file__), 'utils', 'meds.jpg')
            capture_picamera(photo_path=photo_path)
            self.sig_log.emit("📷 Poză capturată cu Pi Camera.")
        except Exception as e:
            self.sig_log.emit(f"❌ Eroare la capturare poză: {e}")
            self.sig_phase.emit('completed')
            return
        self.sig_log.emit("🔄 Procesez imaginea cu EasyOCR…")
        try:
            results = easyocr_detect_medicine(base_name, photo_path)
            self.last_results = results
            if results['matches_found']:
                self.sig_log.emit(f"✅ Găsite {results['matches_found']} regiuni cu '{base_name}'")
                if results['center_for_robot']:
                    self.sig_log.emit(f"📦 Coordonată robot: {results['center_for_robot']}")
                    self.sig_log.emit(f"💾 Salvate coordonate robot în: coords_for_robot.json")
            else:
                self.sig_log.emit(f"❌ Nu am găsit '{base_name}' în imagine")
            self.sig_results.emit(results)
            self.sig_phase.emit('completed')
        except Exception as e:
            self.sig_log.emit(f"❌ Eroare procesare: {e}")
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
