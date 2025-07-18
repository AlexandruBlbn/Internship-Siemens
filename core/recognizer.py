import os, time, threading
import speech_recognition as sr
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
from db_medicamente import MedDatabase, strip_accents, extract_base_name
from .camera import capture_picamera
from .lcd import send_lcd_message
from .ocr import easyocr_detect_medicine

class RecognizerWorker(QObject):
    sig_log     = pyqtSignal(str)
    sig_phase   = pyqtSignal(str)
    sig_error   = pyqtSignal(str)
    sig_results = pyqtSignal(dict)
    sig_finished= pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        base = os.path.dirname(__file__)
        self.db = MedDatabase(excel_path=os.path.join(base,'medicamente.xlsx'))
        self.trigger = "medicament"
        self.rec = sr.Recognizer()
        self.source = sr.Microphone()
        with self.source as s:
            self.rec.adjust_for_ambient_noise(s, duration=2)
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self._running:
            self.sig_phase.emit('trigger')
            with self.source as s:
                try:
                    audio = self.rec.listen(s, timeout=5, phrase_time_limit=5)
                    text = self.rec.recognize_google(audio, language='ro-RO').lower()
                    self.sig_log.emit(f"Auzit: '{text}'")
                except:
                    continue
            if strip_accents(self.trigger) in strip_accents(text):
                self.sig_phase.emit('name')
                try:
                    audio2 = self.rec.listen(self.source, timeout=10, phrase_time_limit=8)
                    name = self.rec.recognize_google(audio2, language='ro-RO').lower()
                    self.sig_log.emit(f"Medicament: '{name}'")
                    self.process_medicine(name)
                except Exception as e:
                    self.sig_log.emit(f"Eroare: {e}")
            time.sleep(1)

    def process_medicine(self, med_name):
        self.sig_phase.emit('processing')
        photo = os.path.join(os.path.dirname(__file__), 'meds.png')
        capture_picamera(photo)
        send_lcd_message("Medicamentul…")
        results = easyocr_detect_medicine(med_name, photo)
        self.sig_results.emit(results)
        self.sig_phase.emit('completed')

    def stop(self):
        self._running = False
        self.sig_finished.emit()