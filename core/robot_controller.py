import sys, json, threading, subprocess
from datetime import datetime
from utils.fileio import save_json
from .config import MANEVRE_SCRIPT, PICKUP_SIGNAL_FILE

class RobotController:
    def __init__(self, parent=None):
        self.parent = parent
        self.process = None
        self.running = False

    def start(self):
        if self.running: return
        self.process = subprocess.Popen(
            [sys.executable, MANEVRE_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        self.running = True
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self):
        while self.running:
            line = self.process.stdout.readline()
            if not line: break
            if self.parent:
                self.parent._append_log(f"[ROBOT] {line.strip()}")

    def send_pickup_signal(self):
        if not self.running: return False
        save_json(PICKUP_SIGNAL_FILE, {
            "action":"pickup_ocr_point",
            "auto_pickup":True,
            "timestamp": datetime.now().isoformat()
        })
        if self.parent:
            self.parent._append_log("Semnal pickup trimis către manevre.py")
        return True

    def stop(self):
        if not self.running: return
        self.process.terminate()
        self.process.wait(timeout=5)
        self.running = False
        if self.parent:
            self.parent._append_log("Procesul manevre.py oprit")