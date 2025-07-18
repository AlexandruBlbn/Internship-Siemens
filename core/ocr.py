import os
import json
import time
import cv2
from picamera2 import Picamera2
from manevre import CAMERA_RES, PICKUP_SIGNAL_FILE

class OCRDetector:
    def __init__(self):
        self.pic = Picamera2()
        self.cfg = self.pic.create_preview_configuration(
            main={'format': 'XRGB8888', 'size': CAMERA_RES}
        )
        self.pic.configure(self.cfg)
        self.pic.start()
        self.pic.set_controls({'ExposureTime': 18500})
        time.sleep(1)
        # Path către coordonatele OCR generate de wake_app
        self.ocr_coords_file = '/home/user/Desktop/Internship-Siemens/wake_app/coords_for_robot.json'
        self.last_ocr_point = None
        self.ocr_tracker = None
        self.tracking_active = False
        self.ocr_point_history = []
        self.max_history = 10
        self.last_coords_timestamp = None
        self.vertical_offset = 0
        self.horizontal_offset = 20
        self.offset_step = 5
        self.auto_pickup_enabled = False
        self.text_detector = None
        self.target_medicine_name = None
        self.detected_texts = []
        self.text_detection_enabled = True
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.ocr_point_bypassed = False
        self.init_text_detector()
        self.init_ocr_tracker()

    def init_text_detector(self):
        self.text_detector_ready = True

    def detect_text_regions(self, frame):
        if not self.text_detector_ready:
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                if (
                    0.3 < aspect_ratio < 20 and
                    50 < area < 80000 and
                    w > 15 and h > 6 and
                    w < frame.shape[1] * 0.9 and
                    h < frame.shape[0] * 0.5 and
                    x > 5 and x + w < frame.shape[1] - 5 and
                    y > 5 and y + h < frame.shape[0] - 5
                ):
                    contour_area = area
                    hull_area = cv2.contourArea(cv2.convexHull(contour))
                    if hull_area > 0:
                        solidity = contour_area / hull_area
                        if solidity > 0.05:
                            center_x = x + w // 2
                            center_y = y + h // 2
                            size_score = min(1.0, area / 15000)
                            center_distance = (
                                abs(center_x - frame.shape[1]//2) +
                                abs(center_y - frame.shape[0]//2)
                            )
                            center_score = max(
                                0.2,
                                1.0 - (center_distance / (frame.shape[1] + frame.shape[0]))
                            )
                            solidity_score = min(1.0, solidity * 3)
                            confidence = (
                                size_score * 0.5 +
                                center_score * 0.1 +
                                solidity_score * 0.4
                            )
                            text_regions.append({
                                'bbox': (x, y, w, h),
                                'center': (center_x, center_y),
                                'confidence': confidence,
                                'area': area,
                                'solidity': solidity,
                                'aspect_ratio': aspect_ratio
                            })
            text_regions.sort(key=lambda x: x['confidence'], reverse=True)
            high_conf_regions = [
                r for r in text_regions if r['confidence'] >= 0.9
            ]
            if not high_conf_regions and text_regions:
                high_conf_regions = text_regions[:3]
            return high_conf_regions[:10]
        except Exception:
            return []

    def filter_text_by_medicine_name(self, text_regions, target_name):
        if not target_name or not text_regions:
            return text_regions
        return text_regions

    def process_detected_text_regions(self, text_regions):
        detected_points = []
        try:
            if text_regions:
                filtered = self.filter_text_by_medicine_name(
                    text_regions, self.target_medicine_name
                )
                if filtered:
                    x0, y0 = filtered[0]['center']
                    adj_x = max(0, min(CAMERA_RES[0], x0 + self.horizontal_offset))
                    adj_y = max(0, min(CAMERA_RES[1], y0 - self.vertical_offset))
                    detected_points.append((adj_x, adj_y))
                    self.last_ocr_point = (adj_x, adj_y)
                for region in text_regions[1:3]:
                    x1, y1 = region['center']
                    detected_points.append((
                        max(0, min(CAMERA_RES[0], x1 + self.horizontal_offset)),
                        max(0, min(CAMERA_RES[1], y1 - self.vertical_offset))
                    ))
        except Exception:
            pass
        return detected_points

    def detect(self):
        # verifică semnal extern de override OCR
        update_file = os.path.join(os.path.dirname(__file__), "ocr_point_update.json")
        try:
            if os.path.exists(update_file):
                with open(update_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if (
                    data.get('action') == 'update_ocr_point' and
                    data.get('bypass_movement_laws')
                ):
                    pt = data.get('ocr_point', [])
                    if isinstance(pt, list) and len(pt) == 2:
                        self.last_ocr_point = (pt[0], pt[1])
                        self.ocr_point_bypassed = True
                os.remove(update_file)
                frame = self.pic.capture_array()
                return frame, [self.last_ocr_point] if self.last_ocr_point else []
        except Exception:
            pass

        # semnal auto pickup
        try:
            if os.path.exists(PICKUP_SIGNAL_FILE):
                with open(PICKUP_SIGNAL_FILE, 'r') as f:
                    sig = json.load(f)
                if sig.get('action') == 'pickup_ocr_point' and sig.get('auto_pickup'):
                    self.auto_pickup_enabled = True
                os.remove(PICKUP_SIGNAL_FILE)
        except Exception:
            pass

        frame = self.pic.capture_array()
        self.detected_texts = self.detect_text_regions(frame)

        # încearcă coordonate din fisier + tracking
        coords = []
        if os.path.exists(self.ocr_coords_file):
            stat = os.stat(self.ocr_coords_file)
            if stat.st_mtime != self.last_coords_timestamp:
                self.last_coords_timestamp = stat.st_mtime
                with open(self.ocr_coords_file, 'r', encoding='utf-8') as f:
                    dat = json.load(f)
                if 'center' in dat:
                    cx, cy = dat['center']
                    if 'medicament' in dat:
                        self.target_medicine_name = dat['medicament']
                    coords = [(
                        max(0, min(CAMERA_RES[0], cx + self.horizontal_offset)),
                        max(0, min(CAMERA_RES[1], cy - self.vertical_offset))
                    )]
        if coords:
            return frame, coords

        # fallback pe text_regions detectate
        if self.detected_texts:
            return frame, self.process_detected_text_regions(self.detected_texts)
        return frame, []

    def init_ocr_tracker(self):
        try:
            # Creați tracker CSRT dacă există
            if hasattr(cv2, 'TrackerCSRT_create'):
                self.ocr_tracker = cv2.TrackerCSRT_create()
            elif hasattr(cv2.legacy, 'TrackerCSRT_create'):
                self.ocr_tracker = cv2.legacy.TrackerCSRT_create()
        except Exception:
            self.ocr_tracker = None
        self.tracking_active = False

    def update_ocr_point_for_robot_movement(self, robot_arm):
        # actualizează self.last_ocr_point bazat pe schimbare de bază
        # (codul original pentru compensare rotatională)
        pass  # păstrați implementarea originală dacă doriți

    def reset_text_tracking(self):
        self.detected_texts = []
        self.last_ocr_point = None
        self.ocr_tracker = None
        self.tracking_active = False

    def cleanup(self):
        try:
            self.pic.stop()
            self.pic.close()
        except Exception:
            pass

import os, cv2, difflib, easyocr
from datetime import datetime
from utils.fileio import save_json
from .config import COORDS_FOR_ROBOT, OCR_UPDATE_FILE
from db_medicamente import strip_accents

def easyocr_detect_medicine(med_name, image_path, output_path=None):
    reader = easyocr.Reader(['ro','en'], gpu=False, verbose=False)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # …logica de detectare contour+OCR (copiază din recognition.py)…
    # la final:
    robot_data = {
        'medicament': med_name,
        'center': [camera_x, camera_y],
        'timestamp': datetime.now().isoformat()
    }
    save_json(COORDS_FOR_ROBOT, robot_data)
    ocr_data = {
        'action':'update_ocr_point',
        'ocr_point': [camera_x, camera_y],
        'bypass_movement_laws': True,
        'timestamp': datetime.now().isoformat()
    }
    save_json(OCR_UPDATE_FILE, ocr_data)
    if output_path:
        cv2.imwrite(output_path, img_for_draw)
    return {
        "medicine_name": med_name,
        "matches_found": len(matches),
        "matches": matches,
        "camera_center": (camera_x, camera_y),
        "marked_image_path": output_path
    }