import cv2
import numpy as np
import serial
import time
import json
import os
from datetime import datetime
import serial
from picamera2 import Picamera2
import mediapipe as mp
from lcd_messages import *

LCD_MESSAGE_FILE = '/home/user/Desktop/Internship-Siemens/lcd_message.json'
CAMERA_REQUEST_FILE = "/home/user/Desktop/Internship-Siemens/camera_request.json"
CAMERA_RELEASE_FILE = "/home/user/Desktop/Internship-Siemens/camera_released.json"
PICKUP_SIGNAL_FILE = '/home/user/Desktop/Internship-Siemens/pickup_signal.json'
CAMERA_RES = (1440, 1080)
OCR_HORIZONTAL_PIXELS_PER_DEGREE = 14 
OCR_VERTICAL_PIXELS_PER_DEGREE = 2    
OCR_MOVEMENT_ENABLED = True            
OCR_VERTICAL_MOVEMENT_ENABLED = True
NUM_HORIZONTAL_AXES = 7
UNREACHABLE_AXIS = 6
CENTER_AXIS_INDEX = 3
CALIBRATION_FILE = 'axis_calibration.json'
HOME_POSITION = [30, 90, 140, 90, 130, 90, 10]
DROP_POSITION = [30, 0, 90, 90, 90, 90, 76]
VERTICAL_CHECK_POSITION = [30, 0, 90, 90, 90, 90, 76]
PERSON_DROP_POSITION = [30, 0, 150, 90, 90, 90, 76]
SHUTDOWN_POSITION = [30, 90, 90, 10, 90, 90, 70]
BACKUP_DROP_SHOULDER_ADJUSTMENT = -20
BACKUP_DROP_ELBOW_ADJUSTMENT = 80
BACKUP_DROP_DELAY_POSITION = 1.5
BACKUP_DROP_DELAY_RELEASE = 1.5
BASE_FINE_STEP = 0.3
BASE_OVERSHOOT_THRESHOLD = 12.0
CENTER_THRESHOLD = 40           
STEP_ANGLE = 1
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 9600
AXIS_MAPPING = {i: i for i in range(NUM_HORIZONTAL_AXES)}

class AxisCalibrator:
    def __init__(self):
        self.calibration_mode = False
        self.selected_axis = None
        self.selected_joint = None
        self.joint_names = ['shoulder', 'elbow', 'wrist']
        self.joint_display = {'shoulder': 'SHOULDER', 'elbow': 'ELBOW', 'wrist': 'WRIST'}
        self.joint_servo_map = {'shoulder': 2, 'elbow': 3, 'wrist': 4}
        self.default_calibration = {
            0: {'shoulder': 146, 'elbow': 131, 'wrist': 148},
            1: {'shoulder': 135, 'elbow': 150, 'wrist': 156},
            2: {'shoulder': 115, 'elbow': 170, 'wrist': 150},
            3: {'shoulder': 105, 'elbow': 180, 'wrist': 150},
            4: {'shoulder': 90, 'elbow': 180, 'wrist': 170},
            5: {'shoulder': 90, 'elbow': 180, 'wrist': 180},
            6: {'shoulder': 90, 'elbow': 130, 'wrist': 120}
        }
        self.calibration_data = self.load_calibration()
    def load_calibration(self):
        try:
            if os.path.exists(CALIBRATION_FILE):
                with open(CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                converted_data = {int(k): v for k, v in data.items()}
                return converted_data
        except Exception as e:
            pass
        return dict(self.default_calibration)
    def save_calibration(self):
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
        except Exception as e:
            pass
    def enter_calibration_mode(self):
        self.calibration_mode = True
        self.selected_axis = None
        self.selected_joint = None
    def exit_calibration_mode(self):
        self.calibration_mode = False
        self.selected_axis = None
        self.selected_joint = None
    def select_axis(self, axis_index):
        if 0 <= axis_index < NUM_HORIZONTAL_AXES:
            if axis_index == UNREACHABLE_AXIS:
                return
            self.selected_axis = axis_index
            if axis_index not in self.calibration_data:
                self.calibration_data[axis_index] = {'shoulder': 90, 'elbow': 90, 'wrist': 90}
    def select_joint(self, joint_name):
        if joint_name in self.joint_names:
            self.selected_joint = joint_name
    def adjust_angle(self, direction, robot_arm):
        if self.selected_axis is None:
            return
        if self.selected_joint is None:
            return
        servo_index = self.joint_servo_map[self.selected_joint]
        current_angle = robot_arm.angles[servo_index]
        new_angle = int(current_angle + (direction * 1))
        if servo_index == 2:
            new_angle = max(15, min(165, new_angle))
        else:
            new_angle = max(0, min(180, new_angle))
        robot_arm.angles[servo_index] = new_angle
        robot_arm.send_command()
    def save_current_position(self, robot_arm):
        if self.selected_axis is None:
            return
        axis_data = {
            'shoulder': robot_arm.angles[2],
            'elbow': robot_arm.angles[3],
            'wrist': robot_arm.angles[4]
        }
        self.calibration_data[self.selected_axis] = axis_data
        self.save_calibration()
    def get_calibration_for_axis(self, axis_index):
        if axis_index in self.calibration_data:
            return self.calibration_data[axis_index]
        else:
            return self.default_calibration.get(axis_index, {'shoulder': 90, 'elbow': 90, 'wrist': 90})
    def show_calibration_table(self):
        for axis_index in range(NUM_HORIZONTAL_AXES):
            if axis_index == UNREACHABLE_AXIS:
                continue
            cal = self.get_calibration_for_axis(axis_index)
    def get_interpolated_calibration(self, y, frame_height):
        axis_height = frame_height / NUM_HORIZONTAL_AXES
        idx = int(y / axis_height)
        idx = max(0, min(NUM_HORIZONTAL_AXES - 1, idx))
        next_idx = min(idx + 1, NUM_HORIZONTAL_AXES - 1)
        ratio = (y - idx * axis_height) / axis_height
        cal1 = self.get_calibration_for_axis(idx)
        cal2 = self.get_calibration_for_axis(next_idx)
        return {
            'shoulder': int(round(cal1['shoulder'] * (1 - ratio) + cal2['shoulder'] * ratio)),
            'elbow':    int(round(cal1['elbow']    * (1 - ratio) + cal2['elbow']    * ratio)),
            'wrist':    int(round(cal1['wrist']    * (1 - ratio) + cal2['wrist']    * ratio))
        }

class OCRDetector:
    def __init__(self):
        self.pic = Picamera2()
        self.cfg = self.pic.create_preview_configuration(main={'format': 'XRGB8888', 'size': CAMERA_RES})
        self.pic.configure(self.cfg)
        self.pic.start()
        self.pic.set_controls({'ExposureTime': 18500})
        time.sleep(1)
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
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                if (0.3 < aspect_ratio < 20 and 50 < area < 80000 and w > 15 and h > 6 and
                    w < frame.shape[1] * 0.9 and h < frame.shape[0] * 0.5 and
                    x > 5 and x + w < frame.shape[1] - 5 and y > 5 and y + h < frame.shape[0] - 5):
                    contour_area = cv2.contourArea(contour)
                    hull_area = cv2.contourArea(cv2.convexHull(contour))
                    if hull_area > 0:
                        solidity = contour_area / hull_area
                        if solidity > 0.05:
                            center_x = x + w // 2
                            center_y = y + h // 2
                            size_score = min(1.0, area / 15000)
                            center_distance = abs(center_x - frame.shape[1]//2) + abs(center_y - frame.shape[0]//2)
                            center_score = max(0.2, 1.0 - (center_distance / (frame.shape[1] + frame.shape[0])))
                            solidity_score = min(1.0, solidity * 3)
                            confidence = (size_score * 0.5 + center_score * 0.1 + solidity_score * 0.4)
                            text_regions.append({
                                'bbox': (x, y, w, h),
                                'center': (center_x, center_y),
                                'confidence': confidence,
                                'area': area,
                                'solidity': solidity,
                                'aspect_ratio': aspect_ratio
                            })
            text_regions.sort(key=lambda x: x['confidence'], reverse=True)
            high_confidence_regions = [region for region in text_regions if region['confidence'] >= 0.9]
            if not high_confidence_regions and text_regions:
                high_confidence_regions = text_regions[:3]
            final_regions = high_confidence_regions[:10]
            return final_regions
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
                filtered_regions = self.filter_text_by_medicine_name(text_regions, self.target_medicine_name)
                if filtered_regions:
                    primary_region = filtered_regions[0]
                    center = primary_region['center']
                    adjusted_x = center[0] + self.horizontal_offset
                    adjusted_y = center[1] - self.vertical_offset
                    adjusted_x = max(0, min(CAMERA_RES[0], adjusted_x))
                    adjusted_y = max(0, min(CAMERA_RES[1], adjusted_y))
                    final_point = (adjusted_x, adjusted_y)
                    detected_points.append(final_point)
                    if not self.last_ocr_point or self.last_ocr_point != final_point:
                        self.last_ocr_point = final_point
                for region in text_regions[1:3]:
                    center = region['center']
                    adjusted_x = center[0] + self.horizontal_offset
                    adjusted_y = center[1] - self.vertical_offset
                    adjusted_x = max(0, min(CAMERA_RES[0], adjusted_x))
                    adjusted_y = max(0, min(CAMERA_RES[1], adjusted_y))
                    detected_points.append((adjusted_x, adjusted_y))
        except Exception:
            pass
        return detected_points
    def detect_and_track_text(self, frame):
        detected_points = []
        try:
            if not self.last_ocr_point:
                text_regions = self.detect_text_regions(frame)
                self.detected_texts = text_regions
                if text_regions:
                    filtered_regions = self.filter_text_by_medicine_name(text_regions, self.target_medicine_name)
                    if filtered_regions:
                        primary_region = filtered_regions[0]
                        center = primary_region['center']
                        adjusted_x = center[0] + self.horizontal_offset
                        adjusted_y = center[1] - self.vertical_offset
                        adjusted_x = max(0, min(CAMERA_RES[0], adjusted_x))
                        adjusted_y = max(0, min(CAMERA_RES[1], adjusted_y))
                        final_point = (adjusted_x, adjusted_y)
                        detected_points.append(final_point)
                        self.last_ocr_point = final_point
                    for region in text_regions[1:3]:
                        center = region['center']
                        adjusted_x = center[0] + self.horizontal_offset
                        adjusted_y = center[1] - self.vertical_offset
                        adjusted_x = max(0, min(CAMERA_RES[0], adjusted_x))
                        adjusted_y = max(0, min(CAMERA_RES[1], adjusted_y))
                        detected_points.append((adjusted_x, adjusted_y))
        except Exception:
            pass
        return detected_points
    def set_target_medicine(self, medicine_name):
        self.target_medicine_name = medicine_name
        self.detected_texts = []
        self.last_ocr_point = None
    def draw_text_detection_debug(self, frame):
        for region in self.detected_texts:
            x, y, w, h = region['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{region['confidence']:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.last_ocr_point:
            cv2.circle(frame, (int(self.last_ocr_point[0]), int(self.last_ocr_point[1])), 10, (255, 0, 0), 2)
    def reset_text_tracking(self):
        self.detected_texts = []
        self.last_ocr_point = None
    def check_auto_pickup_signal(self):
        try:
            if os.path.exists(PICKUP_SIGNAL_FILE):
                with open(PICKUP_SIGNAL_FILE, 'r') as f:
                    signal_data = json.load(f)
                if signal_data.get('action') == 'pickup_ocr_point' and signal_data.get('auto_pickup'):
                    self.auto_pickup_enabled = True
                os.remove(PICKUP_SIGNAL_FILE)
        except Exception:
            pass
    def detect(self):
        ocr_update_file = os.path.join(os.path.dirname(__file__), "ocr_point_update.json")
        try:
            if os.path.exists(ocr_update_file):
                with open(ocr_update_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('action') == 'update_ocr_point' and data.get('bypass_movement_laws'):
                    pt = data.get('ocr_point', [])
                    if isinstance(pt, list) and len(pt) == 2:
                        self.last_ocr_point = (pt[0], pt[1])
                        self.ocr_point_bypassed = True
                os.remove(ocr_update_file)
                frame = self.pic.capture_array()
                return frame, [self.last_ocr_point] if self.last_ocr_point else []
        except Exception:
            pass
        self.check_auto_pickup_signal()
        frame = self.pic.capture_array()
        text_regions = self.detect_text_regions(frame)
        self.detected_texts = text_regions
        ocr_points = self.load_and_interpolate_ocr_coordinates()
        all_points = []
        if ocr_points:
            tracked_points = self.track_ocr_point(frame)
            if tracked_points:
                all_points = tracked_points
            else:
                all_points = ocr_points
        elif text_regions:
            text_points = self.process_detected_text_regions(text_regions)
            all_points = text_points
        return frame, all_points
    def load_and_interpolate_ocr_coordinates(self):
        try:
            if os.path.exists(self.ocr_coords_file):
                file_stat = os.stat(self.ocr_coords_file)
                file_timestamp = file_stat.st_mtime
                if self.last_coords_timestamp != file_timestamp:
                    self.last_coords_timestamp = file_timestamp
                    with open(self.ocr_coords_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'center' in data and data['center']:
                        ocr_x, ocr_y = data['center']
                        if 'medicament' in data and data['medicament']:
                            self.set_target_medicine(data['medicament'])
                        ocr_x += self.horizontal_offset
                        ocr_y -= self.vertical_offset
                        ocr_x = max(0, min(CAMERA_RES[0], ocr_x))
                        ocr_y = max(0, min(CAMERA_RES[1], ocr_y))
                        if 0 <= ocr_x <= CAMERA_RES[0] and 0 <= ocr_y <= CAMERA_RES[1]:
                            if self.last_ocr_point != (ocr_x, ocr_y):
                                self.last_ocr_point = (ocr_x, ocr_y)
                                self.tracking_active = False
                            return [(ocr_x, ocr_y)]
                if self.last_ocr_point:
                    return [self.last_ocr_point]
        except Exception:
            pass
        return []
    def init_ocr_tracker(self):
        try:
            if hasattr(cv2, 'TrackerCSRT_create'):
                self.ocr_tracker = cv2.TrackerCSRT_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                self.ocr_tracker = cv2.legacy.TrackerCSRT_create()
            else:
                self.ocr_tracker = None
        except Exception:
            self.ocr_tracker = None
        self.tracking_active = False
    def track_ocr_point(self, frame):
        if not self.last_ocr_point:
            return []
        try:
            if not self.tracking_active:
                x, y = self.last_ocr_point
                bbox_size = 40
                bbox = (max(0, x - bbox_size // 2),
                        max(0, y - bbox_size // 2),
                        min(bbox_size, CAMERA_RES[0] - x + bbox_size // 2),
                        min(bbox_size, CAMERA_RES[1] - y + bbox_size // 2))
                if self.ocr_tracker:
                    success = self.ocr_tracker.init(frame, bbox)
                    if success:
                        self.tracking_active = True
                        return [self.last_ocr_point]
                return [self.last_ocr_point]
            else:
                success, bbox = self.ocr_tracker.update(frame)
                if success:
                    x = int(bbox[0] + bbox[2] / 2)
                    y = int(bbox[1] + bbox[3] / 2)
                    old_x, old_y = self.last_ocr_point
                    max_movement = 50
                    if abs(x - old_x) > max_movement or abs(y - old_y) > max_movement:
                        self.tracking_active = False
                        self.ocr_point_history.clear()
                        return [self.last_ocr_point]
                    self.ocr_point_history.append((x, y))
                    if len(self.ocr_point_history) > self.max_history:
                        self.ocr_point_history.pop(0)
                    if self.ocr_point_history:
                        avg_x = int(sum(p[0] for p in self.ocr_point_history) / len(self.ocr_point_history))
                        avg_y = int(sum(p[1] for p in self.ocr_point_history) / len(self.ocr_point_history))
                        self.last_ocr_point = (avg_x, avg_y)
                        return [(avg_x, avg_y)]
                else:
                    self.tracking_active = False
                    self.ocr_point_history.clear()
                    return [self.last_ocr_point] if self.last_ocr_point else []
        except Exception:
            self.tracking_active = False
            return [self.last_ocr_point] if self.last_ocr_point else []
        return [self.last_ocr_point] if self.last_ocr_point else []
    def toggle_debug_mode(self):
        self.debug_mode = not getattr(self, "debug_mode", False)
    def draw_circular_path_debug(self, frame):
        pass
    def cleanup(self):
        try:
            self.pic.stop()
            self.pic.close()
            self.detected_texts = []
        except Exception:
            pass
    def update_ocr_point_for_robot_movement(self, robot_arm):
        global OCR_VERTICAL_MOVEMENT_ENABLED
        if not OCR_MOVEMENT_ENABLED or not hasattr(self, 'last_ocr_point') or self.last_ocr_point is None:
            return
        current_base = robot_arm.angles[1]
        if not hasattr(robot_arm, 'previous_base_angle') or robot_arm.previous_base_angle is None:
            robot_arm.previous_base_angle = current_base
            return
        base_change = current_base - robot_arm.previous_base_angle
        if abs(base_change) < 0.1:
            return
        old_x, old_y = self.last_ocr_point
        center_x = CAMERA_RES[0] // 2
        center_y = CAMERA_RES[1] // 2
        distance_from_center_x = abs(old_x - center_x) / (CAMERA_RES[0] // 2)
        distance_from_center_y = abs(old_y - center_y) / (CAMERA_RES[1] // 2)
        is_left = old_x < center_x
        is_top = old_y < center_y
        base_compensation_x = 1.2
        if distance_from_center_x > 0.6:
            compensation_factor_x = base_compensation_x + (distance_from_center_x ** 1.3) * 1.0
        else:
            compensation_factor_x = base_compensation_x + (distance_from_center_x ** 1.8) * 0.6
        if distance_from_center_y < 0.4:
            compensation_factor_y = 0.8 - (distance_from_center_y * 0.3)
        elif distance_from_center_y < 0.7:
            compensation_factor_y = 1.0 + (distance_from_center_y ** 1.5) * 0.2
        else:
            compensation_factor_y = 1.0 + (distance_from_center_y ** 1.2) * 0.15
        if distance_from_center_x > 0.8:
            compensation_factor_x *= 1.2
        if distance_from_center_y > 0.8:
            compensation_factor_y *= 0.9
        horizontal_movement = -base_change * OCR_HORIZONTAL_PIXELS_PER_DEGREE * compensation_factor_x
        if OCR_VERTICAL_MOVEMENT_ENABLED:
            if is_left:
                vertical_movement = -base_change * OCR_VERTICAL_PIXELS_PER_DEGREE * compensation_factor_y
            else:
                vertical_movement = base_change * OCR_VERTICAL_PIXELS_PER_DEGREE * compensation_factor_y
            new_y = old_y - vertical_movement
        else:
            new_y = old_y
        new_x = old_x + horizontal_movement
        new_x = max(50, min(CAMERA_RES[0] - 50, new_x))
        new_y = max(50, min(CAMERA_RES[1] - 50, new_y))
        self.last_ocr_point = (int(new_x), int(new_y))
        robot_arm.previous_base_angle = current_base
    def adjust_ocr_movement_settings(self, horizontal_change=0, vertical_change=0):
        global OCR_HORIZONTAL_PIXELS_PER_DEGREE, OCR_VERTICAL_PIXELS_PER_DEGREE
        if horizontal_change:
            OCR_HORIZONTAL_PIXELS_PER_DEGREE = max(1, OCR_HORIZONTAL_PIXELS_PER_DEGREE + horizontal_change)
        if vertical_change:
            OCR_VERTICAL_PIXELS_PER_DEGREE = max(1, OCR_VERTICAL_PIXELS_PER_DEGREE + vertical_change)
    def toggle_ocr_movement(self):
        global OCR_MOVEMENT_ENABLED
        OCR_MOVEMENT_ENABLED = not OCR_MOVEMENT_ENABLED
        return OCR_MOVEMENT_ENABLED
    def toggle_ocr_vertical_movement(self):
        global OCR_VERTICAL_MOVEMENT_ENABLED
        OCR_VERTICAL_MOVEMENT_ENABLED = not OCR_VERTICAL_MOVEMENT_ENABLED
        return OCR_VERTICAL_MOVEMENT_ENABLED
    def reset_ocr_tracking(self):
        self.tracking_active = False
        self.ocr_tracker = None
        self.ocr_point_history.clear()
        self.last_ocr_point = None
    def draw_compensation_zones_debug(self, frame):
        pass

class RobotArm:
    def __init__(self):
        self.ser = None
        self.angles = list(HOME_POSITION)
        self.is_connected = False
        self.is_centered = False
        self.is_executing = False
        self.previous_base_angle = HOME_POSITION[1]
        try:
            self.face_detector = FaceDetector()
        except Exception:
            self.face_detector = None
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            time.sleep(2)
            self.is_connected = True
        except Exception:
            pass
    def send_command(self, silent=False):
        if not self.is_connected:
            return False
        try:
            self.ser.reset_input_buffer()
            rounded_angles = [int(round(angle)) for angle in self.angles]
            command = ','.join(map(str, rounded_angles)) + '\n'
            self.ser.write(command.encode())
            self.ser.flush()
            time.sleep(0.1)
            return True
        except Exception:
            return False
    def center_horizontally(self, pixel_x, frame_width, ocr_detector=None):
        if self.is_centered:
            return True
        center_x = frame_width // 2
        offset_x = pixel_x - center_x
        if abs(offset_x) <= BASE_OVERSHOOT_THRESHOLD:
            self.is_centered = True
            return True
        distance_factor = min(1.0, abs(offset_x) / 100.0)
        if abs(offset_x) > CENTER_THRESHOLD:
            base_step = STEP_ANGLE * (1 if offset_x > 0 else -1)
        else:
            base_step = BASE_FINE_STEP * (1 if offset_x > 0 else -1)
        adjusted_step = base_step * distance_factor
        if abs(adjusted_step) < 0.8:
            adjusted_step = 1.0 * (1 if offset_x > 0 else -1)
        new_base = round(self.angles[1] + adjusted_step)
        self.angles[1] = max(0, min(180, new_base))
        self.send_command(silent=True)
        if ocr_detector and hasattr(ocr_detector, 'update_ocr_point_for_robot_movement'):
            ocr_detector.update_ocr_point_for_robot_movement(self)
        time.sleep(0.25)
        return False
    def execute_grip_sequence(self, x, y, frame_height, calibrator, det=None):
        if not self.is_centered:
            return
        self.is_executing = True
        try:
            axis_height = frame_height / NUM_HORIZONTAL_AXES
            axis_index = int(y / axis_height)
            position_in_axis = (y % axis_height) / axis_height
            if axis_index == UNREACHABLE_AXIS:
                self.is_executing = False
                return False
            mapped_axis = AXIS_MAPPING.get(axis_index, axis_index)
            cal = calibrator.get_interpolated_calibration(y, frame_height)
            self.angles[2] = cal['shoulder']
            self.send_command()
            time.sleep(0.3)
            self.angles[3] = cal['elbow']
            self.send_command()
            time.sleep(0.3)
            self.angles[4] = cal['wrist']
            self.send_command()
            time.sleep(0.3)
            self.angles[6] = 76
            self.send_command()
            time.sleep(1.5)
            self.angles[2] = max(15, self.angles[2] - 15)
            self.send_command()
            time.sleep(1.2)
            drop_success = self.execute_smart_drop_sequence(det)
            return True
        except Exception:
            self.is_executing = False
            try:
                self.reset_to_home()
            except Exception:
                pass
            return False
        finally:
            self.is_executing = False
    def reset_to_home(self):
        self.is_centered = False
        self.is_executing = False
        for i, angle in enumerate(HOME_POSITION):
            self.angles[i] = angle
        self.previous_base_angle = self.angles[1]
        self.send_command()
        time.sleep(1.0)
    def perform_warmup_sequence(self, calibrator):
        original_angles = list(self.angles)
        try:
            cal = calibrator.get_calibration_for_axis(2)
            self.angles[2] = cal['shoulder']
            self.send_command()
            time.sleep(0.8)
            self.angles[3] = cal['elbow'] - 3
            self.send_command()
            time.sleep(0.8)
            self.angles[4] = cal['wrist'] -30
            self.send_command()
            time.sleep(0.8)
            self.angles[6] = 76
            self.send_command()
            time.sleep(1.0)
            for i, angle in enumerate(HOME_POSITION):
                self.angles[i] = angle
            self.send_command()
            time.sleep(1.0)
        except Exception:
            self.angles = original_angles
            self.send_command()
    def safe_shutdown(self):
        time.sleep(0.5)
        for i, angle in enumerate(SHUTDOWN_POSITION):
            self.angles[i] = angle
        self.send_command()
        time.sleep(1.5)
        if self.ser and self.ser.is_open:
            self.ser.close()
    def verify_person_presence(self, det=None, timeout_seconds=8):
        if not self.face_detector:
            return False
        if not det:
            return False
        start_time = time.time()
        detection_attempts = 0
        while time.time() - start_time < timeout_seconds:
            detection_attempts += 1
            try:
                current_frame, _ = det.detect()
                person_detected, num_faces = self.face_detector.detect_person(current_frame)
                if person_detected:
                    time.sleep(0.5)
                    return True
                time.sleep(0.1)
            except Exception:
                time.sleep(0.1)
                continue
        return False
    def execute_smart_drop_sequence(self, det=None):
        try:
            drop_angles = list(DROP_POSITION)
            drop_angles[6] = 76
            for i, angle in enumerate(drop_angles):
                self.angles[i] = angle
            self.send_command()
            time.sleep(2.0)
            vertical_angles = list(VERTICAL_CHECK_POSITION)
            vertical_angles[6] = 76
            for i, angle in enumerate(vertical_angles):
                self.angles[i] = angle
            self.send_command()
            time.sleep(2.0)
            if det is not None:
                person_detected = self.verify_person_presence(det, timeout_seconds=8)
            else:
                person_detected = False
            if person_detected:
                person_drop_angles = list(PERSON_DROP_POSITION)
                person_drop_angles[6] = 76
                for i, angle in enumerate(person_drop_angles):
                    self.angles[i] = angle
                self.send_command()
                time.sleep(2.0)
                self.angles[6] = 10
                self.send_command()
                time.sleep(1.5)
                self.reset_to_home()
                return True
            else:
                home_angles = list(HOME_POSITION)
                home_angles[6] = 76
                for i, angle in enumerate(home_angles):
                    self.angles[i] = angle
                self.send_command()
                time.sleep(2.0)
                self.angles[2] = HOME_POSITION[2] + BACKUP_DROP_SHOULDER_ADJUSTMENT
                self.angles[3] = HOME_POSITION[3] + BACKUP_DROP_ELBOW_ADJUSTMENT
                self.send_command()
                time.sleep(BACKUP_DROP_DELAY_POSITION)
                self.angles[6] = 10
                self.send_command()
                time.sleep(BACKUP_DROP_DELAY_RELEASE)
                self.reset_to_home()
                return True
        except Exception:
            try:
                home_angles = list(HOME_POSITION)
                home_angles[6] = 76
                for i, angle in enumerate(home_angles):
                    self.angles[i] = angle
                self.send_command()
                time.sleep(2.0)
            except Exception:
                pass
            return False
    def configure_backup_drop_position(self, shoulder_adj=None, elbow_adj=None, 
                                       delay_position=None, delay_release=None):
        global BACKUP_DROP_SHOULDER_ADJUSTMENT, BACKUP_DROP_ELBOW_ADJUSTMENT
        global BACKUP_DROP_DELAY_POSITION, BACKUP_DROP_DELAY_RELEASE
        if shoulder_adj is not None:
            BACKUP_DROP_SHOULDER_ADJUSTMENT = shoulder_adj
        if elbow_adj is not None:
            BACKUP_DROP_ELBOW_ADJUSTMENT = elbow_adj
        if delay_position is not None:
            BACKUP_DROP_DELAY_POSITION = delay_position
        if delay_release is not None:
            BACKUP_DROP_DELAY_RELEASE = delay_release
    def show_backup_drop_config(self):
        pass

def check_camera_release(det):
    if os.path.exists(CAMERA_REQUEST_FILE):
        try:
            det.cleanup()
            time.sleep(0.5)
            with open(CAMERA_RELEASE_FILE, "w") as f:
                json.dump({"status": "released"}, f)
        except Exception:
            pass
        while os.path.exists(CAMERA_REQUEST_FILE):
            time.sleep(0.2)
        det.__init__()

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
    def detect_person(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_detection.process(rgb_frame)
            if results.detections:
                num_faces = len(results.detections)
                return True, num_faces
            else:
                return False, 0
        except Exception:
            return False, 0
    def draw_detections(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_detection.process(rgb_frame)
            rgb_frame.flags.writeable = True
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    self.mp_drawing.draw_detection(bgr_frame, detection)
            return bgr_frame
        except Exception:
            return frame

def main():
    arm = None
    det = None
    calibrator = None
    face_det = None
    window_name = 'preview pentru camera (orinentativ)'
    try:
        calibrator = AxisCalibrator()
        arm = RobotArm()
        det = OCRDetector()
        face_det = FaceDetector()
        if arm.is_connected:
            arm.reset_to_home()
            time.sleep(1.0)
            arm.perform_warmup_sequence(calibrator)
        window_name = 'Camera Preview - OCR Detection'
        selected_point = None
        show_compensation_zones = False
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        while True:
            check_camera_release(det)
            frame, points = det.detect()
            frame_height, frame_width = frame.shape[:2]
            center_x = frame_width // 2
            if det.auto_pickup_enabled and not arm.is_executing and len(points) > 0:
                for idx, (x, y) in enumerate(points):
                    axis_index = int(y / (frame_height / NUM_HORIZONTAL_AXES))
                    if axis_index != UNREACHABLE_AXIS:
                        selected_point = idx
                        arm.is_centered = False
                        det.auto_pickup_enabled = False
                        break
            for i in range(NUM_HORIZONTAL_AXES):
                axis_y = int((i + 0.5) * frame_height / NUM_HORIZONTAL_AXES)
                if i == UNREACHABLE_AXIS:
                    color = (0, 0, 255)
                elif i == CENTER_AXIS_INDEX:
                    color = (0, 255, 255)
                else:
                    color = (128, 128, 128)
                cv2.line(frame, (0, axis_y), (frame_width, axis_y), color, 1)
                cv2.putText(frame, f"AXA {i}",
                            (frame_width//2 - 50, axis_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if i != UNREACHABLE_AXIS:
                    mapped_i = AXIS_MAPPING.get(i, i)
                    cal = calibrator.get_calibration_for_axis(mapped_i)
                    cal_label = f"S{cal['shoulder']} E{cal['elbow']} W{cal['wrist']}"
                    cv2.putText(frame, cal_label, (10, axis_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 255, 0), 2)
            threshold_left = center_x - BASE_OVERSHOOT_THRESHOLD
            threshold_right = center_x + BASE_OVERSHOOT_THRESHOLD
            cv2.line(frame, (int(threshold_left), 0), (int(threshold_left), frame_height), (0, 255, 0), 1)
            cv2.line(frame, (int(threshold_right), 0), (int(threshold_right), frame_height), (0, 255, 0), 1)
            for i, (x, y) in enumerate(points):
                axis_index = int(y / (frame_height / NUM_HORIZONTAL_AXES))
                cv2.circle(frame, (x, y), 15, (0, 255, 255) if i == 0 else (255, 0, 0), -1)
                info_text = f"Punct {i+1} pe axa {axis_index}"
                cv2.putText(frame, info_text, (x + 10, y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            det.draw_text_detection_debug(frame)
            det.draw_circular_path_debug(frame)
            person_detected, num_faces = face_det.detect_person(frame)
            person_status_text = ""
            if person_detected:
                person_status_text = f"[OK] Persoana detectata ({num_faces} fete)"
            else:
                person_status_text = "[X] Nicio persoana detectata"
            cv2.putText(frame, person_status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if person_detected else (255, 255, 0), 2)
            pickup_allowed = True
            status = f"Connected: {arm.is_connected} | Puncte: {len(points)} | Base: {'LOCKED' if arm.is_centered else 'UNLOCKED'}"
            if det.auto_pickup_enabled:
                status += " | AUTO-PICKUP: ENABLED"
            if hasattr(det, 'text_detection_enabled'):
                status += " | Text Detection: AUTO (Always ON)"
                if hasattr(det, 'detected_texts'):
                    status += f" | Text Regions: {len(det.detected_texts)}"
                if hasattr(det, 'target_medicine_name') and det.target_medicine_name:
                    status += f" | Target: {det.target_medicine_name}"
            if show_compensation_zones:
                det.draw_compensation_zones_debug(frame)
            cv2.imshow(window_name, frame)
            if selected_point is not None and selected_point < len(points) and not arm.is_executing:
                x, y = points[selected_point]
                axis_index = int(y / (frame_height / NUM_HORIZONTAL_AXES))
                if axis_index == UNREACHABLE_AXIS:
                    selected_point = None
                else:
                    if not arm.is_centered:
                        arm.center_horizontally(x, frame_width, det)
                    else:
                        arm.execute_grip_sequence(x, y, frame_height, calibrator, det=det)
                        selected_point = None
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                if calibrator.calibration_mode:
                    calibrator.exit_calibration_mode()
                else:
                    break
            elif calibrator.calibration_mode:
                if ord('1') <= key <= ord('7'):
                    axis_index = key - ord('1')
                    calibrator.select_axis(axis_index)
                elif key == ord('t'):
                    calibrator.select_joint('shoulder')
                elif key == ord('y'):
                    calibrator.select_joint('elbow')
                elif key == ord('u'):
                    calibrator.select_joint('wrist')
                elif key == ord('w'):
                    calibrator.adjust_angle(1, arm)
                elif key == ord('s'):
                    calibrator.adjust_angle(-1, arm)
                elif key == ord('f'):
                    calibrator.save_current_position(arm)
            else:
                if ord('1') <= key <= ord('9'):
                    point_index = key - ord('1')
                    if point_index < len(points):
                        if not arm.is_executing:
                            x, y = points[point_index]
                            axis_index = int(y / (frame_height / NUM_HORIZONTAL_AXES))
                            if axis_index != UNREACHABLE_AXIS:
                                selected_point = point_index
                                arm.is_centered = False
                    else:
                        pass
                elif key == ord('c'):
                    calibrator.enter_calibration_mode()
                elif key == ord('t'):
                    calibrator.show_calibration_table()
                elif key == ord('r'):
                    arm.reset_to_home()
                    selected_point = None
                elif key == ord('d'):
                    det.toggle_debug_mode()
                elif key == ord('o'):
                    if hasattr(det, 'reset_ocr_tracking'):
                        det.reset_ocr_tracking()
                elif key == ord('+') or key == ord('='):
                    det.vertical_offset = min(100, det.vertical_offset + det.offset_step)
                elif key == ord('-'):
                    det.vertical_offset = max(-100, det.vertical_offset - det.offset_step)
                elif key == ord('q'):
                    det.horizontal_offset = max(-50, det.horizontal_offset - det.offset_step)
                elif key == ord('w'):
                    det.horizontal_offset = min(50, det.horizontal_offset + det.offset_step)
                elif key == ord('h'):
                    det.adjust_ocr_movement_settings(horizontal_change=1)
                elif key == ord('g'):
                    det.adjust_ocr_movement_settings(horizontal_change=-1)
                elif key == ord('v'):
                    det.adjust_ocr_movement_settings(vertical_change=1)
                elif key == ord('b'):
                    det.adjust_ocr_movement_settings(vertical_change=-1)
                elif key == ord('m'):
                    det.toggle_ocr_movement()
                elif key == ord('n'):
                    det.toggle_ocr_vertical_movement()
                elif key == ord('z'):
                    show_compensation_zones = not show_compensation_zones
                elif key == ord('x'):
                    det.reset_text_tracking()
                elif key == ord('l'):
                    test_medicine = "Nurofen"
                    det.set_target_medicine(test_medicine)
                elif key == ord('p'):
                    arm.configure_backup_drop_position(shoulder_adj=-10, elbow_adj=10, 
                                                      delay_position=2.0, delay_release=1.0)
                elif key == ord('k'):
                    arm.configure_backup_drop_position(shoulder_adj=-5, elbow_adj=5, 
                                                      delay_position=1.0, delay_release=1.0)
                elif key == ord('j'):
                    arm.configure_backup_drop_position(shoulder_adj=-20, elbow_adj=20, 
                                                      delay_position=2.5, delay_release=2.0)
                elif key == ord('i'):
                    arm.show_backup_drop_config()
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        if arm:
            arm.safe_shutdown()
        if det:
            det.cleanup()

if __name__ == '__main__':
    main()