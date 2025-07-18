import time
import serial
from core.face import FaceDetector
from manevre import (
    HOME_POSITION, SHUTDOWN_POSITION, DROP_POSITION,
    VERTICAL_CHECK_POSITION, PERSON_DROP_POSITION,
    UNREACHABLE_AXIS, AXIS_MAPPING, NUM_HORIZONTAL_AXES,
    BACKUP_DROP_SHOULDER_ADJUSTMENT, BACKUP_DROP_ELBOW_ADJUSTMENT,
    BACKUP_DROP_DELAY_POSITION, BACKUP_DROP_DELAY_RELEASE,
    BASE_FINE_STEP, BASE_OVERSHOOT_THRESHOLD,
    CENTER_THRESHOLD, STEP_ANGLE, SERIAL_PORT, BAUDRATE
)

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
            rounded = [int(round(a)) for a in self.angles]
            cmd = ','.join(map(str, rounded)) + '\n'
            self.ser.write(cmd.encode())
            self.ser.flush()
            time.sleep(0.1)
            return True
        except Exception:
            return False

    def center_horizontally(self, pixel_x, frame_width, ocr_detector=None):
        if self.is_centered:
            return True
        center_x = frame_width // 2
        offset = pixel_x - center_x
        if abs(offset) <= BASE_OVERSHOOT_THRESHOLD:
            self.is_centered = True
            return True
        factor = min(1.0, abs(offset) / 100.0)
        if abs(offset) > CENTER_THRESHOLD:
            step = STEP_ANGLE * (1 if offset > 0 else -1)
        else:
            step = BASE_FINE_STEP * (1 if offset > 0 else -1)
        adjusted = step * factor
        if abs(adjusted) < 0.8:
            adjusted = 1.0 * (1 if offset > 0 else -1)
        new_base = round(self.angles[1] + adjusted)
        self.angles[1] = max(0, min(180, new_base))
        self.send_command(silent=True)
        if ocr_detector and hasattr(ocr_detector, 'update_ocr_point_for_robot_movement'):
            ocr_detector.update_ocr_point_for_robot_movement(self)
        time.sleep(0.25)
        return False

    def execute_grip_sequence(self, x, y, frame_height, calibrator, det=None):
        if not self.is_centered:
            return False
        self.is_executing = True
        try:
            axis_height = frame_height / NUM_HORIZONTAL_AXES
            axis_index = int(y / axis_height)
            if axis_index == UNREACHABLE_AXIS:
                return False
            mapped = AXIS_MAPPING.get(axis_index, axis_index)
            cal = calibrator.get_calibration_for_axis(mapped)

            # poziționare umăr, cot, încheietură
            self.angles[2] = cal['shoulder']
            self.send_command(); time.sleep(0.3)
            self.angles[3] = cal['elbow']
            self.send_command(); time.sleep(0.3)
            self.angles[4] = cal['wrist']
            self.send_command(); time.sleep(0.3)

            # pregătire prindere
            self.angles[6] = 76
            self.send_command(); time.sleep(1.5)
            self.angles[2] = max(15, self.angles[2] - 15)
            self.send_command(); time.sleep(1.2)

            # executare drop inteligent
            return self.execute_smart_drop_sequence(det)
        except Exception:
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
        original = list(self.angles)
        try:
            cal = calibrator.get_calibration_for_axis(2)
            self.angles[2] = cal['shoulder']; self.send_command(); time.sleep(0.8)
            self.angles[3] = cal['elbow'] - 3; self.send_command(); time.sleep(0.8)
            self.angles[4] = cal['wrist'] - 30; self.send_command(); time.sleep(0.8)
            self.angles[6] = 76; self.send_command(); time.sleep(1.0)
            for i, a in enumerate(HOME_POSITION):
                self.angles[i] = a
            self.send_command(); time.sleep(1.0)
        except Exception:
            self.angles = original
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
        if not self.face_detector or det is None:
            return False
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                frame, _ = det.detect()
                ok, count = self.face_detector.detect_person(frame)
                if ok:
                    time.sleep(0.5)
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def execute_smart_drop_sequence(self, det=None):
        try:
            # poziție drop
            drops = list(DROP_POSITION); drops[6] = 76
            for i, a in enumerate(drops):
                self.angles[i] = a
            self.send_command(); time.sleep(2.0)

            # poziții verticale pentru verificare persoană
            verts = list(VERTICAL_CHECK_POSITION); verts[6] = 76
            for i, a in enumerate(verts):
                self.angles[i] = a
            self.send_command(); time.sleep(2.0)

            person = self.verify_person_presence(det) if det else False
            if person:
                # drop persoană
                pd = list(PERSON_DROP_POSITION); pd[6] = 76
                for i, a in enumerate(pd):
                    self.angles[i] = a
                self.send_command(); time.sleep(2.0)
                self.angles[6] = 10; self.send_command(); time.sleep(1.5)
                self.reset_to_home()
                return True

            # backup drop
            home = list(HOME_POSITION); home[6] = 76
            for i, a in enumerate(home):
                self.angles[i] = a
            self.send_command(); time.sleep(2.0)
            self.angles[2] = HOME_POSITION[2] + BACKUP_DROP_SHOULDER_ADJUSTMENT
            self.angles[3] = HOME_POSITION[3] + BACKUP_DROP_ELBOW_ADJUSTMENT
            self.send_command(); time.sleep(BACKUP_DROP_DELAY_POSITION)
            self.angles[6] = 10; self.send_command(); time.sleep(BACKUP_DROP_DELAY_RELEASE)
            self.reset_to_home()
            return True
        except Exception:
            try:
                home = list(HOME_POSITION); home[6] = 76
                for i, a in enumerate(home):
                    self.angles[i] = a
                self.send_command(); time.sleep(2.0)
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
