import os
import json
from manevre import CALIBRATION_FILE, NUM_HORIZONTAL_AXES, UNREACHABLE_AXIS

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
            4: {'shoulder': 90,  'elbow': 180, 'wrist': 170},
            5: {'shoulder': 90,  'elbow': 180, 'wrist': 180},
            6: {'shoulder': 90,  'elbow': 130, 'wrist': 120}
        }
        self.calibration_data = self.load_calibration()

    def load_calibration(self):
        try:
            if os.path.exists(CALIBRATION_FILE):
                with open(CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                return {int(k): v for k, v in data.items()}
        except Exception:
            pass
        return dict(self.default_calibration)

    def save_calibration(self):
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
        except Exception:
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
        if self.selected_axis is None or self.selected_joint is None:
            return
        servo_index = self.joint_servo_map[self.selected_joint]
        current_angle = robot_arm.angles[servo_index]
        new_angle = int(current_angle + direction)
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
            'elbow':    robot_arm.angles[3],
            'wrist':    robot_arm.angles[4]
        }
        self.calibration_data[self.selected_axis] = axis_data
        self.save_calibration()

    def get_calibration_for_axis(self, axis_index):
        return self.calibration_data.get(
            axis_index,
            self.default_calibration.get(axis_index, {'shoulder': 90, 'elbow': 90, 'wrist': 90})
        )

    def show_calibration_table(self):
        for axis_index in range(NUM_HORIZONTAL_AXES):
            if axis_index == UNREACHABLE_AXIS:
                continue
            cal = self.get_calibration_for_axis(axis_index)
          