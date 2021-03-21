# Adapted from https://github.com/anki/vector-python-sdk/blob/master/examples/apps/remote_control/remote_control.py
import argparse
import sys

import anki_vector
import pyglet
from pyglet.window import key

import vector_utils as utils


class RemoteControlVector:
    def __init__(self, robot):
        self.vector = robot
        self.last_lift = None
        self.last_head = None
        self.last_wheels = None
        self.drive_forwards = 0
        self.drive_back = 0
        self.turn_left = 0
        self.turn_right = 0
        self.lift_up = 0
        self.lift_down = 0
        self.head_up = 0
        self.head_down = 0
        self.go_fast = 0
        self.go_slow = 0

    def update_drive_state(self, key_code, is_key_down, speed_changed):
        update_driving = True
        if key_code == ord('W'):
            self.drive_forwards = is_key_down
        elif key_code == ord('S'):
            self.drive_back = is_key_down
        elif key_code == ord('A'):
            self.turn_left = is_key_down
        elif key_code == ord('D'):
            self.turn_right = is_key_down
        else:
            if not speed_changed:
                update_driving = False
        return update_driving

    def update_lift_state(self, key_code, is_key_down, speed_changed):
        update_lift = True
        if key_code == ord('R'):
            self.lift_up = is_key_down
        elif key_code == ord('F'):
            self.lift_down = is_key_down
        else:
            if not speed_changed:
                update_lift = False
        return update_lift

    def update_head_state(self, key_code, is_key_down, speed_changed):
        update_head = True
        if key_code == ord('T'):
            self.head_up = is_key_down
        elif key_code == ord('G'):
            self.head_down = is_key_down
        else:
            if not speed_changed:
                update_head = False
        return update_head

    def handle_key(self, key_code, is_shift_down, is_alt_down, is_key_down):
        was_go_fast = self.go_fast
        was_go_slow = self.go_slow
        self.go_fast = is_shift_down
        self.go_slow = is_alt_down
        speed_changed = (was_go_fast != self.go_fast) or (was_go_slow != self.go_slow)
        update_driving = self.update_drive_state(key_code, is_key_down, speed_changed)
        update_lift = self.update_lift_state(key_code, is_key_down, speed_changed)
        update_head = self.update_head_state(key_code, is_key_down, speed_changed)
        if update_driving:
            self.update_driving()
        if update_head:
            self.update_head()
        if update_lift:
            self.update_lift()

    def pick_speed(self, fast_speed, mid_speed, slow_speed):
        if self.go_fast:
            if not self.go_slow:
                return fast_speed
        elif self.go_slow:
            return slow_speed
        return mid_speed

    def update_lift(self):
        lift_speed = self.pick_speed(8, 4, 2)
        lift_vel = (self.lift_up - self.lift_down) * lift_speed
        if self.last_lift and lift_vel == self.last_lift:
            return
        self.last_lift = lift_vel
        self.vector.motors.set_lift_motor(lift_vel)

    def update_head(self):
        head_speed = self.pick_speed(2, 1, 0.5)
        head_vel = (self.head_up - self.head_down) * head_speed
        if self.last_head and head_vel == self.last_head:
            return
        self.last_head = head_vel
        self.vector.motors.set_head_motor(head_vel)

    def update_driving(self):
        drive_dir = (self.drive_forwards - self.drive_back)
        turn_dir = (self.turn_right - self.turn_left)
        if drive_dir < 0:
            turn_dir = -turn_dir
        forward_speed = self.pick_speed(150, 75, 50)
        turn_speed = self.pick_speed(100, 50, 30)
        l_wheel_speed = (drive_dir * forward_speed) + (turn_speed * turn_dir)
        r_wheel_speed = (drive_dir * forward_speed) - (turn_speed * turn_dir)
        wheel_params = (l_wheel_speed, r_wheel_speed, l_wheel_speed * 4, r_wheel_speed * 4)
        if self.last_wheels and wheel_params == self.last_wheels:
            return
        self.last_wheels = wheel_params
        self.vector.motors.set_wheel_motors(*wheel_params)

def main(args):
    # Connect to robot
    robot_serial = utils.get_robot_serial(robot_index=args.robot_index)
    with anki_vector.Robot(serial=robot_serial, default_logging=False, behavior_control_level=anki_vector.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
        remote_control_vector = RemoteControlVector(robot)

        # Display available keyboard commands
        width, height = 320, 240
        window = pyglet.window.Window(width, height)
        lines = [
            'ws - drive forward/backward',
            'ad - turn left/right',
            'rf - lift up/down',
            'tg - head up/down'
        ]
        label = pyglet.text.Label('\n'.join(lines), x=(width // 2), y=(height // 2), width=width, anchor_x='center', anchor_y='center', align='center', multiline=True)
        keys = key.KeyStateHandler()
        window.push_handlers(keys)

        @window.event
        def on_draw():
            window.clear()
            label.draw()
            is_shift_down = keys[key.LSHIFT] or keys[key.RSHIFT]
            is_alt_down = (keys[key.LOPTION] or keys[key.ROPTION]) if sys.platform == 'darwin' else (keys[key.LALT] or keys[key.RALT])
            for k in ['W', 'S', 'A', 'D', 'R', 'F', 'T', 'G']:
                key_code = ord(k)
                key_constant = ord(k.lower())
                is_key_down = keys[key_constant]
                remote_control_vector.handle_key(key_code, is_shift_down, is_alt_down, is_key_down)

        pyglet.app.run()

parser = argparse.ArgumentParser()
parser.add_argument('--robot-index', type=int, default=0)
main(parser.parse_args())
