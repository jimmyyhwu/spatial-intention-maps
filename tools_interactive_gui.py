import argparse
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.transforms as transforms
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QMutex

from envs import VectorEnv, orientation_to_heading
import utils


def image_to_pixmap(image):
    b = BytesIO()
    utils.enlarge_image(Image.fromarray(image), scale_factor=2).save(b, format='PNG')
    return QtGui.QPixmap.fromImage(QtGui.QImage.fromData(b.getvalue()))

class TransitionTracker:
    def __init__(self, initial_state):
        self.prev_state = initial_state
        self.prev_action = [[None for _ in g] for g in self.prev_state]

    def update_action(self, action):
        for i, g in enumerate(action):
            for j, a in enumerate(g):
                if a is not None:
                    self.prev_action[i][j] = a

    def update_step_completed(self, reward, state, done):
        transitions = []
        for i, g in enumerate(state):
            for j, s in enumerate(g):
                if s is not None or done:
                    if self.prev_state[i][j] is not None:
                        transition = (i, j), (self.prev_state[i][j], self.prev_action[i][j], reward[i][j])
                        transitions.append(transition)
                    self.prev_state[i][j] = s
        return transitions

class TransitionViewer(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle('Transition Viewer')
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.transition_images = [[QtWidgets.QLabel() for _ in g] for g in self.main_window.state]
        transition_vis_layout = QtWidgets.QHBoxLayout()
        for i, g in enumerate(self.main_window.state):
            for j, _ in enumerate(g):
                transition_vis_layout.addWidget(self.transition_images[i][j])
        main_layout.addLayout(transition_vis_layout)

        self.reset()

    def reset(self):
        for i, g in enumerate(self.main_window.state):
            for j, _ in enumerate(g):
                self.update(i, j, None, None, 0)

    def update(self, i, j, s, a, r):
        transition_vis = utils.to_uint8_image(utils.get_transition_visualization(s, a, r))
        self.transition_images[i][j].setPixmap(image_to_pixmap(transition_vis))

    def closeEvent(self, _):
        self.main_window.show_transition_viewer.setChecked(False)

class PolicyOutput(QtWidgets.QWidget):
    def __init__(self, main_window, policy):
        super().__init__()
        self.main_window = main_window
        self.policy = policy
        self.action = None
        self.q_value_map = None
        self.q_value_map_image = QtWidgets.QLabel()

        self.setWindowTitle('Policy Output')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.q_value_map_image)
        use_policy_action_button = QtWidgets.QPushButton('Use policy action')
        use_policy_action_button.clicked.connect(self.use_policy_action)
        layout.addWidget(use_policy_action_button)

    def use_policy_action(self):
        self.main_window.store_new_action(*self.action)

    def refresh(self):
        if not self.isVisible():
            return

        state = self.main_window.env.get_state()
        action, info = self.policy.step(state, debug=True)
        policy_output = info['output']
        for i, g in enumerate(state):
            for j, s in enumerate(g):
                if s is not None:
                    self.action = (i, j, action[i][j])
                    overhead_image = utils.get_overhead_image(s)
                    o = policy_output[i][j]
                    o = utils.to_uint8_image(utils.scale_min_max(o))
                    policy_output_channels = [utils.get_output_visualization(overhead_image, o_chan) for o_chan in o]
                    policy_output_vis = utils.to_uint8_image(np.concatenate(policy_output_channels, axis=0))
                    self.q_value_map = policy_output_vis
                    self.q_value_map_image.setPixmap(image_to_pixmap(policy_output_vis))

    def closeEvent(self, _):
        self.main_window.show_policy_output.setChecked(False)

class DraggablePolygon:
    lock = None
    selected = None

    def __init__(self, canvas, polygon):
        self.canvas = canvas
        self.polygon = polygon
        self.canvas.ax.add_patch(polygon)
        self.position = (0, 0)
        self.heading = 0
        self.click_data = None
        self.original_edge_color = self.polygon.get_ec()
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.update_transform()

    def on_click(self, event):
        if DraggablePolygon.lock is None:
            contains, _ = self.polygon.contains(event)
            if contains:
                DraggablePolygon.lock = self
                self.click_data = self.position, event.xdata, event.ydata

    def on_motion(self, event):
        if self.click_data is not None:
            position, x, y = self.click_data
            dx = event.xdata - x
            dy = event.ydata - y
            self.position = (position[0] + dx, position[1] + dy)
            self.update_transform()
            self.canvas.draw()

    def on_release(self, _):
        if self.click_data is not None:
            click_position, _, _ = self.click_data
            self.click_data = None
            DraggablePolygon.lock = None
            if self.position == click_position:
                self.toggle_selected()
            else:
                self.on_new_pose()

    def toggle_selected(self):
        if DraggablePolygon.selected is self:
            self.polygon.set_ec(self.original_edge_color)
            self.canvas.draw()
            self.on_new_pose()
            DraggablePolygon.selected = None
        elif DraggablePolygon.selected is None:
            self.polygon.set_ec('r')
            self.canvas.draw()
            DraggablePolygon.selected = self

    def on_scroll(self, event):
        if DraggablePolygon.selected is self:
            self.heading -= (0.005 if sys.platform == 'darwin' else 0.1) * event.step
            self.update_transform()
            self.canvas.draw()

    def update_transform(self):
        transform = transforms.Affine2D().rotate_deg(np.degrees(self.heading)).translate(*self.position)
        self.polygon.set_transform(transform + self.polygon.axes.transData)

    def set_pose(self, position, heading, hidden=False):
        self.position = position
        self.heading = heading
        self.polygon.set_visible(not hidden)
        self.update_transform()

    def on_new_pose(self):
        raise NotImplementedError

class DraggableRobotPolygon(DraggablePolygon):
    def __init__(self, canvas, robot):
        rect_xy = [
            [robot.BACKPACK_OFFSET, -robot.HALF_WIDTH],
            [robot.BASE_LENGTH + robot.BACKPACK_OFFSET, -robot.HALF_WIDTH],
            [robot.BASE_LENGTH + robot.BACKPACK_OFFSET, robot.HALF_WIDTH],
            [robot.BACKPACK_OFFSET, robot.HALF_WIDTH],
        ]
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 16)
        circle_xy = np.stack((robot.BACKPACK_OFFSET + robot.HALF_WIDTH * np.cos(theta), robot.HALF_WIDTH * np.sin(theta)), axis=1)
        robot_xy = rect_xy + circle_xy.tolist()
        polygon = Polygon(robot_xy, True, color=robot.COLOR[:3])
        super().__init__(canvas, polygon)
        self.env = self.canvas.main_window.env
        self.env_mutex = self.canvas.main_window.env_mutex
        self.robot = robot

    def on_new_pose(self):
        if self.env_mutex.tryLock():
            self.robot.reset()
            self.robot.reset_pose(self.position[0], self.position[1], self.heading)
            self.env.step_simulation()
            self.robot.update_map()
            self.canvas.main_window.refresh()
            self.env_mutex.unlock()

class DraggableCubePolygon(DraggablePolygon):
    def __init__(self, canvas, cube_id):
        cube_xy = ((VectorEnv.CUBE_WIDTH / 2) * np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])).tolist()
        polygon = Polygon(cube_xy, True, color=VectorEnv.CUBE_COLOR)
        super().__init__(canvas, polygon)
        self.env = self.canvas.main_window.env
        self.env_mutex = self.canvas.main_window.env_mutex
        self.cube_id = cube_id

    def on_new_pose(self):
        if self.env_mutex.tryLock():
            self.env.reset_cube_pose(self.cube_id, self.position[0], self.position[1], self.heading)
            self.env.step_simulation()
            self.canvas.main_window.refresh()
            self.env_mutex.unlock()

class VectorEnvCanvas(FigureCanvas):
    def __init__(self, main_window):
        self.main_window = main_window
        self.env = self.main_window.env
        ratio = self.env.room_width / self.env.room_length
        super().__init__(Figure(figsize=(4, ratio * 4)))
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.ax.axis('off')
        self.ax.axis((-self.env.room_length / 2, self.env.room_length / 2, -self.env.room_width / 2, self.env.room_width / 2))

        self.robot_polygons = []
        for robot in self.env.robots:
            self.robot_polygons.append(DraggableRobotPolygon(self, robot))
        self.cube_polygons = []
        for cube_id in self.env.cube_ids:
            self.cube_polygons.append(DraggableCubePolygon(self, cube_id))

    def refresh(self):
        for robot, polygon in zip(self.env.robots, self.robot_polygons):
            polygon.set_pose(robot.get_position()[:2], robot.get_heading())
        for cube_id, polygon in zip(self.env.cube_ids, self.cube_polygons):
            cube_position, cube_orientation = self.env.get_cube_pose(cube_id)
            hidden = (cube_position[2] < 0.9 * VectorEnv.REMOVED_BODY_Z)
            polygon.set_pose(cube_position[:2], orientation_to_heading(cube_orientation), hidden=hidden)
        self.draw()

class PoseEditor(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.canvas = VectorEnvCanvas(self.main_window)

        self.setWindowTitle('Pose Editor')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.canvas)
        label = QtWidgets.QLabel('Drag and drop to translate. Click and scroll to rotate.')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    def refresh(self):
        self.canvas.refresh()

    def closeEvent(self, _):
        self.main_window.show_pose_editor.setChecked(False)

class StateImage(QtWidgets.QLabel):
    def __init__(self, main_window, i, j):
        super().__init__()
        self.main_window = main_window
        self.i = i
        self.j = j
        self.state_width = VectorEnv.get_state_width()
        self.robot_group_types = self.main_window.env.get_robot_group_types()

    def mousePressEvent(self, event):
        x = (event.pos().x() - self.contentsMargins().left()) / self.pixmap().width()
        y = (event.pos().y() - self.contentsMargins().top()) / self.pixmap().height()
        if x > 1 or y > 1:
            return

        num_output_channels = VectorEnv.get_num_output_channels(self.robot_group_types[self.i])
        a = round(y * num_output_channels * self.state_width) * self.state_width + round(x * self.state_width)
        if event.button() == Qt.RightButton and a < self.state_width * self.state_width:
            a += self.state_width * self.state_width

        self.main_window.store_new_action(self.i, self.j, a)

class MainWindow(QtWidgets.QWidget):
    def __init__(self, env, policy):
        super().__init__()
        self.env = env
        self.policy = policy
        self.state = self.env.reset()
        self.transition_tracker = TransitionTracker(self.state)
        self.env_mutex = QMutex()

        self.setWindowTitle('Action Selector')
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # State visualization
        self.state_images = [[StateImage(self, i, j) for j, _ in enumerate(g)] for i, g in enumerate(self.state)]
        self.state_labels = [[QtWidgets.QLabel() for _ in g] for g in self.state]
        state_vis_layout = QtWidgets.QHBoxLayout()
        for i, g in enumerate(self.state):
            for j, _ in enumerate(g):
                state_and_label_layout = QtWidgets.QVBoxLayout()
                state_and_label_layout.addWidget(self.state_images[i][j])
                state_and_label_layout.addStretch(1)
                state_and_label_layout.addWidget(self.state_labels[i][j])
                state_vis_layout.addLayout(state_and_label_layout)
                self.state_labels[i][j].setAlignment(Qt.AlignCenter)
        main_layout.addLayout(state_vis_layout)

        # Other windows
        options_box_layout = QtWidgets.QVBoxLayout()
        self.transition_viewer_window = TransitionViewer(self)
        self.show_transition_viewer = QtWidgets.QCheckBox('Show transition viewer')
        self.show_transition_viewer.stateChanged.connect(self.show_transition_viewer_window)
        options_box_layout.addWidget(self.show_transition_viewer)
        if self.policy is not None:
            self.policy_output_window = PolicyOutput(self, policy)
            self.show_policy_output = QtWidgets.QCheckBox('Show policy output')
            self.show_policy_output.stateChanged.connect(self.show_policy_output_window)
            options_box_layout.addWidget(self.show_policy_output)
        self.pose_editor_window = PoseEditor(self)
        self.show_pose_editor = QtWidgets.QCheckBox('Show pose editor')
        self.show_pose_editor.stateChanged.connect(self.show_pose_editor_window)
        options_box_layout.addWidget(self.show_pose_editor)
        options_box = QtWidgets.QGroupBox()
        options_box.setLayout(options_box_layout)
        main_layout.addWidget(options_box)

        # Buttons
        reset_env_button = QtWidgets.QPushButton('Reset env')
        reset_env_button.clicked.connect(self.reset_env)
        main_layout.addWidget(reset_env_button)
        save_figures_button = QtWidgets.QPushButton('Save figures')
        save_figures_button.clicked.connect(self.save_figures)
        main_layout.addWidget(save_figures_button)
        if self.env.real:
            force_update_poses_button = QtWidgets.QPushButton('Update poses')
            force_update_poses_button.clicked.connect(self.force_update_poses)
            main_layout.addWidget(force_update_poses_button)

        self.refresh()

    def show_transition_viewer_window(self):
        if self.show_transition_viewer.isChecked():
            self.transition_viewer_window.show()
        else:
            self.transition_viewer_window.hide()

    def show_policy_output_window(self):
        if self.show_policy_output.isChecked():
            self.policy_output_window.show()
            if self.env_mutex.tryLock():
                self.policy_output_window.refresh()
                self.env_mutex.unlock()
        else:
            self.policy_output_window.hide()

    def show_pose_editor_window(self):
        if self.show_pose_editor.isChecked():
            self.pose_editor_window.show()
            if self.env_mutex.tryLock():
                self.pose_editor_window.refresh()
                self.env_mutex.unlock()
        else:
            self.pose_editor_window.hide()

    def reset_env(self):
        if self.env_mutex.tryLock():
            self.state = self.env.reset()
            self.transition_tracker = TransitionTracker(self.state)
            self.refresh()
            self.env_mutex.unlock()
        self.transition_viewer_window.reset()

    def save_figures(self):
        if self.env_mutex.tryLock():
            self.env.get_state(save_figures=True)
            if self.policy is not None and self.policy_output_window.q_value_map is not None:
                for robot in self.env.robots:
                    if robot.awaiting_new_action:
                        output_dir = Path('figures') / 'robot_id_{}'.format(robot.id)
                        utils.enlarge_image(Image.fromarray(self.policy_output_window.q_value_map)).save(output_dir / 'q-value-map.png')
            self.env_mutex.unlock()

    def force_update_poses(self):
        if self.env_mutex.tryLock():
            for robot in self.env.robots:
                robot.reset()
            self.env.update_poses()
            for robot in self.env.robots:
                if robot.awaiting_new_action:
                    robot.update_map()
            self.refresh()
            self.env_mutex.unlock()

    def store_new_action(self, i, j, a):
        if self.env_mutex.tryLock():
            action = [[None for _ in g] for g in self.state]
            action[i][j] = a
            if self.env.robot_groups[i][j].awaiting_new_action:
                self.transition_tracker.update_action(action)
                self.state, reward, done, _ = self.env.step(action)
                for idx, transition in self.transition_tracker.update_step_completed(reward, self.state, done):
                    self.transition_viewer_window.update(*idx, *transition)
            else:
                self.env.store_new_action(action)
            self.refresh()
            self.env_mutex.unlock()

    def refresh(self):
        for i, g in enumerate(self.env.get_state(all_robots=True)):
            for j, s in enumerate(g):

                # Update state visualization
                state_vis = utils.get_state_visualization(s)
                num_output_channels = VectorEnv.get_num_output_channels(self.env.robot_group_types[i])
                state_vis = utils.to_uint8_image(np.concatenate(num_output_channels * [state_vis], axis=0))
                self.state_images[i][j].setPixmap(image_to_pixmap(state_vis))

                # Update state label
                if not self.env.robot_groups[i][j].is_idle():
                    text = 'Executing action'
                elif self.env.robot_groups[i][j].awaiting_new_action:
                    text = 'Awaiting new action'
                else:
                    text = 'Idle'
                self.state_labels[i][j].setText(text)

                # Draw red border if robot is awaiting new action
                if len(self.env.robots) > 1 and self.env.robot_groups[i][j].awaiting_new_action:
                    self.state_images[i][j].setStyleSheet('border: 5px solid red;')
                else:
                    self.state_images[i][j].setStyleSheet('border: 5px solid transparent;')

        self.pose_editor_window.refresh()
        if self.policy is not None:
            self.policy_output_window.refresh()

    def closeEvent(self, _):
        self.transition_viewer_window.close()
        if self.policy is not None:
            self.policy_output_window.close()
        self.pose_editor_window.close()

def main(args):
    kwargs = {}

    # Visualization
    kwargs['show_gui'] = True
    #kwargs['show_debug_annotations'] = True
    #kwargs['show_occupancy_maps'] = True

    # Real robot
    #kwargs['real'] = True
    #kwargs['real_robot_indices'] = [0, 1, 2, 3]
    #kwargs['real_cube_indices'] = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]
    #kwargs['real_debug'] = True

    # Set up env and policy
    if args.config_path is not None:
        cfg = utils.load_config(args.config_path)
        print(args.config_path)
        env = utils.get_env_from_cfg(cfg, **kwargs)
        policy = utils.get_policy_from_cfg(cfg, env.get_robot_group_types())

    else:
        # Env name
        env_name = 'small_empty'
        #env_name = 'small_divider'
        #env_name = 'large_empty'
        #env_name = 'large_doors'
        #env_name = 'large_tunnels'
        #env_name = 'large_rooms'

        # Robot config
        robot_config = [{'lifting_robot': 1}]
        #robot_config = [{'pushing_robot': 1}]
        #robot_config = [{'throwing_robot': 1}]
        #robot_config = [{'rescue_robot': 1}]
        #robot_config = [{'lifting_robot': 4}]
        #robot_config = [{'pushing_robot': 4}]
        #robot_config = [{'rescue_robot': 4}]
        #robot_config = [{'lifting_robot': 2}, {'pushing_robot': 2}]
        #robot_config = [{'lifting_robot': 2}, {'throwing_robot': 2}]

        # Room config
        kwargs['robot_config'] = robot_config
        kwargs['env_name'] = env_name
        utils.apply_misc_env_modifications(kwargs, env_name)

        # Misc
        kwargs['use_intention_map'] = True

        env = VectorEnv(**kwargs)
        policy = None

    # Run the GUI
    app = QtWidgets.QApplication([])
    main_window = MainWindow(env, policy)
    main_window.show()
    app.exec()

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    main(parser.parse_args())
