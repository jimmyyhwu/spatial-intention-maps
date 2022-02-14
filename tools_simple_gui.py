import os
import cv2
import numpy as np

from envs import VectorEnv
import utils


# Fix for macOS Big Sur
os.environ['QT_MAC_WANTS_LAYER'] = '1'

class SimpleGui:
    def __init__(self, env):
        self.env = env
        self.window_name = 'window'
        self.reward_image_height = 12
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self.state_width = VectorEnv.get_state_width()
        self.selected_action = None

    def run(self):
        state = self.env.reset()
        last_reward = 0
        done = False
        force_reset_env = False
        while True:
            # Display state and last reward
            self._update_display(state[0][0], last_reward)

            # Read keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                force_reset_env = True
            elif key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Execute selected action
            if self.selected_action is not None:
                action = [[(self.state_width * self.state_width) * self.selected_action[0] + self.state_width * self.selected_action[1] + self.selected_action[2]]]
                state, reward, done, _ = self.env.step(action)
                last_reward = reward[0][0]
                self.selected_action = None
            else:
                #import pybullet as p; p.stepSimulation(); import time; time.sleep(1.0 / 120)  # Uncomment to make pybullet window interactive on macOS
                pass

            # Reset environment
            if done or force_reset_env:
                state = self.env.reset()
                done = False
                force_reset_env = False
                last_reward = 0

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            # All robots - left click to just move to selected location
            self.selected_action = (0, max(0, y - self.reward_image_height), x)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Lifting, throwing, and rescue robots - right click to move to selected location and then execute end effector action
            self.selected_action = (1, max(0, y - self.reward_image_height), x)

    def _get_reward_image(self, reward):
        reward_image = np.zeros((self.reward_image_height, self.state_width, 3), dtype=np.float32)
        text = '{:+.02f}'.format(reward)
        cv2.putText(reward_image, text, (self.state_width - 5 * len(text), 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (1, 1, 1))
        return reward_image

    def _update_display(self, state, last_reward):
        reward_image = self._get_reward_image(last_reward)
        state_image = utils.get_state_visualization(state)
        cv2.imshow(self.window_name, np.concatenate((reward_image, state_image), axis=0)[:, :, ::-1])

def main():
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
    assert sum(sum(g.values()) for g in robot_config) == 1, 'This GUI only supports single-agent'

    # Room config
    kwargs = {}
    kwargs['robot_config'] = robot_config
    kwargs['env_name'] = env_name
    utils.apply_misc_env_modifications(kwargs, env_name)

    # Visualization
    kwargs['show_gui'] = True
    kwargs['show_debug_annotations'] = True
    #kwargs['show_occupancy_maps'] = True

    # Real robot
    #kwargs['real'] = True
    #kwargs['real_robot_indices'] = [0]
    #kwargs['real_cube_indices'] = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]
    #kwargs['real_debug'] = True

    env = VectorEnv(**kwargs)
    SimpleGui(env).run()
    env.close()

main()
