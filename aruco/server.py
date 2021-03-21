import argparse
import math
import time
from multiprocessing.connection import Listener
from queue import Queue
from threading import Thread
import cv2
import numpy as np
import utils

class Camera:
    def __init__(self, serial):
        image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(serial)
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (image_width, image_height), cv2.CV_32FC1)
        self._cap = utils.get_video_cap(serial, image_width, image_height)
        self._queue = Queue(maxsize=1)
        self._thread = Thread(target=self._worker)
        self._thread.start()

    def _worker(self):
        while True:
            if self._queue.empty():
                self._queue.put((time.time(), self._read()))
            time.sleep(0.001)

    def _read(self):
        image = None
        while image is None:
            _, image = self._cap.read()
        return cv2.remap(image, self._map_x, self._map_y, cv2.INTER_LINEAR)  # Undistort image

    def read(self):
        capture_time, image = self._queue.get()
        if time.time() - capture_time > 0.1:  # 100 ms
            self._queue.get()  # Flush camera buffer
            _, image = self._queue.get()
        return image

    def release(self):
        self._cap.release()

class Board:
    def __init__(self, name, length, width, orientation, corner_offset):
        self.name = name
        self.length = length
        self.width = width
        self.orientation = orientation

        # Aruco dicts
        dict_id = utils.get_marker_dict_id()
        self.board_dict = cv2.aruco.Dictionary_get(dict_id)
        self.board_dict.bytesList = self.board_dict.bytesList[utils.get_marker_ids('corners_{}'.format(self.name))]
        self.marker_dict = cv2.aruco.Dictionary_get(dict_id)
        self.marker_dict.bytesList = self.marker_dict.bytesList[utils.get_marker_ids(self.name)]

        # Warping
        self.pixels_per_mm = 2  # 2 gives much better marker detections than 1
        self.pixels_per_m = 1000 * self.pixels_per_mm
        self.length_pixels = int(self.length * self.pixels_per_m)
        self.width_pixels = int(self.width * self.pixels_per_m)
        self.corner_offset_pixels = int(corner_offset * self.pixels_per_m)
        self.transformation_matrix = None
        self.initialized = False

        # Orientation-specific adjustments
        marker_diag_len_pixels =  math.ceil(18 * math.sqrt(2) * self.pixels_per_mm)
        if self.orientation == 'left':
            self.pixel_xy_to_position = self.pixel_xy_to_position_left
            self.heading_offset = math.pi / 2
            self.padding_bottom_pixels = marker_diag_len_pixels  # Add padding to detect markers in between the left and right boards
        elif self.orientation == 'right':
            self.pixel_xy_to_position = self.pixel_xy_to_position_right
            self.heading_offset = -math.pi / 2
            self.padding_bottom_pixels = marker_diag_len_pixels
        else:
            self.pixel_xy_to_position = self.pixel_xy_to_position_default
            self.heading_offset = 0
            self.padding_bottom_pixels = 0

    def reset(self):
        self.transformation_matrix = None
        self.initialized = False

    def initialize(self, image, debug=False):
        # Detect board corner markers (assumes board won't move since this is only done once)
        corners, indices, _ = cv2.aruco.detectMarkers(image, self.board_dict)

        # Show detections
        if debug:
            image_copy = image.copy()
            if indices is not None:
                cv2.aruco.drawDetectedMarkers(image_copy, corners, indices)
            cv2.imshow('board_corners_{}'.format(self.name), image_copy)
            cv2.waitKey(1)

        # Board was not found
        if indices is None or len(indices) < 4:
            return

        # Compute perspective transform for warp
        src_points = np.array([corner.squeeze(0).mean(axis=0) for _, corner in sorted(zip(indices, corners))], dtype=np.float32)
        dst_points = np.array([
            [-self.corner_offset_pixels, -self.corner_offset_pixels],
            [self.length_pixels + self.corner_offset_pixels, -self.corner_offset_pixels],
            [self.length_pixels + self.corner_offset_pixels, self.width_pixels + self.corner_offset_pixels],
            [-self.corner_offset_pixels, self.width_pixels + self.corner_offset_pixels]
        ], dtype=np.float32)
        self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.initialized = True

    def get_marker_poses(self, image, debug_data=None, debug=False):
        # Warp the board
        image = cv2.warpPerspective(image, self.transformation_matrix, (self.length_pixels, self.width_pixels + self.padding_bottom_pixels))

        # Detect markers in warped image
        corners, indices, _ = cv2.aruco.detectMarkers(image, self.marker_dict)

        if debug:
            image_copy = image.copy()
            if indices is not None:
                # Show detections
                cv2.aruco.drawDetectedMarkers(image_copy, corners, indices)

                if self.name.startswith('robots') and debug_data is not None:
                    for waypoint_positions, target_end_effector_position, controller_debug_data in debug_data:
                        # Show waypoints
                        if waypoint_positions is not None:
                            waypoint_pixel_xys = list(map(self.position_to_pixel_xy, waypoint_positions))
                            for i in range(1, len(waypoint_pixel_xys)):
                                cv2.line(image_copy, waypoint_pixel_xys[i - 1], waypoint_pixel_xys[i], (255, 0, 0), 2)

                        # Show target end effector position
                        if target_end_effector_position is not None:
                            cv2.circle(image_copy, self.position_to_pixel_xy(target_end_effector_position), 10, (255, 0, 0), 2)

                        # Show pure pursuit
                        if controller_debug_data is not None:
                            lookahead_position, signed_radius, heading_diff, current_position, current_heading = controller_debug_data
                            cv2.circle(image_copy, self.position_to_pixel_xy(lookahead_position), 10, (0, 0, 255), 2)
                            if signed_radius is not None:
                                center = (current_position[0] - signed_radius * math.sin(current_heading), current_position[1] + signed_radius * math.cos(current_heading))
                                radius = int(self.pixels_per_m * abs(signed_radius))
                                angle = math.degrees(-current_heading) + math.copysign(1, signed_radius) * 90
                                end_angle = 2 * math.degrees(-heading_diff)
                                cv2.ellipse(image_copy, self.position_to_pixel_xy(center), (radius, radius), angle, 0, end_angle, (0, 0, 255), 2)

            image_copy = cv2.resize(image_copy, (int(image_copy.shape[1] / self.pixels_per_mm), int(image_copy.shape[0] / self.pixels_per_mm)))
            cv2.imshow('{}'.format(self.name), image_copy)
            cv2.waitKey(1)

        if indices is None:
            return {}

        # Compute poses
        data = {}
        for marker_index, corner in zip(indices, corners):
            marker_index = marker_index.item()
            marker_corners = corner.squeeze(0)
            marker_center = marker_corners.mean(axis=0)

            # Compute marker heading, making sure to deal with wraparound
            diffs = marker_corners - marker_center
            angles = np.arctan2(-diffs[:, 1], diffs[:, 0]) + np.radians([-135, -45, 45, 135]) + self.heading_offset
            angles = np.mod(angles + math.pi, 2 * math.pi) - math.pi
            angles_plus_pi = angles + math.pi
            angles_plus_pi = np.mod(angles_plus_pi + math.pi, 2 * math.pi) - math.pi
            angle_std = angles.std()
            angle_plus_pi_std = angles_plus_pi.std()
            if angle_plus_pi_std < angle_std:
                heading = angles_plus_pi.mean() - math.pi
                angle_std = angle_plus_pi_std
            else:
                heading = angles.mean()

            # Ignore markers on sides of cubes
            if self.name.startswith('cubes') and angle_std > math.radians(5):
                continue

            # Compute marker position
            position = self.pixel_xy_to_position(marker_center[0], marker_center[1])

            # Store pose
            marker_data = {'position': position, 'heading': heading}
            data[marker_index] = marker_data

        return data

    def pixel_xy_to_position_default(self, x, y):
        return (x / self.pixels_per_m - self.length / 2, self.width / 2 - y / self.pixels_per_m)

    def pixel_xy_to_position_left(self, x, y):
        return (y / self.pixels_per_m - self.width, x / self.pixels_per_m - self.length / 2)

    def pixel_xy_to_position_right(self, x, y):
        return (self.width - y / self.pixels_per_m, self.length / 2 - x / self.pixels_per_m)

    def position_to_pixel_xy(self, position):
        assert self.orientation is None  # This is only used for debug, so only the default orientation is implemented
        return int((position[0] + self.length / 2) * self.pixels_per_m), int(self.pixels_per_m * (self.width / 2 - position[1]))

class Server:
    def __init__(self, camera, board_length, board_width, board_orientation=None, debug=False):
        self.camera = camera
        self.debug = debug

        # Listener
        self.address = 'localhost'
        #self.address = '0.0.0.0'
        if board_orientation == 'left':
            self.port = 6001
        elif board_orientation == 'right':
            self.port = 6002
        else:
            self.port = 6000
        self.listener = Listener((self.address, self.port), authkey=b'secret password')

        # Boards
        suffix = '' if board_orientation is None else '_{}'.format(board_orientation)
        self.robot_board = Board('robots' + suffix, board_length, board_width, board_orientation, 0.036)
        self.cube_board = Board('cubes' + suffix, board_length, board_width, board_orientation, 0.012)

    def initialize_boards(self):
        boards = [self.robot_board, self.cube_board]
        for board in boards:
            board.reset()
        while not all(board.initialized for board in boards):
            image = self.camera.read()
            for board in boards:
                if not board.initialized:
                    board.initialize(image, debug=self.debug)

    def run(self):
        try:
            while True:
                # Connect to client
                print('Waiting for connection ({}:{})'.format(self.address, self.port))
                conn = self.listener.accept()
                print('Connected!')

                # Initialize boards
                self.initialize_boards()

                # Send pose estimates
                debug_data = None
                while True:
                    start_time = time.time()
                    try:
                        debug_data = conn.recv()  # Wait for request
                        image = self.camera.read()
                        robot_poses = self.robot_board.get_marker_poses(image, debug_data=debug_data, debug=self.debug)
                        cube_poses = self.cube_board.get_marker_poses(image, debug_data=debug_data, debug=self.debug)
                        conn.send((robot_poses, cube_poses))
                    except (ConnectionResetError, EOFError):
                        break
                    print('{:.1f} ms'.format(1000 * (time.time() - start_time)))
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

def main(args):
    if args.serial is None:
        args.serial = 'E4298F4E' if args.board_orientation == 'right' else '099A11EE'
    camera = Camera(args.serial)

    cv2.setNumThreads(4)  # Based on 12 CPUs
    server = Server(camera, args.board_length, args.board_width, args.board_orientation, debug=args.debug)
    server.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default=None)
    parser.add_argument('--board-length', type=float, default=1.0)
    parser.add_argument('--board-width', type=float, default=0.5)
    parser.add_argument('--board-orientation', default=None)  # For large environments: left or right
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
