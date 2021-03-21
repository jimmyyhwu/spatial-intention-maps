import sys
from pathlib import Path
import cv2

################################################################################
# Board and markers

def get_marker_dict_id():
    return cv2.aruco.DICT_4X4_50

def get_marker_ids(marker_type):
    if marker_type.startswith('robots'):
        return list(range(10))
    if marker_type.startswith('cubes'):
        return list(range(10, 34))
    if marker_type in ('corners_robots', 'corners_robots_right'):
        return list(range(42, 46))
    if marker_type in ('corners_cubes', 'corners_cubes_right'):
        return list(range(46, 50))
    if marker_type == 'corners_robots_left':
        return list(range(34, 38))
    if marker_type == 'corners_cubes_left':
        return list(range(38, 42))
    if marker_type == 'corners':
        marker_ids = []
        for mt in ['corners_robots', 'corners_cubes', 'corners_robots_left', 'corners_cubes_left']:
            marker_ids.extend(get_marker_ids(mt))
        return marker_ids
    raise Exception

def get_charuco_board_params():
    return {
        'squares_x': 10,
        'squares_y': 7,
        'square_length': 0.024,  # 24 mm
        'marker_length': 0.018,  # 18 mm
    }

def get_paper_params(orientation='P'):
    width, height, margin = 8.5, 11, 0.5
    if orientation == 'L':
        width, height = height, width
    ppi = 600
    mm_per_in = 25.4
    params = {}
    params['width_mm'] = mm_per_in * width
    params['height_mm'] = mm_per_in * height
    params['margin_mm'] = mm_per_in * margin
    params['mm_per_printed_pixel'] = mm_per_in / ppi
    return params

################################################################################
# Camera

def get_video_cap(serial, frame_width, frame_height):
    if sys.platform == 'darwin':
        return cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{}-video-index0'.format(serial))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Gives much better latency
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 30)
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == frame_width
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == frame_height
    assert cap.get(cv2.CAP_PROP_BUFFERSIZE) == 1
    assert cap.get(cv2.CAP_PROP_AUTOFOCUS) == 0
    assert cap.get(cv2.CAP_PROP_FOCUS) == 30
    return cap

def get_camera_params(serial):
    camera_params_file_path = Path('camera_params') / '{}.yml'.format(serial)
    assert camera_params_file_path.exists()
    fs = cv2.FileStorage(str(camera_params_file_path), cv2.FILE_STORAGE_READ)
    image_width = int(fs.getNode('image_width').real())
    image_height = int(fs.getNode('image_height').real())
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('distortion_coefficients').mat()
    fs.release()
    return image_width, image_height, camera_matrix, dist_coeffs
