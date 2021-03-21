# Adapted from https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_markers.cpp
import argparse
import cv2
import utils

def main(serial):
    # Read camera parameters
    image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(serial)

    # Set up webcam
    cap = utils.get_video_cap(serial, image_width, image_height)

    # Set up aruco dict
    aruco_dict = cv2.aruco.Dictionary_get(utils.get_marker_dict_id())

    while True:
        if cv2.waitKey(1) == 27:  # Esc key
            break

        image = None
        while image is None:
            _, image = cap.read()

        # Undistort image and detect markers
        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)

        # Show detections
        image_copy = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
        cv2.imshow('out', image_copy)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default=None)
    parser.add_argument('--camera2', action='store_true')  # Use second camera
    args = parser.parse_args()
    if args.serial is None:
        args.serial = 'E4298F4E' if args.camera2 else '099A11EE'
    main(args.serial)
