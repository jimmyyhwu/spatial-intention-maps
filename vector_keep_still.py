import time
from multiprocessing.dummy import Pool
import anki_vector
import vector_utils as utils

def reserve_control(args):
    robot_index, (robot_name, robot_serial) = args
    try:
        anki_vector.behavior.ReserveBehaviorControl(serial=robot_serial)._conn.connect()
        with anki_vector.Robot(serial=robot_serial, default_logging=False) as robot:
            robot.behavior.set_head_angle(anki_vector.util.degrees(0))
        print('Connected to {} ({})'.format(robot_name, robot_index))
    except anki_vector.exceptions.VectorNotFoundException:
        print('Could not find {} ({})'.format(robot_name, robot_index))
    except anki_vector.exceptions.VectorControlTimeoutException:
        print('Could not connect to {} ({})'.format(robot_name, robot_index))

robot_names = utils.get_robot_names()
robot_serials = utils.get_robot_serials()
with Pool(len(robot_names)) as p:
    p.map(reserve_control, enumerate(zip(robot_names, robot_serials)))
while True:
    time.sleep(1)
