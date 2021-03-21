import configparser
from pathlib import Path

def get_config_path():
    return Path.home() / '.anki_vector/sdk_config.ini'

def get_config():
    config = configparser.ConfigParser()
    config.read(get_config_path())
    return config

def write_config(config):
    with open(get_config_path(), 'w') as f:
        config.write(f)

def get_robot_names():
    config = get_config()
    return [config[serial]['name'] for serial in config.sections()]

def get_robot_serials():
    return get_config().sections()

def get_robot_name(robot_index):
    return get_robot_names()[robot_index]

def get_robot_serial(robot_index):
    return get_robot_serials()[robot_index]
