import subprocess
from multiprocessing.dummy import Pool
import anki_vector
import vector_utils as utils

config = utils.get_config()
robot_names = utils.get_robot_names()
robot_serials = utils.get_robot_serials()

with Pool(len(robot_names)) as p:
    mdns_results = p.map(anki_vector.mdns.VectorMdns.find_vector, robot_names)
for name, serial, result in zip(robot_names, robot_serials, mdns_results):
    if result is None:
        print('{} was not found'.format(name))
    else:
        hostname = result['name'].lower()[:-1]
        ip = result['ipv4']
        subprocess.run(['ping', '-c', '1', ip], stdout=subprocess.DEVNULL, check=False)
        arp_output = str(subprocess.run(['arp', '-an'], stdout=subprocess.PIPE, check=False).stdout)
        mac = arp_output[arp_output.find(ip):].split(' ')[2]
        print('Hostname:    {}'.format(hostname))
        print('IP address:  {}'.format(ip))
        print('MAC address: {}'.format(mac))

        # Update IP address in config file
        config[serial]['ip'] = ip
    print()

utils.write_config(config)
