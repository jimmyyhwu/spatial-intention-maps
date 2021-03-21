import argparse
from pprint import pprint
import time
from multiprocessing.connection import Client

parser = argparse.ArgumentParser()
parser.add_argument('--large', action='store_true')
parser.add_argument('--benchmark', action='store_true')
args = parser.parse_args()

address = 'localhost'
if args.large:
    conns = [Client((address, 6001), authkey=b'secret password'), Client((address, 6002), authkey=b'secret password')]
else:
    conns = [Client((address, 6000), authkey=b'secret password')]

while True:
    if args.benchmark:
        start_time = time.time()

    for conn in conns:
        conn.send(None)

    for conn in conns:
        data = conn.recv()
        if not args.benchmark:
            pprint(data)

    if args.benchmark:
        print('{:.1f} ms'.format(1000 * (time.time() - start_time)))
