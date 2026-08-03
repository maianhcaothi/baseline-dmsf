import sys
import os
import signal
import yaml

# The console summary format is pinned verbatim by the guide and contains em
# dashes. Piping this server's output to a file (`python server.py > run.log`)
# hands stdout the ANSI codepage, which cannot encode them — and an encode error
# inside the shutdown path would take the whole results pipeline with it.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

# Thêm src/ vào path để DmsfServer.py tìm được DmsfScheduler
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.DmsfServer import DmsfServer
from src.DmsfUtils import delete_old_queues

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

rabbit = config['rabbit']


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(rabbit['address'], rabbit.get('username', 'guest'),
                      rabbit.get('password', 'guest'), rabbit.get('virtual-host', '/'))
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    delete_old_queues(rabbit['address'], rabbit.get('username', 'guest'),
                      rabbit.get('password', 'guest'), rabbit.get('virtual-host', '/'))
    server = DmsfServer(config)
    server.start()
