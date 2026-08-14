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


server = None
_interrupted = False


def signal_handler(sig, frame):
    """Ctrl+C ends the run; it does not throw the run away.

    The previous handler called sys.exit(0) straight from here, so
    DmsfServer.start() never reached its shutdown writers: fps_cluster.log,
    utilization.log, utilization_cluster.log and latency_cluster.log stayed at
    zero bytes and no archive directory was written at all. Worse, it also ran
    delete_old_queues() on the way out, which deletes utilization_queue -- the
    queue holding the device reports that had not been drained yet. A run
    stopped by hand lost its results twice over, silently.

    So: ask the server to stop consuming, and let start() finish normally
    through its finalizer. Queue cleanup belongs after that, not here. A second
    Ctrl+C is the escape hatch for an operator who does not want to wait.
    """
    global _interrupted
    if _interrupted:
        print("\n[Server] Second Ctrl+C - exiting now, results discarded.")
        os._exit(1)
    _interrupted = True
    if server is None:
        # Interrupted before the server exists: nothing has been measured yet.
        sys.exit(0)
    print("\nCatch stop signal Ctrl+C. Finishing the run and writing results ...")
    print("[Server] (press Ctrl+C again to abandon them)")
    server.request_stop("interrupted by Ctrl+C")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    delete_old_queues(rabbit['address'], rabbit.get('username', 'guest'),
                      rabbit.get('password', 'guest'), rabbit.get('virtual-host', '/'))
    server = DmsfServer(config)
    server.start()
