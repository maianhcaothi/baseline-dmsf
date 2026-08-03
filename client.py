import argparse
import json
import os
import pickle
import sys
import time
import uuid

import numpy as np
import pika
import torch
import yaml

# See server.py: redirected stdout gets the ANSI codepage, which cannot encode
# the em dashes in the pinned summary format. Telemetry must not kill the run.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

# Thêm src/ vào path để các module trong src/ tìm được nhau
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.DmsfScheduler import DmsfScheduler
from src.DmsfRpcClient import DmsfRpcClient

parser = argparse.ArgumentParser(description='DMSF Baseline Client')
parser.add_argument('--layer_id', type=int, required=True, help='1=edge, 2=cloud')
parser.add_argument('--device',   type=str, default=None)
parser.add_argument('--name',     type=str, default=None,
                    help='Tên thiết bị — dùng để tra profile trong unified profile format')
args = parser.parse_args()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

rabbit   = config['rabbit']
dmsf_cfg = config.get('dmsf', config.get('server', {}))
client_id = uuid.uuid4()

device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Client] layer_id={args.layer_id}  device={device}  id={str(client_id)[:8]}")

credentials = pika.PlainCredentials(rabbit.get('username', 'guest'),
                                    rabbit.get('password', 'guest'))
connection = pika.BlockingConnection(pika.ConnectionParameters(
    host=rabbit['address'],
    port=rabbit.get('port', 5672),
    virtual_host=rabbit.get('virtual-host', '/'),
    credentials=credentials,
    heartbeat=0,
    blocked_connection_timeout=300,
))
channel = connection.channel()


def measure_bandwidth(ch, client_id_str, payload_size_mb=1.0, runs=3):
    reply_queue = f"reply_{client_id_str}"
    ch.queue_declare(reply_queue, durable=False)
    payload = os.urandom(int(payload_size_mb * 1024 * 1024))
    samples = []
    for _ in range(runs):
        msg = pickle.dumps({'action': 'BW_TEST', 'client_id': client_id_str, 'payload': payload})
        t0 = time.perf_counter()
        ch.basic_publish(exchange='', routing_key='rpc_queue', body=msg)
        while True:
            _, _, body = ch.basic_get(queue=reply_queue, auto_ack=True)
            if body:
                break
            if time.perf_counter() - t0 > 30.0:
                raise TimeoutError("Bandwidth measurement timed out (server not responding)")
            time.sleep(0.005)
        samples.append(payload_size_mb / max(time.perf_counter() - t0, 1e-6))
    bw = float(np.median(samples))
    print(f"[Bandwidth] {bw:.1f} MB/s  (samples: {[f'{s:.1f}' for s in samples]} MB/s, "
          f"payload={payload_size_mb} MB x{runs})")
    return bw


# ── Load profile ──────────────────────────────────────────────────────────────
edge_times_ms, cloud_times_ms = None, None
sp_cfg = str(dmsf_cfg.get('split-point', 'auto'))

if sp_cfg == 'auto':
    profile_file = dmsf_cfg.get('profile-cache', '')
    if profile_file and os.path.exists(profile_file):
        with open(profile_file) as pf:
            prof = json.load(pf)

        if 'devices' in prof:
            # Unified format: {"devices": {"machine-2": {"edge_times_ms": ..., ...}}}
            name = args.name
            if not name:
                print("[Client] WARNING: unified profile requires --name. Using first entry.")
                name = next(iter(prof['devices']))
            device_prof = prof['devices'].get(name)
            if device_prof is None:
                raise KeyError(f"Device '{name}' not found in profile. "
                               f"Available: {list(prof['devices'].keys())}")
            if args.layer_id == 1:
                edge_times_ms = {int(k): v for k, v in device_prof['edge_times_ms'].items()}
                print(f"[Client] Loaded edge profile: {name} from {profile_file}")
            else:
                cloud_times_ms = {int(k): v for k, v in device_prof['cloud_times_ms'].items()}
                print(f"[Client] Loaded cloud profile: {name} from {profile_file}")
        else:
            # Legacy format: {"edge_times_ms": ..., "cloud_times_ms": ...}
            if args.layer_id == 1:
                edge_times_ms = {int(k): v for k, v in prof['edge_times_ms'].items()}
                print(f"[Client] Loaded edge profile from {profile_file}")
            else:
                cloud_times_ms = {int(k): v for k, v in prof['cloud_times_ms'].items()}
                print(f"[Client] Loaded cloud profile from {profile_file}")
    else:
        # Live profiling (chậm trên CPU — nên dùng profile-cache)
        from models.dmsf_26n import DMSF26n
        from utils.split_selector import LayerProfiler
        nc      = dmsf_cfg.get('nc', 10)
        weights = dmsf_cfg.get('weights', 'best_dmsf26n.pt')
        imgsz   = dmsf_cfg.get('imgsz', 640)
        batch   = config['server']['batch-size']
        _dev    = torch.device(device)
        _model  = DMSF26n(nc=nc).to(_dev)
        _ckpt   = torch.load(weights, map_location=_dev, weights_only=False)
        _state  = _ckpt.get('model', _ckpt)
        _model.load_state_dict(
            _state if isinstance(_state, dict) else _state.state_dict(), strict=False)
        _model.eval()
        print(f"[Client] Live profiling on {device} ...")
        _profiler = LayerProfiler(_model, _dev)
        _et, _ct  = _profiler.profile(img_size=imgsz, batch_size=batch)
        edge_times_ms  = _et
        cloud_times_ms = _ct
        del _model, _ckpt

# ── Đo bandwidth (chỉ edge, khi split-point auto) ─────────────────────────────
bandwidth_mb_s = None
if (args.layer_id == 1
        and dmsf_cfg.get('measure_bandwidth', True)
        and sp_cfg == 'auto'):
    try:
        bandwidth_mb_s = measure_bandwidth(channel, str(client_id))
    except Exception as e:
        print(f"[Bandwidth] Warning: {e}")
        if not channel.is_open:
            channel = connection.channel()

if __name__ == '__main__':
    data = {
        'action':          'REGISTER',
        'client_id':       client_id,
        'layer_id':        args.layer_id,
        'message':         'Hello from DMSF client',
        'client_name':     args.name,
        'device':          device,
        'edge_times_ms':   edge_times_ms,
        'cloud_times_ms':  cloud_times_ms,
        'bandwidth_mb_s':  bandwidth_mb_s,
    }
    scheduler = DmsfScheduler(client_id, args.layer_id, channel, device, config, name=args.name)
    client    = DmsfRpcClient(client_id, args.layer_id, channel,
                              scheduler.inference_func, device)
    client.send_to_server(data)
    client.wait_response()
