import csv
import os
import pickle
import time

import cv2
import psutil
import torch
from tqdm import tqdm

from utils.metrics import non_max_suppression


INTERMEDIATE_QUEUE = 'dmsf_intermediate_queue'
METRICS_EXCHANGE   = 'dmsf_metrics_fanout'


class DmsfScheduler:
    def __init__(self, client_id, layer_id, channel, device, config, name=None):
        self.client_id = client_id
        self.layer_id  = layer_id
        self.channel   = channel
        self.device    = device
        self.config    = config
        self._my_metrics_queue = None

        self._id_tag = name if name else str(client_id).replace('-', '')[:12]
        self._timing_log_edge  = f"timing_edge_{self._id_tag}.log"
        self._timing_log_cloud = f"timing_cloud_{self._id_tag}.log"
        for tlog in [self._timing_log_edge, self._timing_log_cloud]:
            if os.path.exists(tlog):
                try:
                    os.remove(tlog)
                except Exception:
                    pass

        self.channel.queue_declare(queue=INTERMEDIATE_QUEUE, durable=False)

    # ---------------------------------------------------------------------- #
    # Metrics
    # ---------------------------------------------------------------------- #
    def _write_metrics(self, log_path, split_point, role, batch_id, batch_size,
                       latency_ms, fps, ram_mb, msg_bytes, e2e_ms, edge_start_time):
        path = os.path.join(log_path, f"metrics_raw_{self._id_tag}.csv")
        new_file = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['mode', 'role', 'best_cut', 'batch_id', 'batch_size',
                            'latency_ms', 'fps', 'ram_mb', 'message_size_bytes',
                            'e2e_latency_ms', 'edge_start_time'])
            w.writerow(['split', role, split_point, batch_id, batch_size,
                        round(latency_ms, 3), round(fps, 3), round(ram_mb, 3),
                        msg_bytes, round(e2e_ms, 3),
                        edge_start_time if edge_start_time else ''])

    # ---------------------------------------------------------------------- #
    # Edge: layer_id == 1
    # ---------------------------------------------------------------------- #
    def first_layer(self, model, split_point, data, batch_size, log_path):
        model.eval()
        cap = cv2.VideoCapture(data)
        if not cap.isOpened():
            print(f"[Edge] Cannot open video: {data}")
            return

        proc = psutil.Process()
        pbar = tqdm(desc='[Edge] Processing', unit='frame')
        buf, batch_id, prev_end = [], 0, None
        imgsz = (self.config.get('inference') or {}).get('imgsz') \
                or self.config.get('dmsf', {}).get('imgsz', 640)

        with open(self._timing_log_edge, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (imgsz, imgsz))
            t = torch.from_numpy(frame[:, :, ::-1].copy()).float() / 255.0
            buf.append(t.permute(2, 0, 1))

            if len(buf) < batch_size:
                continue

            with open(self._timing_log_edge, "a") as _tf:
                print(str(time.time_ns()) + " get input", file=_tf)

            imgs = torch.stack(buf).to(self.device)
            buf  = []

            edge_start_wall = time.time()
            t0 = time.perf_counter()

            payload = model.forward_edge(imgs, split_point)
            payload['edge_start_time'] = edge_start_wall
            msg = pickle.dumps({'action': 'FEATURES', 'data': payload})
            msg_bytes = len(msg)

            self.channel.basic_publish(exchange='', routing_key=INTERMEDIATE_QUEUE, body=msg)

            with open(self._timing_log_edge, "a") as _tf:
                print(str(time.time_ns()) + " output", file=_tf)

            latency_ms = (time.perf_counter() - t0) * 1000
            fps_val    = batch_size / (time.perf_counter() - prev_end) if prev_end else 0.0
            ram_mb     = proc.memory_info().rss / 1e6

            self._write_metrics(log_path, split_point, 'edge', batch_id, batch_size,
                                latency_ms, fps_val, ram_mb, msg_bytes, 0.0, edge_start_wall)

            batch_id += 1
            prev_end  = time.perf_counter()
            pbar.update(batch_size)

        # Flush remaining
        if buf:
            imgs    = torch.stack(buf).to(self.device)
            payload = model.forward_edge(imgs, split_point)
            payload['edge_start_time'] = time.time()
            msg = pickle.dumps({'action': 'FEATURES', 'data': payload})
            self.channel.basic_publish(exchange='', routing_key=INTERMEDIATE_QUEUE, body=msg)

        with open(self._timing_log_edge, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)

        cap.release()
        pbar.close()
        print(f"[Edge] Done. {batch_id} batches sent.")

        # Broadcast edge metrics CSV to all cloud clients via fanout exchange
        metrics_file = os.path.join(log_path, f"metrics_raw_{self._id_tag}.csv")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'rb') as f:
                    metrics_data = f.read()
                self.channel.exchange_declare(exchange=METRICS_EXCHANGE, exchange_type='fanout', durable=False)
                self.channel.basic_publish(
                    exchange=METRICS_EXCHANGE, routing_key='',
                    body=pickle.dumps({"action": "METRICS",
                                       "filename": os.path.basename(metrics_file),
                                       "data": metrics_data}))
                print(f"[Edge] Broadcast metrics via fanout ({len(metrics_data)} bytes)")
            except Exception as e:
                print(f"[Edge] Warning: could not broadcast metrics: {e}")

        # Notify server
        self.channel.basic_publish(
            exchange='', routing_key='rpc_queue',
            body=pickle.dumps({'action': 'NOTIFY',
                               'client_id': self.client_id,
                               'layer_id': self.layer_id}))

        # Wait for STOP
        reply_q = f"reply_{self.client_id}"
        while True:
            _, _, body = self.channel.basic_get(queue=reply_q, auto_ack=True)
            if body:
                msg = pickle.loads(body)
                if msg.get('action') == 'STOP':
                    print("[Edge] Finish!")
                    break
            time.sleep(0.5)

    # ---------------------------------------------------------------------- #
    # Cloud: layer_id == 2
    # ---------------------------------------------------------------------- #
    def _setup_metrics_fanout_queue(self):
        my_queue = f"dmsf_mfq_{str(self.client_id).replace('-', '')}"
        try:
            self.channel.exchange_declare(exchange=METRICS_EXCHANGE, exchange_type='fanout', durable=False)
            self.channel.queue_declare(my_queue, durable=False)
            self.channel.queue_bind(queue=my_queue, exchange=METRICS_EXCHANGE)
            self._my_metrics_queue = my_queue
        except Exception as e:
            print(f"[Cloud] Metrics fanout setup failed: {e}")
            self._my_metrics_queue = None

    def last_layer(self, model, split_point, batch_size, conf, iou, log_path):
        self._setup_metrics_fanout_queue()
        model.eval()
        self.channel.basic_qos(prefetch_count=10)
        proc = psutil.Process()
        pbar = tqdm(desc='[Cloud] Processing', unit='frame')
        batch_id, prev_end = 0, None

        with open(self._timing_log_cloud, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)

        while True:
            method, _, body = self.channel.basic_get(
                queue=INTERMEDIATE_QUEUE, auto_ack=True)

            if method and body:
                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " get input", file=_tf)

                recv_size = len(body)
                msg = pickle.loads(body)
                payload = msg['data']
                edge_start_time = payload.pop('edge_start_time', time.time())

                t0 = time.perf_counter()
                out  = model.forward_cloud(
                    {'x_bit': payload['x_bit'], 'mu': payload['mu'],
                     'sigma': payload['sigma'], 'split': payload['split']},
                    device=str(self.device)
                )
                pred = out[0] if isinstance(out, tuple) else out
                non_max_suppression(pred, conf, iou)

                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " output", file=_tf)

                latency_ms  = (time.perf_counter() - t0) * 1000
                cloud_end   = time.time()
                e2e_ms      = (cloud_end - edge_start_time) * 1000
                fps_val     = batch_size / (time.perf_counter() - prev_end) if prev_end else 0.0
                ram_mb      = proc.memory_info().rss / 1e6
                bs          = pred.shape[0]

                self._write_metrics(log_path, split_point, 'cloud', batch_id, bs,
                                    latency_ms, fps_val, ram_mb, recv_size,
                                    e2e_ms, edge_start_time)

                batch_id += 1
                prev_end  = time.perf_counter()
                pbar.update(bs)

            else:
                method2, _, body2 = self.channel.basic_get(
                    queue=f"reply_{self.client_id}", auto_ack=True)
                if body2:
                    reply_msg = pickle.loads(body2)
                    if reply_msg.get('action') == 'STOP':
                        print(f"[Cloud] Received STOP. {batch_id} batches processed.")
                        break
                time.sleep(0.1)

        with open(self._timing_log_cloud, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)

        pbar.close()
        self._pivot_and_save(log_path)

    # ---------------------------------------------------------------------- #
    # Pivot summary (mirrors split_inference Scheduler._pivot_and_save)
    # ---------------------------------------------------------------------- #
    def _pivot_and_save(self, log_path):
        import glob as _glob

        lock_path = os.path.join(log_path, 'metrics_pivot.lock')
        out_path  = os.path.join(log_path, 'metrics_pivoted_dmsf.csv')

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return

        time.sleep(2.0)

        # Collect edge metrics CSV sent via fanout exchange
        if self._my_metrics_queue:
            try:
                while True:
                    method, _, body = self.channel.basic_get(queue=self._my_metrics_queue, auto_ack=True)
                    if not method:
                        break
                    msg = pickle.loads(body)
                    if msg.get("action") == "METRICS":
                        fname = os.path.join(log_path, msg["filename"])
                        with open(fname, 'wb') as f:
                            f.write(msg["data"])
                        print(f"[Cloud] Received remote edge metrics: {os.path.basename(fname)}")
            except Exception as e:
                print(f"[Cloud] Warning collecting remote metrics: {e}")

        edge_rows, cloud_rows = [], []
        for fpath in sorted(_glob.glob(os.path.join(log_path, 'metrics_raw_*.csv'))):
            with open(fpath, newline='') as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            if rows[0]['role'] == 'edge':
                edge_rows.extend(rows)
            elif rows[0]['role'] == 'cloud':
                cloud_rows.extend(rows)

        fieldnames = ['batch_id', 'batch_size', 'best_cut',
                      'edge_latency_ms', 'edge_fps', 'edge_ram_mb', 'edge_message_size_bytes',
                      'cloud_latency_ms', 'cloud_fps', 'cloud_ram_mb', 'cloud_message_size_bytes',
                      'e2e_latency_ms']

        edge_by_time = {r['edge_start_time']: r for r in edge_rows if r.get('edge_start_time')}
        pairs = []
        for c in cloud_rows:
            t = c.get('edge_start_time', '')
            pairs.append((edge_by_time.get(t, {}), c))

        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i, (e, c) in enumerate(pairs):
                w.writerow({
                    'batch_id':                i,
                    'batch_size':              e.get('batch_size') or c.get('batch_size', ''),
                    'best_cut':                e.get('best_cut')   or c.get('best_cut', ''),
                    'edge_latency_ms':         e.get('latency_ms', ''),
                    'edge_fps':                e.get('fps', ''),
                    'edge_ram_mb':             e.get('ram_mb', ''),
                    'edge_message_size_bytes': e.get('message_size_bytes', ''),
                    'cloud_latency_ms':        c.get('latency_ms', ''),
                    'cloud_fps':               c.get('fps', ''),
                    'cloud_ram_mb':            c.get('ram_mb', ''),
                    'cloud_message_size_bytes':c.get('message_size_bytes', ''),
                    'e2e_latency_ms':          c.get('e2e_latency_ms', ''),
                })

        def _avg(rows, key):
            vals = [float(r[key]) for r in rows if r.get(key) and float(r.get(key, 0)) > 0]
            return round(sum(vals) / len(vals), 3) if vals else None

        print('=' * 50)
        print(f"  DMSF SUMMARY  |  batches={len(pairs)}")
        print('=' * 50)
        print(f"  [EDGE]  latency={_avg(edge_rows,'latency_ms')} ms  fps={_avg(edge_rows,'fps')}  ram={_avg(edge_rows,'ram_mb')} MB")
        print(f"  [CLOUD] latency={_avg(cloud_rows,'latency_ms')} ms  fps={_avg(cloud_rows,'fps')}  ram={_avg(cloud_rows,'ram_mb')} MB")
        print(f"  [E2E]   latency={_avg(cloud_rows,'e2e_latency_ms')} ms")
        print('=' * 50)
        print(f"[Server] Saved {out_path}")

        for fpath in _glob.glob(os.path.join(log_path, 'metrics_raw_*.csv')):
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

    # ---------------------------------------------------------------------- #
    # Entry point called by DmsfRpcClient
    # ---------------------------------------------------------------------- #
    def inference_func(self, model, split_point, data, batch_size,
                       conf, iou, log_path, num_layers):
        if self.layer_id == 1:
            self.first_layer(model, split_point, data, batch_size, log_path)
        elif self.layer_id == num_layers:
            self.last_layer(model, split_point, batch_size, conf, iou, log_path)
