import os
import pickle
import sys
import time
import pika

from utils.split_selector import SplitSelector, SPLIT_CHANNELS_26N
from DmsfScheduler import INTERMEDIATE_QUEUE


class DmsfServer:
    def __init__(self, config):
        self.config = config
        rabbit = config['rabbit']
        self.total_clients = config['server']['clients']
        # Đọc từ section 'dmsf' nếu tích hợp vào split_inference, fallback về 'server'
        dmsf_cfg = config.get('dmsf', config.get('server', {}))
        self.split_point_cfg = str(dmsf_cfg.get('split-point', 'auto'))
        self.bandwidth_bps   = dmsf_cfg.get('bandwidth-mbps', 100.0) * 1e6

        credentials = pika.PlainCredentials(rabbit.get('username', 'guest'),
                                            rabbit.get('password', 'guest'))
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=rabbit['address'],
            port=rabbit.get('port', 5672),
            virtual_host=rabbit.get('virtual-host', '/'),
            credentials=credentials,
            heartbeat=0,
            blocked_connection_timeout=300,
        ))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue', durable=False)
        self.channel.queue_purge(queue='rpc_queue')
        self.channel.queue_declare(queue=INTERMEDIATE_QUEUE, durable=False)
        self.channel.queue_purge(queue=INTERMEDIATE_QUEUE)
        self.channel.queue_declare(queue='fps_queue', durable=False)
        self.channel.queue_purge(queue='fps_queue')
        self.channel.basic_consume(queue='fps_queue',
                                   on_message_callback=self.on_fps, auto_ack=False)
        self.reply_channel = self.connection.channel()

        # FPS tracking (fps_guide §7, §11.1)
        self._fps_times = []
        self._fps_start_t = None
        self._fps_window = 16
        self._fps_printed = False
        self._fps_stop_t = None
        self._last_done_t = None
        self._count_cloud_done = 0
        self._fps_hardcap_s = 300.0
        self._util_records = []   # collected from NOTIFY + CLOUD_DONE payloads
        log_path = config.get('log-path', '.')
        self._batch_log_path = os.path.join(log_path, 'batch_done_ns.log')
        self._util_log_path  = os.path.join(log_path, 'utilization.log')
        self._batch_size = config['server']['batch-size']
        open(self._batch_log_path, 'w').close()   # truncate at server start
        open(self._util_log_path,  'w').close()

        self.register_clients = [0] * len(self.total_clients)
        self.list_clients = []
        self.registered_ids = set()
        self.notified = False
        self.count_notify = 0

        self.edge_times_ms  = {}   # client_id -> {sp: ms}
        self.cloud_times_ms = {}   # client_id -> {sp: ms}
        self.client_devices = {}   # client_id -> device str
        self.client_bandwidth_mb_s = {}  # client_id -> measured uplink MB/s

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        print(f"[Server] Waiting for {self.total_clients} clients ...")

    def on_request(self, ch, method, _, body):
        msg = pickle.loads(body)
        action = msg['action']

        if action == 'REGISTER':
            client_id = msg['client_id']
            layer_id  = msg['layer_id']

            if str(client_id) in self.registered_ids:
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            self.registered_ids.add(str(client_id))
            self.list_clients.append((str(client_id), layer_id))
            self.register_clients[layer_id - 1] += 1

            self.client_devices[str(client_id)] = msg.get('device', 'cpu')
            et = msg.get('edge_times_ms')
            ct = msg.get('cloud_times_ms')
            if et:
                self.edge_times_ms[str(client_id)] = {int(k): v for k, v in et.items()}
            if ct:
                self.cloud_times_ms[str(client_id)] = {int(k): v for k, v in ct.items()}

            bw = msg.get('bandwidth_mb_s')
            if bw is not None:
                self.client_bandwidth_mb_s[str(client_id)] = float(bw)
                print(f"[Server] Measured bandwidth from {str(client_id)[:8]}: {bw:.2f} MB/s")

            print(f"[Server] REGISTER layer={layer_id} id={str(client_id)[:8]}  "
                  f"registered={self.register_clients}")

            if self.register_clients == self.total_clients and not self.notified:
                self.notified = True
                self.notify_clients()

        elif action == 'BW_TEST':
            client_id = msg['client_id']
            self._send_to_client(client_id, {'action': 'BW_ACK'})

        elif action == 'NOTIFY':
            # Save edge utilization piggybacked in this message
            util = msg.get('utilization')
            if util:
                util['client_id'] = str(msg.get('client_id', ''))
                self._util_records.append(util)
            self.count_notify += 1
            if self.count_notify == self.total_clients[0]:
                # Push one sentinel per cloud into INTERMEDIATE_QUEUE (after all edge batches).
                # Queue is FIFO → sentinel guaranteed to arrive after all real batches.
                # Each cloud exits when it dequeues its sentinel.
                n_cloud = self.total_clients[1]
                for _ in range(n_cloud):
                    self.channel.basic_publish(
                        exchange='', routing_key=INTERMEDIATE_QUEUE,
                        body=pickle.dumps({'action': 'SENTINEL'}))
                print(f"[Server] All edges done — pushed {n_cloud} sentinels. Sending STOP.")
                self.notify_clients(start=False)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self._fps_stop_t = time.time()
                self.connection.call_later(10.0, self._fps_drain_check)
                return

        elif action == 'CLOUD_DONE':
            # Save cloud utilization piggybacked in this message
            util = msg.get('utilization')
            if util:
                util['client_id'] = str(msg.get('client_id', ''))
                self._util_records.append(util)
            self._count_cloud_done += 1
            batches = msg.get('batches', 0)
            n_cloud = self.total_clients[1]
            print(f"[Server] CLOUD_DONE {self._count_cloud_done}/{n_cloud}  "
                  f"client={str(msg.get('client_id',''))[:8]}  batches={batches}")
            if self._count_cloud_done >= n_cloud:
                # All clouds finished — wait 2s for any in-flight DONEs then stop
                self.connection.call_later(2.0,
                    lambda: self._finish_fps_and_stop("all clouds done"))

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _get_bandwidth_bps(self, edge_clients):
        measured = [self.client_bandwidth_mb_s[str(cid)]
                    for cid, _ in edge_clients if str(cid) in self.client_bandwidth_mb_s]
        if measured:
            avg_mb_s = sum(measured) / len(measured)
            bps = avg_mb_s * 1e6 * 8
            print(f"[Server] Using measured bandwidth: {avg_mb_s:.2f} MB/s ({bps/1e6:.1f} Mbps)")
            return bps
        print(f"[Server] Using configured bandwidth: {self.bandwidth_bps/1e6:.1f} Mbps")
        return self.bandwidth_bps

    def _select_split_point(self):
        if self.split_point_cfg in ('3', '5', '7', '10'):
            sp = int(self.split_point_cfg)
            print(f"[Server] Fixed split_point={sp}")
            return sp

        # auto: dùng profile data từ clients
        edge_clients = [(cid, lid) for cid, lid in self.list_clients if lid == 1]
        cloud_clients = [(cid, lid) for cid, lid in self.list_clients if lid == len(self.total_clients)]

        if edge_clients and cloud_clients:
            # Lấy profile của client đầu tiên (1-1) hoặc trung bình (N-M)
            first_edge  = str(edge_clients[0][0])
            first_cloud = str(cloud_clients[0][0])
            et = self.edge_times_ms.get(first_edge)
            ct = self.cloud_times_ms.get(first_cloud)
            if et and ct:
                bandwidth_bps = self._get_bandwidth_bps(edge_clients)
                selector = SplitSelector(et, ct, channels=SPLIT_CHANNELS_26N)
                sp = selector.select(bandwidth_bps)
                selector.report(bandwidth_bps)
                print(f"[Server] Auto selected split_point={sp}")
                return sp

        print("[Server] No profile data available, defaulting to split_point=7")
        return 7

    def on_fps(self, ch, method, _props, body):
        t_ns = time.time_ns()
        t = t_ns / 1e9
        self._fps_times.append(t)
        self._last_done_t = t
        n = len(self._fps_times)
        W = self._fps_window
        bs = self._batch_size
        window_fps = None
        if n >= W:
            span = self._fps_times[-1] - self._fps_times[-W]
            if span > 0:
                window_fps = (W - 1) * bs / span
                print(f"[FPS] DONE #{n}  window_fps={window_fps:6.2f} (last {W} batches)")
        with open(self._batch_log_path, 'a') as f:
            if window_fps is None:
                f.write(f"{t_ns}\n")
            else:
                f.write(f"{t_ns} {window_fps:.2f}\n")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _fps_drain_check(self):
        # Safety hardcap only — primary stop is via CLOUD_DONE
        if self._fps_printed:
            return
        elapsed = time.time() - self._fps_stop_t if self._fps_stop_t else 0
        n_cloud = self.total_clients[1]
        if elapsed >= self._fps_hardcap_s:
            if self._count_cloud_done < n_cloud:
                # Clouds still running — extend deadline by 60s and keep waiting
                print(f"[Server] Hardcap {self._fps_hardcap_s:.0f}s reached but only "
                      f"{self._count_cloud_done}/{n_cloud} CLOUD_DONE — extending 60s ...")
                self._fps_stop_t = time.time() - (self._fps_hardcap_s - 60)
            else:
                self._finish_fps_and_stop(f"hard cap {self._fps_hardcap_s:.0f}s reached")
                return
        self.connection.call_later(10.0, self._fps_drain_check)

    def _finish_fps_and_stop(self, reason="grace elapsed"):
        if self._fps_printed:
            return
        self._fps_printed = True
        t = self._fps_times
        n = len(t)
        bs = self._batch_size
        print("=" * 60)
        if n >= 1 and self._fps_start_t is not None and t[-1] > self._fps_start_t:
            total_time = t[-1] - self._fps_start_t
            system_fps = n * bs / total_time
            print(f"  [SYSTEM FPS]      {system_fps:8.3f} fps   "
                  f"= {n} DONE x {bs} / {total_time:.2f}s  (START -> last DONE)")
            if n >= 2 and t[-1] > t[0]:
                span = t[-1] - t[0]
                steady = (n - 1) * bs / span
                print(f"  [steady-state]    {steady:8.3f} fps   "
                      f"= {n - 1} x {bs} / {span:.2f}s  (first -> last DONE)")
            if n >= 2:
                gaps = [t[i] - t[i - 1] for i in range(1, n) if t[i] > t[i - 1]]
                if gaps:
                    ref_mean = sum(bs / g for g in gaps) / len(gaps)
                    print(f"  [ref mean, N/U]   {ref_mean:8.3f} fps   "
                          f"(arithmetic mean of 1/dt — reference only, biased high)")
        else:
            print("  [SYSTEM FPS]      no DONEs received — nothing to report")
        print(f"  batches counted: {n}   stop reason: {reason}")
        print("=" * 60)
        self.channel.stop_consuming()

    def _collect_utilization(self):
        import datetime
        records = self._util_records
        n_expected = sum(self.total_clients)
        if len(records) < n_expected:
            print(f"[Utilization] Warning: only {len(records)}/{n_expected} reports received")

        # ── group by role ──────────────────────────────────────────────────────
        by_role = {}
        for r in records:
            role = r.get('role', '?')
            by_role.setdefault(role, []).append(r)

        lines_md   = []
        lines_print = []

        dmsf_cfg   = self.config.get('dmsf', self.config.get('server', {}))
        model_name = dmsf_cfg.get('weights', 'unknown')
        batch_size = self.config['server']['batch-size']
        clients_cfg = self.total_clients
        ts_str      = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        header = (f"# Utilization Report\n\n"
                  f"- **Time:** {ts_str}\n"
                  f"- **Model:** {model_name}\n"
                  f"- **Batch size:** {batch_size}\n"
                  f"- **Clients:** edge={clients_cfg[0]}  cloud={clients_cfg[1]}\n\n")
        lines_md.append(header)

        sep = "=" * 68
        lines_print.append(sep)
        lines_print.append(f"  UTILIZATION SUMMARY  |  {ts_str}")
        lines_print.append(sep)

        for role in ('edge', 'cloud'):
            recs = by_role.get(role, [])
            if not recs:
                continue
            utils   = [r.get('utilization', 0.0) for r in recs]
            pkgs    = [r.get('packages', 0)       for r in recs]
            busy_s  = [r.get('busy_ns',  0) / 1e9 for r in recs]
            total_s = [r.get('total_ns', 0) / 1e9 for r in recs]
            n       = len(recs)
            avg_u   = sum(utils)   / n
            min_u   = min(utils)
            max_u   = max(utils)
            avg_pkg = sum(pkgs)    / n
            avg_busy  = sum(busy_s)  / n
            avg_total = sum(total_s) / n

            # per-client rows
            per_rows = []
            for r in sorted(recs, key=lambda x: str(x.get('client_id',''))):
                cid = str(r.get('client_id', ''))[:8]
                per_rows.append(
                    f"| `{cid}` | {r.get('packages',0):>4d} "
                    f"| {r.get('busy_ns',0)/1e9:>8.3f} "
                    f"| {r.get('total_ns',0)/1e9:>8.3f} "
                    f"| **{r.get('utilization',0.0)*100:5.2f}%** |")

            lines_md.append(f"## {role.capitalize()} clients  ({n} devices)\n")
            lines_md.append(f"| client_id | packages | busy (s) | total (s) | utilization |")
            lines_md.append(f"|-----------|:--------:|:--------:|:---------:|:-----------:|")
            lines_md.extend(per_rows)
            lines_md.append(f"\n**Summary:** avg={avg_u*100:.2f}%  "
                            f"min={min_u*100:.2f}%  max={max_u*100:.2f}%  "
                            f"(avg packages={avg_pkg:.1f}  "
                            f"avg busy={avg_busy:.3f}s  avg total={avg_total:.3f}s)\n")

            lines_print.append(f"  {role.upper()} ({n} devices)")
            lines_print.append(f"    {'client':36s}  {'pkg':>5}  {'busy(s)':>8}  "
                               f"{'total(s)':>9}  {'util':>7}")
            lines_print.append(f"    {'-'*36}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*7}")
            for r in sorted(recs, key=lambda x: str(x.get('client_id',''))):
                cid = str(r.get('client_id', ''))
                lines_print.append(
                    f"    {cid:36s}  {r.get('packages',0):>5d}  "
                    f"{r.get('busy_ns',0)/1e9:>8.3f}  "
                    f"{r.get('total_ns',0)/1e9:>9.3f}  "
                    f"{r.get('utilization',0.0)*100:>6.2f}%")
            lines_print.append(
                f"    {'AVERAGE':36s}  {avg_pkg:>5.1f}  "
                f"{avg_busy:>8.3f}  {avg_total:>9.3f}  {avg_u*100:>6.2f}%")
            lines_print.append(
                f"    {'MIN / MAX':36s}  {'':>5}  {'':>8}  "
                f"{'':>9}  {min_u*100:.2f}% / {max_u*100:.2f}%")
            lines_print.append("")

        lines_print.append(sep)
        for ln in lines_print:
            print(ln)

        # ── write CSV + MD ─────────────────────────────────────────────────────
        with open(self._util_log_path, 'w') as f:
            f.write("role,client_id,utilization,packages,busy_ns,total_ns\n")
            for r in records:
                f.write(f"{r.get('role','?')},{r.get('client_id','')[:8]},"
                        f"{r.get('utilization',0.0):.6f},"
                        f"{r.get('packages',0)},"
                        f"{r.get('busy_ns',0)},"
                        f"{r.get('total_ns',0)}\n")

        util_md_path = os.path.join(os.path.dirname(self._util_log_path), 'utilization.md')
        with open(util_md_path, 'w') as f:
            f.write('\n'.join(lines_md))
        print(f"[Utilization] {len(records)} records → {self._util_log_path}  |  {util_md_path}")

    def notify_clients(self, start=True):
        if start:
            split_point = self._select_split_point()
            # Đọc DMSF-specific config: ưu tiên section 'dmsf', fallback về 'server'/'inference'
            dmsf_cfg = self.config.get('dmsf', self.config.get('server', {}))
            inf_cfg  = self.config.get('inference', {})
            for (client_id, layer_id) in self.list_clients:
                # Device: dùng device mà client đã đăng ký (client tự biết hardware của mình)
                client_device = self.client_devices.get(str(client_id), 'cpu')
                response = {
                    'action':      'START',
                    'message':     'Server accept the connection',
                    'split_point': split_point,
                    'batch_size':  self.config['server']['batch-size'],
                    'nc':          dmsf_cfg.get('nc', 10),
                    'weights':     dmsf_cfg.get('weights', 'best_dmsf26n.pt'),
                    'data':        self.config['data'],
                    'conf':        dmsf_cfg.get('conf', inf_cfg.get('conf', 0.001)),
                    'iou':         dmsf_cfg.get('iou', inf_cfg.get('iou', 0.6)),
                    'imgsz':       dmsf_cfg.get('imgsz', inf_cfg.get('imgsz', 640)),
                    'log_path':    self.config.get('log-path', '.'),
                    'device_edge': client_device if layer_id == 1 else 'cpu',
                    'device_cloud': client_device if layer_id > 1 else 'cpu',
                    'num_layers':  len(self.total_clients),
                }
                self._send_to_client(client_id, response)
            self._fps_start_t = time.time()   # clock starts when START is dispatched
        else:
            response = {'action': 'STOP', 'message': 'Stop inference !!!'}
            for (client_id, _) in self.list_clients:
                self._send_to_client(client_id, response)

    def _send_to_client(self, client_id, message):
        reply_q = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_q, durable=False)
        self.reply_channel.basic_publish(
            exchange='', routing_key=reply_q, body=pickle.dumps(message))
        print(f"[Server] Sent {message['action']} to {str(client_id)[:8]}")

    def start(self):
        self.channel.start_consuming()
        self._collect_utilization()   # reads from _util_records collected during event loop
        self.connection.close()
        sys.exit(0)
