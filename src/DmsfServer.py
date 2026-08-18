import os
import pickle
import socket
import sys
import time
import pika

import DmsfResults as R
import DmsfFreeTime as FT
import DmsfMapEval as MapEval
from DmsfBrokerRam import BrokerRamSampler
from utils.split_selector import SplitSelector, SPLIT_CHANNELS_26N
from DmsfScheduler import (CLUSTER_ID, INTERMEDIATE_QUEUE, UTILIZATION_QUEUE,
                           FREETIME_QUEUE, MSGSIZE_QUEUE, MAP_QUEUE)


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
        # A crashed run's stale completions must not inflate this run's count.
        self.channel.queue_purge(queue='fps_queue')
        self.channel.basic_consume(queue='fps_queue',
                                   on_message_callback=self.on_fps, auto_ack=False)
        # Device measurement reports land here. A dedicated queue is what lets
        # publisher and consumer be alive at different moments: the clouds
        # finish long after the server stops reading the control queue, so a
        # report sent to rpc_queue would never be consumed.
        self.channel.queue_declare(queue=UTILIZATION_QUEUE, durable=False)
        self.channel.queue_purge(queue=UTILIZATION_QUEUE)
        # Every measurement queue is declared and purged here, whether or not
        # its feature is on: a crashed run's stale reports must not be counted
        # into this one, and a queue that exists but is never published to costs
        # nothing.
        for queue in (FREETIME_QUEUE, MSGSIZE_QUEUE, MAP_QUEUE):
            self.channel.queue_declare(queue=queue, durable=False)
            self.channel.queue_purge(queue=queue)
        self.reply_channel = self.connection.channel()

        # ── measurement features (00 §5) ───────────────────────────────────────
        # Every flag lives HERE, in the server's config, and travels in the
        # dispatch message. No worker reads one from its own config file at run
        # time, or a measurement setting has to be changed on N machines, the
        # copies drift, and a run silently mixes two configurations
        # (README invariant 9).
        self.free_time_cfg = config.get('free_time') or {}
        self.free_time_enable = bool(self.free_time_cfg.get('enable', False))
        self.msg_size_cfg = config.get('message_size') or {}
        self.msg_size_enable = bool(self.msg_size_cfg.get('enable', False))
        self.map_cfg = config.get('map') or {}
        self.map_enable = bool(self.map_cfg.get('enable', False))
        self.map_window_batches = int(
            self.map_cfg.get('window_batches', MapEval.DEFAULT_WINDOW_BATCHES))
        # Exactly one worker measures payload size, and the SERVER picks it: the
        # first edge to register (12 §1). Registration order needs no
        # configuration, is stable within a run, and is known before dispatch.
        self._msg_size_client = None

        # Throughput tracking (guide 02)
        self._fps_times = []          # every arrival — the authoritative series
        self._cluster_times = {}      # cluster id -> arrivals, for the breakdown
        self._fps_start_t = None      # the shared START, set when work is dispatched
        self._fps_window = R.WINDOW
        self._fps_printed = False
        self._fps_stop_t = None
        self._last_done_t = None
        self._count_cloud_done = 0
        # 02 §9. grace_s is how long to keep collecting once nothing is left to
        # arrive; shutdown_timeout_s is the absolute cap from the stop broadcast,
        # and it only ever fires on failure — on a healthy run grace wins first.
        fps_cfg = config.get('fps') or {}
        self._fps_grace_s   = float(fps_cfg.get('grace_s', 10.0))
        self._fps_hardcap_s = float(fps_cfg.get('shutdown_timeout_s', 300.0))
        self._util_records = []       # drained from UTILIZATION_QUEUE at shutdown

        self.log_path = config.get('log-path', '.')
        self._batch_log_path   = os.path.join(self.log_path, 'batch_done_ns.log')
        self._rate_ns_log_path = os.path.join(self.log_path, 'fps_cluster_ns.log')
        self._rate_log_path    = os.path.join(self.log_path, 'fps_cluster.log')
        self._util_log_path    = os.path.join(self.log_path, 'utilization.log')
        self._util_cl_log_path = os.path.join(self.log_path, 'utilization_cluster.log')
        self._lat_log_path     = os.path.join(self.log_path, 'latency_cluster.log')
        self._cut_log_path     = os.path.join(self.log_path, 'cut_change_ns.log')
        self._ft_log_path      = os.path.join(self.log_path, 'free_time.log')
        self._ft_cl_log_path   = os.path.join(self.log_path, 'free_time_cluster.log')
        self._ft_ser_log_path  = os.path.join(self.log_path, 'free_time_series.log')
        self._ram_ns_log_path  = os.path.join(self.log_path, 'broker_ram_ns.log')
        self._ram_log_path     = os.path.join(self.log_path, 'broker_ram.log')
        self._msz_log_path     = os.path.join(self.log_path, 'message_size.log')
        self._msz_ser_log_path = os.path.join(self.log_path, 'message_size_series.log')
        self._map_log_path     = os.path.join(self.log_path, 'map.log')
        self._map_win_log_path = os.path.join(self.log_path, 'map_window.log')
        # Ground truth is the only input to this measurement the run does not
        # produce, so it is the only one worth relocating: `map.label_path`
        # points at real VisDrone labels, a second reference set, or a folder
        # shared between checkouts. Resolved ONCE, here, and printed at startup
        # — an empty label folder is otherwise only discovered at shutdown,
        # after the whole run has been paid for.
        self.map_label_dir = MapEval.resolve_label_dir(self.map_cfg, self.log_path)
        if self.map_enable:
            n_labels = 0
            if os.path.isdir(self.map_label_dir):
                n_labels = len([n for n in os.listdir(self.map_label_dir)
                                if n.endswith('.txt')])
            print(f"[mAP] ground truth: {self.map_label_dir} "
                  f"({n_labels} label file(s))")
            if not n_labels:
                print("[mAP] WARNING: no labels there — this run will produce "
                      "empty map.log and map_window.log. Generate them with "
                      "`python make_map_labels.py`, or fix map.label_path.")
        self._batch_size = config['server']['batch-size']
        # All sixteen, once, centrally, before any worker can write. Truncating
        # per-worker instead lets a late starter wipe a file another worker is
        # already appending to.
        R.truncate_all(self.log_path)
        # Scratch from a previous run is cleared on the same terms (05 §5): a
        # leftover metrics_pivot.lock makes every later run skip its pivot, and
        # leftover metrics_raw_*.csv rows get merged into this run's summary.
        R.purge_scratch(self.log_path)
        # And the write-once prediction cache, which is the dangerous one: it is
        # keyed by frame index, so a leftover from run N wins forever — run N+1
        # silently reuses it for every frame they share and the numbers look
        # entirely plausible. map/label/ is ground truth and is never touched.
        MapEval.purge_scratch(self.log_path, self.map_label_dir)

        # The queue host, sampled from the server for the whole run. The window
        # opens HERE, in the constructor, before any worker has registered and
        # long before anything is published: without a stretch of the host at
        # rest there is no denominator for "what did running the system cost it"
        # (11 §6).
        self._ram = BrokerRamSampler(config, rabbit, self._ram_ns_log_path)
        self._ram.start()
        self._host = socket.gethostname()
        self._host_cpu0 = FT.cpu_snapshot()

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

            # First edge to register measures the payload size — one worker, not
            # the fleet. Every edge here publishes the same payload shape from
            # the same split point, so nine measuring workers would produce one
            # number nine times at nine times the cost (12 §1).
            if layer_id == 1 and self._msg_size_client is None:
                self._msg_size_client = str(client_id)

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
            # Pure control message. Measurement rides UTILIZATION_QUEUE so that
            # a device's report survives the server no longer reading this queue.
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
                # NOTE: no stop_consuming() here. The clouds are still draining
                # their backlog, and every completion they publish from now on
                # is one this run would otherwise undercount.
                self.connection.call_later(1.0, self._fps_drain_check)
                return

        elif action == 'CLOUD_DONE':
            # Pure control message; see NOTIFY above.
            self._count_cloud_done += 1
            batches = msg.get('batches', 0)
            n_cloud = self.total_clients[1]
            print(f"[Server] CLOUD_DONE {self._count_cloud_done}/{n_cloud}  "
                  f"client={str(msg.get('client_id',''))[:8]}  batches={batches}")
            if self._count_cloud_done >= n_cloud:
                # Every cloud has published everything it will publish. Grace
                # covers the DONEs still in flight on the broker between their
                # last publish and this control message.
                self.connection.call_later(
                    self._fps_grace_s,
                    lambda: self._finish_fps_and_stop("all clouds done + grace"))

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
        # The arrival is the event. The body carries an identity — the producing
        # cluster — and is read ONLY to bucket this unit for the breakdown, never
        # for timing. A garbled body can mis-bucket one unit; it can never
        # distort a rate. One clock reading serves both the math and the log.
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

        # An unrecognized body is bucketed as 'unknown', never dropped: an old
        # worker degrades the breakdown instead of losing units. Every arrival
        # lands in exactly one bucket, so cluster done/frames sum to SYSTEM.
        try:
            cluster = body.decode().strip() or 'unknown'
        except Exception:
            cluster = 'unknown'
        times = self._cluster_times.setdefault(cluster, [])
        times.append(t)
        # A cluster reaches its first full window later than the system does.
        # That is correct, not a bug — it needs W completions of its own.
        c_window = None
        if len(times) >= W:
            c_span = times[-1] - times[-W]
            if c_span > 0:
                c_window = (W - 1) * bs / c_span

        # A failed write must not escape this callback. It would skip the ack
        # below, the broker would redeliver, and the unit would be counted
        # twice - which is worse than losing a line from the live series. The
        # in-memory lists are already updated, so the shutdown summary is intact.
        try:
            R.append_batch_done(self._batch_log_path, t_ns, window_fps)
            R.append_cluster_rate(self._rate_ns_log_path, t_ns, cluster,
                                  len(times), c_window)
        except Exception as e:
            print(f"[FPS] Warning: could not append DONE #{n} to the live logs: {e}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def _fps_drain_check(self):
        """Drain-watch: keep collecting past first-stage shutdown — 02 §5.3.

        The primary stop is the CLOUD_DONE path; this is the safety net for the
        case where a cloud never reports at all.

        The hard cap is **absolute**, measured from the stop broadcast. An
        earlier version extended it by 60 s whenever a cloud was still missing,
        which is unbounded: a cloud that has died is missing forever, so the
        server never exited and never wrote its summary. A cloud that is merely
        slow is not the case this cap is for — it is still publishing, so it
        finishes through CLOUD_DONE long before the cap, and a run that needs
        longer than the cap says so in ``fps.shutdown_timeout_s`` (02 §9).
        """
        if self._fps_printed:
            return
        now = time.time()
        n_cloud = self.total_clients[1]
        elapsed = now - self._fps_stop_t if self._fps_stop_t else 0.0

        if elapsed >= self._fps_hardcap_s:
            self._finish_fps_and_stop(
                f"hard cap {self._fps_hardcap_s:.0f}s reached with "
                f"{self._count_cloud_done}/{n_cloud} CLOUD_DONE")
            return

        # Nothing has arrived for a whole grace window and every cloud has
        # reported: the backlog is drained and the in-flight unit has landed.
        last = self._last_done_t or self._fps_stop_t
        if (self._count_cloud_done >= n_cloud
                and last is not None and (now - last) >= self._fps_grace_s):
            self._finish_fps_and_stop("work drained + grace")
            return

        self.connection.call_later(1.0, self._fps_drain_check)

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

    def _write_rate_summary(self):
        """fps_cluster.log — the throughput summary (01 §3.3)."""
        try:
            system = R.write_rate_summary(self._rate_log_path, self._fps_start_t,
                                          self._cluster_times, self._batch_size)
            if system is None:
                print("[FPS] WARNING: no completions recorded — "
                      f"{os.path.basename(self._rate_log_path)} left empty")
            else:
                print(f"[FPS] Summary -> {self._rate_log_path}  "
                      f"({system['clusters']} cluster(s), {system['done']} units, "
                      f"{system['fps']:.3f} fps)")
        except Exception as e:
            print(f"[FPS] Warning: could not write rate summary: {e}")

    def _drain_utilization_reports(self, timeout_s=30.0):
        """Poll the dedicated report queue until every device is in, or we time out.

        A partial collection is acceptable and warns; it must never hang or
        abort the run. By the time this runs the throughput drain has already
        waited for the work queues plus a grace window, so every tail has seen
        STOP and published — 30 s is comfortable.
        """
        n_expected = sum(self.total_clients)
        seen = {}
        deadline = time.time() + timeout_s
        while len(seen) < n_expected and time.time() < deadline:
            method, _, body = self.channel.basic_get(queue=UTILIZATION_QUEUE,
                                                     auto_ack=True)
            if not method:
                time.sleep(0.2)
                continue
            try:
                msg = pickle.loads(body)
            except Exception:
                continue                      # unpicklable body: skip, never raise
            if msg.get('action') != 'UTILIZATION':
                continue
            msg['client_id'] = str(msg.get('client_id', ''))
            # Column 1 of utilization.log is the report's ARRIVAL at the server,
            # not a device timestamp — device clocks are never assumed synced.
            msg['arrival_ns'] = time.time_ns()
            seen[msg['client_id']] = msg       # last write wins: dedupe on retry

        self._util_records = list(seen.values())
        if len(self._util_records) < n_expected:
            print(f"[Utilization] Collected {len(self._util_records)}/{n_expected} "
                  f"reports before timeout")
        return self._util_records

    def _collect_utilization(self):
        import datetime
        try:
            records = self._drain_utilization_reports()
        except Exception as e:
            # A broker error here used to propagate out of start() and take the
            # archive with it, so a hiccup on the last drain cost the whole run.
            print(f"[Utilization] Warning: could not drain reports: {e}")
            records = []
        if not records:
            print("[Utilization] WARNING: no device reports — utilization and "
                  "latency logs left empty")
            return

        # ── conformant result files (01 §3.4-3.6) ──────────────────────────────
        try:
            for r in sorted(records, key=lambda x: (x.get('role', ''),
                                                    x.get('client_id', ''))):
                R.append_utilization_device(self._util_log_path, r)
            R.write_utilization_cluster(self._util_cl_log_path, records)
            R.write_latency_cluster(self._lat_log_path, records)
            print(f"[Utilization] {len(records)} device line(s) -> "
                  f"{self._util_log_path}")
            print(f"[Utilization] roll-up -> {self._util_cl_log_path}")
            print(f"[Latency]     pooled  -> {self._lat_log_path}")
        except Exception as e:
            print(f"[Utilization] Warning: could not write result logs: {e}")

        # Σ service samples must equal that device's busy_s (04 §2.1). They come
        # from the same intervals, so a mismatch means one of the two is
        # instrumented at the wrong point — worth saying out loud, not asserting.
        for r in records:
            svc = r.get('service_ms') or []
            if not svc:
                continue
            drift = abs(sum(svc) / 1e3 - r.get('busy_ns', 0) / 1e9)
            if drift > 0.01:
                # ASCII only: this runs on a Windows console whose default
                # codepage cannot encode most of what looks fine in an editor,
                # and an encode error here would take the shutdown path with it.
                print(f"[Latency] Warning: client={r['client_id'][:8]} "
                      f"sum(service) differs from busy_s by {drift:.3f}s")

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

        # ── human-readable companion ───────────────────────────────────────────
        # Not a result file: utilization.log above is the machine-readable one,
        # and this stays out of the archive manifest deliberately (05 §5).
        util_md_path = os.path.join(self.log_path, 'utilization.md')
        with open(util_md_path, 'w') as f:
            f.write('\n'.join(lines_md))
        print(f"[Utilization] human report -> {util_md_path}")

    def _drain_reports(self, queue, action, expected, timeout_s, label):
        """Poll one measurement queue until every expected report is in, or the
        timeout expires. Partial is acceptable and warns; it never hangs the run.

        Reports are keyed by client, so a retry replaces rather than duplicates.
        An unpicklable body is skipped, never raised — telemetry that takes the
        shutdown path with it costs the whole run's results.
        """
        seen = {}
        deadline = time.time() + timeout_s
        while len(seen) < expected and time.time() < deadline:
            method, _, body = self.channel.basic_get(queue=queue, auto_ack=True)
            if not method:
                time.sleep(0.2)
                continue
            try:
                msg = pickle.loads(body)
            except Exception:
                continue
            if msg.get('action') != action:
                continue
            msg['client_id'] = str(msg.get('client_id', ''))
            # Column 1 of every one of these files is the report's ARRIVAL at
            # the server, not a device timestamp: device clocks are never
            # assumed to be in sync, and the position in the run travels
            # separately as an offset on the device's own clock.
            msg['arrival_ns'] = time.time_ns()
            seen[msg['client_id']] = msg
        if len(seen) < expected:
            print(f"[{label}] Collected {len(seen)}/{expected} report(s) before timeout")
        return list(seen.values())

    def _collect_free_time(self):
        """free_time.log + free_time_cluster.log + free_time_series.log (10).

        Skipped entirely when the feature is off, so shutdown never burns a
        timeout polling a queue nobody will publish to — a stall and a scary
        ``0/N`` caused by a setting working exactly as intended
        (README invariant 10). The three files still exist, empty.
        """
        if not self.free_time_enable:
            return
        try:
            records = self._drain_reports(
                FREETIME_QUEUE, 'FREE_TIME', sum(self.total_clients),
                float(self.free_time_cfg.get('collect_timeout_s', 30)), 'FreeTime')
        except Exception as e:
            print(f"[FreeTime] Warning: could not drain reports: {e}")
            return
        if not records:
            print("[FreeTime] WARNING: no device reports — the three free-time "
                  "logs are empty for this run")
            return

        host_idle = None
        cpu1 = FT.cpu_snapshot()
        if self._host_cpu0 and cpu1 and cpu1[1] > self._host_cpu0[1]:
            host_idle = ((cpu1[0] - self._host_cpu0[0])
                         / (cpu1[1] - self._host_cpu0[1]))
        ts_ns = time.time_ns()
        try:
            R.write_free_time(self._ft_log_path, records, ts_ns)
            R.write_free_time_cluster(self._ft_cl_log_path, records, ts_ns,
                                      server_host=self._host,
                                      server_host_idle=host_idle)
            R.write_free_time_series(self._ft_ser_log_path, records, ts_ns)
        except Exception as e:
            print(f"[FreeTime] Warning: could not write result logs: {e}")
            return
        span = sum(r.get('span_ns', 0) for r in records)
        free = sum(r.get('free_ns', 0) for r in records)
        print(f"[FreeTime]    {len(records)} device line(s) -> {self._ft_log_path}")
        print(f"[FreeTime]    roll-up -> {self._ft_cl_log_path}  "
              f"(SYSTEM free={free / span * 100 if span else 0:.2f}%)")
        print(f"[FreeTime]    series  -> {self._ft_ser_log_path}")

    def _collect_message_size(self):
        """message_size.log + message_size_series.log (12).

        Exactly one report is expected because exactly one worker was told to
        measure. Skipped whole when the feature is off.
        """
        if not self.msg_size_enable:
            return
        try:
            reports = self._drain_reports(
                MSGSIZE_QUEUE, 'MESSAGE_SIZE', 1,
                float(self.msg_size_cfg.get('collect_timeout_s', 30)), 'MessageSize')
        except Exception as e:
            print(f"[MessageSize] Warning: could not drain reports: {e}")
            return
        if not reports:
            print("[MessageSize] WARNING: no report from the elected worker — "
                  "both message-size logs are empty for this run")
            return
        ts_ns = time.time_ns()
        try:
            lines = R.write_message_size(self._msz_log_path, reports,
                                         self._batch_size, ts_ns)
            R.write_message_size_series(self._msz_ser_log_path, reports, ts_ns)
            for line in lines:
                print(f"[MessageSize] {line}")
        except Exception as e:
            print(f"[MessageSize] Warning: could not write result logs: {e}")

    def _collect_map(self):
        """Drain map_queue: one zip of raw predictions per completing device.

        Merged per CLUSTER, because the cluster is the scope the metric is
        reported at and its devices share a split point, so their detections
        form one valid pooled set. Expected counts the clouds, since each ships
        one report and each holds part of the frame range.
        """
        if not self.map_enable:
            return {}
        collected = os.path.join(self.log_path, MapEval.COLLECT_ROOT)
        expected = max(self.total_clients[-1], 1)
        by_cluster, seen = {}, 0
        deadline = time.time() + float(self.map_cfg.get('collect_timeout_s', 30))
        while seen < expected and time.time() < deadline:
            method, _, body = self.channel.basic_get(queue=MAP_QUEUE, auto_ack=True)
            if not method:
                time.sleep(0.2)
                continue
            try:
                msg = pickle.loads(body)
            except Exception:
                continue
            if msg.get('action') != 'MAP_PRED':
                continue
            seen += 1
            cluster = msg.get('cluster') or 'unknown'
            destination = os.path.join(collected, MapEval.safe_name(cluster))
            n = MapEval.unpack_predictions(msg.get('payload') or b'', destination)
            by_cluster[cluster] = destination
            print(f"[mAP] {cluster}: {n} prediction file(s) after "
                  f"{str(msg.get('client_id', ''))[:8]}")
        if seen < expected:
            print(f"[mAP] Collected {seen}/{expected} device report(s) before timeout")
        return by_cluster

    def _write_map_files(self, by_cluster):
        """map.log + map_window.log — this project's extension, scored HERE.

        mAP is a ranking metric over a pooled detection set, so a per-device
        score has no combination that reconstructs the cluster's. Every failure
        path degrades to a warning and an omitted line: a missing ground-truth
        file must never cost the run one of its fourteen result files.
        """
        if not self.map_enable:
            return
        label_dir = self.map_label_dir
        gts = MapEval.load_boxes(label_dir)
        if not gts:
            print(f"[mAP] WARNING: no ground truth in {label_dir} — "
                  f"map.log and map_window.log are empty for this run")
            return
        if not by_cluster:
            print("[mAP] WARNING: no predictions collected — "
                  "map.log and map_window.log are empty for this run")
            return

        max_det = int(self.map_cfg.get('max_det', 100))
        per_cluster = {}
        for cluster, directory in by_cluster.items():
            preds = MapEval.load_boxes(directory, with_conf=True, max_det=max_det)
            if preds:
                n_batches = (max(preds) - 1) // self._batch_size + 1
                print(f"[mAP] scoring {cluster}: {len(preds)} predicted frame(s) "
                      f"vs {len(gts)} GT frame(s), "
                      f"{max(n_batches - self.map_window_batches + 1, 1)} window(s), "
                      f"max_det={max_det} — this takes a minute or two")
            if not preds:
                print(f"[mAP] WARNING: {cluster} shipped no readable predictions "
                      f"— omitted (a 0.0000 would be a false accuracy claim)")
                continue
            result = MapEval.evaluate_cluster(preds, gts, self._batch_size,
                                              self.map_window_batches)
            if result["all"][0] is None and not result["windows"]:
                print(f"[mAP] WARNING: {cluster} matched no ground-truth frame "
                      f"— omitted rather than written as zeros")
                continue
            per_cluster[cluster] = result

        ts_ns = time.time_ns()          # one timestamp for the whole collection
        try:
            for line in R.write_map(self._map_log_path, ts_ns, per_cluster,
                                    self.map_window_batches):
                print(f"[mAP] {line}")
            windows = R.write_map_window(self._map_win_log_path, ts_ns, per_cluster)
            print(f"[mAP] {len(windows)} window line(s) -> {self._map_win_log_path}")
        except Exception as e:
            print(f"[mAP] Warning: could not write the accuracy logs: {e}")

    def _collect_and_write_map(self):
        """Collect the predictions, then score them. One step so that a failure
        in either half cannot leave the other half half-done."""
        self._write_map_files(self._collect_map())

    def _write_broker_ram(self):
        """Stop the sampler past the drain, then write its summary (11 §6)."""
        if not self._ram.enabled:
            return
        try:
            self._ram.stop()
            lines = R.write_broker_ram(self._ram_log_path, self._ram.samples,
                                       self._ram.meta())
            for line in lines:
                print(f"[BrokerRAM] {line}")
        except Exception as e:
            print(f"[BrokerRAM] Warning: could not write the summary: {e}")

    def notify_clients(self, start=True):
        if start:
            split_point = self._select_split_point()
            # The one control-plane decision this baseline makes. Timestamped
            # before the broadcast so it marks when the decision was made rather
            # than when it landed (01 §3.7). A run with an adaptive controller
            # would append here on every cut change too.
            try:
                mode = ('auto' if self.split_point_cfg == 'auto' else 'fixed')
                R.append_event(self._cut_log_path, time.time_ns(), CLUSTER_ID,
                               f"split point set to {split_point} ({mode})")
            except Exception as e:
                print(f"[Events] Warning: could not log split-point event: {e}")
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
                    # Every measurement setting for this run, decided here and
                    # carried to the worker. A worker reads none of these from
                    # its own config file (README invariant 9), so a run can
                    # never silently mix two measurement configurations.
                    'measure': {
                        'free_time':      self.free_time_enable,
                        'free_time_bucket_s': float(
                            self.free_time_cfg.get('bucket_s', 1.0)),
                        # ... and exactly one worker is told to measure size.
                        'message_size':   (self.msg_size_enable
                                           and str(client_id) == self._msg_size_client),
                        'map':            self.map_enable,
                        'map_conf':       float(self.map_cfg.get('conf', 0.001)),
                        'cluster':        CLUSTER_ID,
                    },
                }
                self._send_to_client(client_id, response)
            self._fps_start_t = time.time()   # clock starts when START is dispatched
            # The host stops being at rest here, not when the first message
            # lands: marks partition the RAM series, they never gate it (11 §6).
            self._ram.mark('run')
            if self.msg_size_enable and self._msg_size_client:
                print(f"[MessageSize] elected {self._msg_size_client[:8]} "
                      f"(first edge to register) as the measuring worker")
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

    def _archive_run(self):
        """Snapshot this run's logs + the config that produced them (05).

        Runs last, after every shutdown writer, so the snapshot is complete.
        Failure is non-fatal: the run still closes its connections and exits.
        """
        try:
            R.archive_run(self.log_path,
                          self.config.get('results-path',
                                          os.path.join(self.log_path, 'results')),
                          R.run_tag(self.config),
                          self.config.get('config-path', 'config.yaml'))
        except Exception as e:
            print(f"[Archive] Warning: could not archive run: {e}")

    def request_stop(self, reason="interrupted"):
        """End the consume loop from outside it — the Ctrl+C handler's entry
        point. Prints the same summary a self-terminating run prints, so an
        interrupted run is still a readable one."""
        try:
            self._finish_fps_and_stop(reason)
        except Exception as e:
            print(f"[Server] Warning: could not stop cleanly ({e}); "
                  f"forcing the consume loop to end")
            try:
                self.channel.stop_consuming()
            except Exception:
                pass

    def start(self):
        # stop_consuming() fires in the finalizer, not at first-stage shutdown —
        # that is what actually ends the loop, and leaving it out until then is
        # what lets late completions from the slower tier still be counted.
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            # Reached only if the signal lands somewhere the handler cannot
            # convert into a clean stop. The results are still worth writing.
            print("\n[Server] Interrupted — writing what this run produced.")
        finally:
            self._finalize()
        sys.exit(0)

    def _finalize(self):
        """Every shutdown writer, once, whatever ended the run.

        In a `finally`, and each step independently guarded, because these were
        four bare statements after start_consuming(): anything that ended the
        loop other than a clean return — Ctrl+C, a broker error during the
        drain — skipped all of them, and the archive silently never happened.
        The archive runs last and unconditionally, so a run that produced only
        a partial set of logs still keeps that partial set.
        """
        if getattr(self, '_finalized', False):
            return
        self._finalized = True
        # Shutdown order per 01 §4: throughput summary, then the device reports
        # and everything rolled up from them, then the RAM summary — which
        # closes its window a couple of seconds past the last real collector —
        # then the accuracy pair, then the archive. The archive runs last so the
        # snapshot is complete, and unconditionally so a run that produced a
        # partial set still keeps that partial set.
        #
        # The accuracy step is deliberately LAST of the measurement steps, and
        # that is also why the RAM window closes BEFORE it: scoring takes
        # minutes on a long run, and leaving it inside the window stretches
        # `phase=run` across a stretch where nothing is running, quietly
        # dragging `run_minus_idle_mb` toward zero. Nothing in the accuracy path
        # may cost — or distort — one of the fourteen result files.
        for step in (self._write_rate_summary, self._collect_utilization,
                     self._collect_free_time, self._collect_message_size,
                     self._write_broker_ram, self._collect_and_write_map,
                     self._archive_run):
            try:
                step()
            except Exception as e:
                print(f"[Shutdown] Warning: {step.__name__} failed: {e}")
        try:
            self.connection.close()
        except Exception:
            pass
