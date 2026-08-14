import csv
import os
import pickle
import time

import cv2
import psutil
import torch
from tqdm import tqdm

from utils.metrics import non_max_suppression
from DmsfFreeTime import FreeTimeTracker
from DmsfMessageSize import MessageSizeRecorder
from DmsfMapEval import PredictionWriter


INTERMEDIATE_QUEUE = 'dmsf_intermediate_queue'
METRICS_EXCHANGE   = 'dmsf_metrics_fanout'
UTILIZATION_QUEUE  = 'utilization_queue'
# One dedicated queue per measurement, for the reason UTILIZATION_QUEUE exists:
# the server stops reading control the moment the last edge reports, but the
# clouds finish later, so a report sent to rpc_queue would never be consumed. On
# its own queue a report simply waits on the broker — publisher and consumer
# never need to be alive at the same moment.
FREETIME_QUEUE     = 'freetime_queue'
MSGSIZE_QUEUE      = 'msgsize_queue'
MAP_QUEUE          = 'map_queue'

# This baseline runs a single edge->cloud cluster. The result format is written
# cluster-generic anyway (guide 01 §3.2-3.3), so an N-cluster variant only has to
# hand each device a different id here — every log, roll-up and chart follows.
CLUSTER_ID = f"{INTERMEDIATE_QUEUE}_0"


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

        # FPS tracking state (populated by last_layer → _finish_fps)
        self._fps_start_t = None
        self._fps_times = []
        self._fps_batch_sizes = []
        self._fps_system = None   # set by _finish_fps(), used by _pivot_and_save()

        # Raw per-unit latency samples, shipped to the server at shutdown and
        # pooled there (04 §1). Devices never pre-reduce: the mean of two
        # devices' p95 values is not the p95 of their combined population.
        #   pipeline = in-stage residency, everything from the unit entering
        #              this stage to it leaving. Contains `service`, which is
        #              only the get input -> output span.
        #   e2e      = edge start -> cloud output, reported by the cloud alone.
        self._pipeline_ms = []
        self._e2e_ms = []

        # Optional measurements. Every one of them is off until the server says
        # otherwise in the dispatch message — this device reads no measurement
        # setting from its own config file (README invariant 9), so a stale
        # config on one machine can never make a run mix two configurations.
        self._measure = {}
        self._ft = FreeTimeTracker(enabled=False)
        self._msg_size = MessageSizeRecorder(enabled=False)
        self._preds = PredictionWriter(enabled=False)

        self.channel.queue_declare(queue=INTERMEDIATE_QUEUE, durable=False)
        self.channel.queue_declare(queue=UTILIZATION_QUEUE, durable=False)

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

        ft = self._ft
        with open(self._timing_log_edge, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)
        # Opened where the timing log opens, so free time's span_s and
        # utilization's total_s describe the same stretch of this device's life.
        ft.start()

        batch_open_ns = None      # when this unit's first frame entered the stage

        while True:
            # Frame capture is real work that happens BEFORE `get input`, so
            # utilization cannot see it and free time must. This is the whole
            # reason free time is not 1 - utilization on this project's edge.
            t_lane = ft.now()
            ret, frame = cap.read()
            if not ret:
                ft.add_work('capture', t_lane)
                break
            if not buf:
                # The unit starts existing here. Frame decode + resize for the
                # whole batch sits between this and `get input`, so it is inside
                # `pipeline` and outside `service` — that gap is exactly the
                # buffering this stage adds.
                batch_open_ns = time.time_ns()
            frame = cv2.resize(frame, (imgsz, imgsz))
            t = torch.from_numpy(frame[:, :, ::-1].copy()).float() / 255.0
            buf.append(t.permute(2, 0, 1))
            ft.add_work('capture', t_lane)

            if len(buf) < batch_size:
                continue

            with open(self._timing_log_edge, "a") as _tf:
                print(str(time.time_ns()) + " get input", file=_tf)

            t_lane = ft.now()
            imgs = torch.stack(buf).to(self.device)
            buf  = []
            ft.add_work('tensor', t_lane)

            edge_start_wall = time.time()
            t0 = time.perf_counter()

            t_lane = ft.now()
            payload = model.forward_edge(imgs, split_point)
            ft.add_work('inference', t_lane)
            payload['edge_start_time'] = edge_start_wall
            # The unit's position in the SOURCE, so the cloud can name the
            # frames it predicts. A cloud takes an arbitrary subset off a shared
            # queue, so its own counter would name every frame wrongly.
            payload['batch_id'] = batch_id
            t_lane = ft.now()
            msg = pickle.dumps({'action': 'FEATURES', 'data': payload})
            msg_bytes = len(msg)
            ft.add_work('serialize', t_lane)

            # Recorded BEFORE the publish call: a broker at its high-water mark
            # stops accepting, and those are exactly the runs this measurement
            # exists to explain (12 §2). The size is the serialized byte count
            # handed to the transport, not a pre-serialization tensor size.
            self._msg_size.record(msg_bytes, batch_id)
            t_lane = ft.now()
            self.channel.basic_publish(exchange='', routing_key=INTERMEDIATE_QUEUE, body=msg)
            ft.add_work('send', t_lane)

            out_ns = time.time_ns()
            with open(self._timing_log_edge, "a") as _tf:
                print(str(out_ns) + " output", file=_tf)
            if batch_open_ns is not None:
                self._pipeline_ms.append((out_ns - batch_open_ns) / 1e6)
                batch_open_ns = None

            t_lane = ft.now()
            latency_ms = (time.perf_counter() - t0) * 1000
            fps_val    = batch_size / (time.perf_counter() - prev_end) if prev_end else 0.0
            ram_mb     = proc.memory_info().rss / 1e6

            self._write_metrics(log_path, split_point, 'edge', batch_id, batch_size,
                                latency_ms, fps_val, ram_mb, msg_bytes, 0.0, edge_start_wall)
            ft.add_work('metrics', t_lane)

            batch_id += 1
            prev_end  = time.perf_counter()
            pbar.update(batch_size)

        # Flush remaining. Marked up like a full batch so the last, partial unit
        # lands in busy_s and in the service samples the same way every other
        # one does — the cloud counts its completion regardless.
        if buf:
            with open(self._timing_log_edge, "a") as _tf:
                print(str(time.time_ns()) + " get input", file=_tf)
            t_lane = ft.now()
            imgs    = torch.stack(buf).to(self.device)
            payload = model.forward_edge(imgs, split_point)
            payload['edge_start_time'] = time.time()
            payload['batch_id'] = batch_id
            msg = pickle.dumps({'action': 'FEATURES', 'data': payload})
            ft.add_work('inference', t_lane)
            self._msg_size.record(len(msg), batch_id)
            t_lane = ft.now()
            self.channel.basic_publish(exchange='', routing_key=INTERMEDIATE_QUEUE, body=msg)
            ft.add_work('send', t_lane)
            out_ns = time.time_ns()
            with open(self._timing_log_edge, "a") as _tf:
                print(str(out_ns) + " output", file=_tf)
            if batch_open_ns is not None:
                self._pipeline_ms.append((out_ns - batch_open_ns) / 1e6)

        with open(self._timing_log_edge, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)
        ft.stop()

        total_batches_sent = batch_id + (1 if buf else 0)
        cap.release()
        pbar.close()
        print(f"[Edge] Done. {total_batches_sent} batches sent.")

        # Measurement goes out on its own queue, before the control message, so
        # it is already sitting on the broker when the server drains at shutdown.
        self._send_utilization(self._compute_utilization(self._timing_log_edge, 'edge'))
        self._send_free_time()
        self._send_message_size()

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

        # Control message only — measurement already went out above.
        self.channel.basic_publish(
            exchange='', routing_key='rpc_queue',
            body=pickle.dumps({'action': 'NOTIFY',
                               'client_id': self.client_id,
                               'layer_id': self.layer_id,
                               'batch_count': total_batches_sent}))

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

        # System FPS tracking (fps_guide §10): total_frames / total_time, not mean of 1/Δt
        self._fps_start_t = None
        self._fps_times = []
        self._fps_batch_sizes = []
        self._fps_system = None

        self.channel.queue_declare(queue='fps_queue', durable=False)

        ft = self._ft
        with open(self._timing_log_cloud, "w") as _tf:
            print(str(time.time_ns()) + " start", file=_tf)
        ft.start()
        map_conf = float(self._measure.get('map_conf', 0.001))

        while True:
            # Only classifiable after it returns: an empty get is free time, a
            # get that yields a unit is work. Take the timestamp before the
            # call and label the span after it, rather than guessing (10 §2).
            t_lane = ft.now()
            method, _, body = self.channel.basic_get(
                queue=INTERMEDIATE_QUEUE, auto_ack=True)

            if method and body:
                msg = pickle.loads(body)
                ft.add_work('recv', t_lane)

                # Sentinel = server signals all edge batches are in the queue
                if msg.get('action') == 'SENTINEL':
                    print(f"[Cloud] SENTINEL received — {batch_id} batches processed.")
                    break

                if self._fps_start_t is None:
                    self._fps_start_t = time.time()

                # The unit enters this stage the moment it is dequeued.
                deq_ns = time.time_ns()
                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(deq_ns) + " get input", file=_tf)

                recv_size = len(body)
                payload = msg['data']
                edge_start_time = payload.pop('edge_start_time', time.time())
                src_batch_id = payload.pop('batch_id', None)

                t0 = time.perf_counter()
                t_lane = ft.now()
                out  = model.forward_cloud(
                    {'x_bit': payload['x_bit'], 'mu': payload['mu'],
                     'sigma': payload['sigma'], 'split': payload['split']},
                    device=str(self.device)
                )
                ft.add_work('inference', t_lane)
                pred = out[0] if isinstance(out, tuple) else out
                t_lane = ft.now()
                dets = non_max_suppression(pred, conf, iou)
                ft.add_work('postprocess', t_lane)

                if self._preds.enabled:
                    # The accuracy path, and everything about it is inside the
                    # get input -> output window on purpose: it is work this run
                    # really did, so it belongs in busy_s, service, pipeline and
                    # e2e. That is also why a run with this on is not comparable
                    # with one without it.
                    t_lane = ft.now()
                    low = (dets if abs(map_conf - conf) < 1e-12
                           else non_max_suppression(pred, map_conf, iou))
                    self._preds.write_batch(low, src_batch_id, batch_size)
                    ft.add_work('map', t_lane)

                with open(self._timing_log_cloud, "a") as _tf:
                    print(str(time.time_ns()) + " output", file=_tf)

                latency_ms   = (time.perf_counter() - t0) * 1000
                cloud_end_ns = time.time_ns()
                cloud_end    = cloud_end_ns / 1e9
                e2e_ms       = (cloud_end - edge_start_time) * 1000
                fps_val      = batch_size / (time.perf_counter() - prev_end) if prev_end else 0.0
                ram_mb       = proc.memory_info().rss / 1e6
                bs           = pred.shape[0]

                self._fps_times.append(cloud_end)
                self._fps_batch_sizes.append(bs)
                # Two machines by definition, so this inherits any offset
                # between their clocks — report it, but treat it as indicative.
                self._e2e_ms.append(e2e_ms)

                # Exactly one stage publishes per unit, and for a head+tail
                # split that is the tail. Two publishers would double the
                # measured throughput. The body is an identity — the producing
                # cluster — with no timestamp and no unit id in it.
                t_lane = ft.now()
                self.channel.basic_publish(
                    exchange='', routing_key='fps_queue',
                    body=CLUSTER_ID.encode())
                ft.add_work('send', t_lane)
                self._pipeline_ms.append((time.time_ns() - deq_ns) / 1e6)

                t_lane = ft.now()
                self._write_metrics(log_path, split_point, 'cloud', batch_id, bs,
                                    latency_ms, fps_val, ram_mb, recv_size,
                                    e2e_ms, edge_start_time)
                ft.add_work('metrics', t_lane)

                batch_id += 1
                prev_end  = time.perf_counter()
                pbar.update(bs)

            else:
                time.sleep(0.1)
                # The empty get and the sleep that follows it are one idle
                # stretch, and the reason is starvation: this cloud has nothing
                # to work on because no edge has published yet.
                ft.add_wait('input', t_lane)

        self._finish_fps()

        with open(self._timing_log_cloud, "a") as _tf:
            print(str(time.time_ns()) + " end", file=_tf)
        ft.stop()

        # Measurement first, on its own queue, then the control message.
        self._send_utilization(self._compute_utilization(self._timing_log_cloud, 'cloud'))
        self._send_free_time()
        self._send_predictions()

        self.channel.basic_publish(
            exchange='', routing_key='rpc_queue',
            body=pickle.dumps({'action': 'CLOUD_DONE',
                               'client_id': self.client_id,
                               'batches': batch_id}))

        pbar.close()
        self._pivot_and_save(log_path)

    # ---------------------------------------------------------------------- #
    # FPS summary (fps_guide §10 — total_frames / total_time, not mean of 1/Δt)
    # ---------------------------------------------------------------------- #
    def _finish_fps(self):
        """This cloud's own view of throughput — a local reference, not the result.

        The authoritative system number is the server's: it counts every
        arrival from every cloud against the shared START recorded when work was
        dispatched. This one starts at *this* device's first arrival, so it
        excludes pipeline fill-up and knows nothing about the other clouds.
        Labelled so the two can never be confused in a console scrollback.
        """
        t = self._fps_times
        n = len(t)
        total_frames = sum(self._fps_batch_sizes)
        print("=" * 60)
        if n >= 1 and self._fps_start_t is not None and t[-1] > self._fps_start_t:
            total_time = t[-1] - self._fps_start_t
            system_fps = total_frames / total_time
            self._fps_system = system_fps
            print(f"  [cloud-local FPS] {system_fps:8.3f} fps   "
                  f"= {total_frames} frames / {total_time:.2f}s  (first arrival -> last DONE)")
            if n >= 2 and t[-1] > t[0]:
                span = t[-1] - t[0]
                steady_frames = total_frames - self._fps_batch_sizes[0]
                if span > 0 and steady_frames > 0:
                    steady = steady_frames / span
                    print(f"  [steady-state]    {steady:8.3f} fps   "
                          f"= {steady_frames} / {span:.2f}s  (first -> last DONE)")
            if n >= 2:
                gaps = [t[i] - t[i - 1] for i in range(1, n) if t[i] > t[i - 1]]
                if gaps:
                    ref_mean = sum(self._fps_batch_sizes[i] / g
                                   for i, g in enumerate(gaps, 1)) / len(gaps)
                    print(f"  [ref mean, N/U]   {ref_mean:8.3f} fps   "
                          f"(arithmetic mean of 1/dt — reference only, biased high)")
        else:
            print("  [cloud-local FPS] no batches received — nothing to report")
        print(f"  batches counted: {n}   (server holds the authoritative total)")
        print("=" * 60)

    # ---------------------------------------------------------------------- #
    # Utilization (utilization_guide.md)
    # ---------------------------------------------------------------------- #
    def _compute_utilization(self, tlog_path, role):
        """One ratio for the whole run, computed after it, from this device's log.

        Numerator and denominator both come from this device's own clock, so
        clock skew between machines cannot distort the ratio. The per-unit
        intervals are summed *before* dividing, never averaged.

        The same intervals are returned as `service_ms`, which is what makes
        `Σ service == busy_s` true by construction rather than by luck (04 §2.1).
        """
        t_start = t_end = t_input = None
        busy_ns = 0
        n_packages = 0
        service_ms = []
        try:
            with open(tlog_path) as f:
                for line in f:
                    # Split ONCE: event names contain spaces ("get input").
                    parts = line.strip().split(" ", 1)
                    if len(parts) < 2 or not parts[0].isdigit():
                        continue
                    ts, event = int(parts[0]), parts[1]
                    if event == 'start':
                        t_start = ts
                    elif event == 'end':
                        t_end = ts
                    elif event == 'get input':
                        t_input = ts
                    elif event == 'output':
                        # An unmatched 'get input' — a crash mid-unit — is
                        # dropped rather than counted as an infinite interval.
                        if t_input is not None:
                            busy_ns += ts - t_input
                            service_ms.append((ts - t_input) / 1e6)
                            n_packages += 1
                            t_input = None
                    # Any other event is ignored by design, so new markers can
                    # be added without breaking this parser.
        except Exception as e:
            print(f"[Utilization][{role}] Warning: could not read log: {e}")
            return None
        if t_start is None or t_end is None or t_end <= t_start:
            print(f"[Utilization][{role}] Warning: incomplete timing log")
            return None
        total_ns = t_end - t_start
        util = busy_ns / total_ns
        print(f"[Utilization][{role}] packages={n_packages} "
              f"busy={busy_ns/1e9:.3f}s total={total_ns/1e9:.3f}s "
              f"utilization={util*100:.2f}%")
        return {"role": role, "packages": n_packages,
                "busy_ns": busy_ns, "total_ns": total_ns, "utilization": util,
                "service_ms": service_ms}

    def _send_utilization(self, stats):
        """Ship this device's report — ratio plus raw latency arrays — at shutdown.

        Goes to a dedicated queue, not the control queue: the server stops
        reading control the moment the last edge reports done, but the clouds
        finish later, so a report sent there would never be consumed. On its own
        queue the report simply waits on the broker. Publisher and consumer
        never need to be alive at the same moment.
        """
        if stats is None:
            print(f"[Utilization] Skipped (stats=None) client={str(self.client_id)[:8]}")
            return
        try:
            self.channel.queue_declare(queue=UTILIZATION_QUEUE, durable=False)
            self.channel.basic_publish(
                exchange='', routing_key=UTILIZATION_QUEUE,
                body=pickle.dumps({
                    "action": "UTILIZATION",
                    "client_id": str(self.client_id),
                    "layer_id": self.layer_id,
                    "cluster": CLUSTER_ID,
                    **stats,
                    # Raw samples, never pre-reduced percentiles — the server
                    # pools across devices before it takes any percentile.
                    "pipeline_ms": list(self._pipeline_ms),
                    "e2e_ms": list(self._e2e_ms),
                }))
            print(f"[Utilization] Sent  role={stats['role']:5s}  "
                  f"client={str(self.client_id)[:8]}  "
                  f"util={stats['utilization']*100:.2f}%  "
                  f"samples: service={len(stats.get('service_ms') or [])} "
                  f"pipeline={len(self._pipeline_ms)} e2e={len(self._e2e_ms)}")
        except Exception as e:
            # Telemetry never kills the run: a broken metric loses a number.
            print(f"[Utilization] Warning: could not send: {e}")

    # ---------------------------------------------------------------------- #
    # Optional measurements — each ships on its own queue, at finish
    # ---------------------------------------------------------------------- #
    def _publish_report(self, queue, payload, label):
        """One report onto one dedicated measurement queue.

        Same reasoning as the utilization queue: the server drains these at
        shutdown, long after this device has exited, so the report waits on the
        broker instead of needing both ends alive at once. Every failure path
        here is a warning — telemetry never kills the run.
        """
        try:
            self.channel.queue_declare(queue=queue, durable=False)
            self.channel.basic_publish(exchange='', routing_key=queue,
                                       body=pickle.dumps(payload))
            return True
        except Exception as e:
            print(f"[{label}] Warning: could not send report: {e}")
            return False

    def _send_free_time(self):
        """This device's idle time — the report, plus its own local copy.

        A disabled tracker returns no report at all rather than a report full of
        zeros: "we did not measure" and "this device was never idle" are
        different statements and must not be written the same way.
        """
        report = self._ft.report()
        if report is None:
            return
        self._ft.write_local(report)
        report['layer_id'] = self.layer_id
        if self._publish_report(FREETIME_QUEUE, report, 'FreeTime'):
            print(f"[FreeTime] Sent  role={report['role']:5s} "
                  f"span={report['span_ns'] / 1e9:.3f}s "
                  f"busy={report['busy_ns'] / 1e9:.3f}s "
                  f"free={report['free_ns'] / 1e9:.3f}s "
                  f"({report['free_ns'] / report['span_ns'] * 100:.2f}%, "
                  f"{report['gaps']} gap(s))")

    def _send_message_size(self):
        """The elected edge's egress. Every other worker returns None here."""
        report = self._msg_size.report()
        if report is None:
            return
        if self._publish_report(MSGSIZE_QUEUE, report, 'MessageSize'):
            print(f"[MessageSize] Sent  n={report['n']} "
                  f"total_mb={report['total_bytes'] / 1e6:.3f} "
                  f"mean_mb={report['mean_bytes'] / 1e6:.3f}")

    def _send_predictions(self):
        """Ship raw per-frame predictions; the SERVER scores them.

        mAP is a ranking metric over a pooled detection set — the mean of two
        devices' scores is not the score of their combined detections, and no
        weighted combination reconstructs it. So nothing is reduced here. The
        transfer happens at shutdown, entirely outside the measured window.
        """
        if not self._preds.enabled and not self._preds.written:
            return
        payload = self._preds.pack()
        if payload is None:
            return
        if self._publish_report(MAP_QUEUE, {
            'action': 'MAP_PRED',
            'client_id': str(self.client_id),
            'cluster': CLUSTER_ID,
            'payload': payload,
        }, 'mAP'):
            print(f"[mAP] Sent  {self._preds.frames} prediction file(s) "
                  f"({len(payload) / 1e3:.1f} kB zipped, "
                  f"{self._preds.skipped} frame(s) already written by another device)")

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

        # Use system FPS (total_frames / total_time) from _finish_fps if available,
        # otherwise fall back to _avg (biased high, kept only as reference).
        cloud_fps_display = (round(self._fps_system, 3)
                             if self._fps_system is not None
                             else _avg(cloud_rows, 'fps'))

        print('=' * 50)
        print(f"  DMSF SUMMARY  |  batches={len(pairs)}")
        print('=' * 50)
        print(f"  [EDGE]  latency={_avg(edge_rows,'latency_ms')} ms  fps={_avg(edge_rows,'fps')}  ram={_avg(edge_rows,'ram_mb')} MB")
        print(f"  [CLOUD] latency={_avg(cloud_rows,'latency_ms')} ms  fps={cloud_fps_display}  ram={_avg(cloud_rows,'ram_mb')} MB")
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
                       conf, iou, log_path, num_layers, measure=None,
                       imgsz=640):
        """Entry point called by DmsfRpcClient once START arrives.

        ``measure`` is the server's measurement block, straight out of the
        dispatch message. Everything optional this device does is switched on
        from there and from nowhere else, so turning a feature off on the server
        turns it off on every worker in the same run (README invariant 9).
        """
        self._measure = dict(measure or {})
        self._ft = FreeTimeTracker(
            enabled=self._measure.get('free_time', False),
            cluster=CLUSTER_ID, client_id=self.client_id,
            role='edge' if self.layer_id == 1 else 'cloud',
            device=self.device,
            bucket_s=self._measure.get('free_time_bucket_s', 1.0),
            log_path=log_path)
        if self.layer_id == 1:
            self._msg_size = MessageSizeRecorder(
                enabled=self._measure.get('message_size', False),
                cluster=CLUSTER_ID, client_id=self.client_id, role='edge',
                context={'mode': 'split', 'split_point': split_point,
                         'compress': 'on', 'num_bit': 1,
                         'machine': self._id_tag},
                log_path=log_path)
            self.first_layer(model, split_point, data, batch_size, log_path)
        elif self.layer_id == num_layers:
            self._preds = PredictionWriter(
                enabled=self._measure.get('map', False),
                cluster=CLUSTER_ID, imgsz=imgsz, log_path=log_path)
            self.last_layer(model, split_point, batch_size, conf, iou, log_path)