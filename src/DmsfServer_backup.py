import pickle
import sys
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
        self.reply_channel = self.connection.channel()

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
            self.count_notify += 1
            if self.count_notify == self.total_clients[0]:
                print("[Server] All edges done. Sending STOP.")
                self.notify_clients(start=False)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.channel.stop_consuming()
                return

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
        self.connection.close()
        sys.exit(0)
