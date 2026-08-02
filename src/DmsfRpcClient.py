import pickle
import time
import torch

from models.dmsf_26n import DMSF26n


class DmsfRpcClient:
    def __init__(self, client_id, layer_id, channel, inference_func, device):
        self.client_id = client_id
        self.layer_id  = layer_id
        self.channel   = channel
        self.inference_func = inference_func
        self.device    = device

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(
            exchange='', routing_key='rpc_queue', body=pickle.dumps(message))

    def wait_response(self):
        reply_q = f"reply_{self.client_id}"
        self.channel.queue_declare(reply_q, durable=False)
        running = True
        while running:
            method, _, body = self.channel.basic_get(queue=reply_q, auto_ack=True)
            if body:
                running = self._handle(pickle.loads(body))
            time.sleep(0.5)

    def _handle(self, msg):
        action = msg['action']
        print(f"[Client] Received: {action}")

        if action == 'START':
            split_point = msg['split_point']
            nc          = msg['nc']
            weights     = msg['weights']
            batch_size  = msg['batch_size']
            data        = msg['data']
            conf        = msg['conf']
            iou         = msg['iou']
            log_path    = msg.get('log_path', '.')
            num_layers  = msg['num_layers']

            dev = torch.device(
                msg['device_edge'] if self.layer_id == 1 else msg['device_cloud'])

            model = DMSF26n(nc=nc).to(dev)
            ckpt  = torch.load(weights, map_location=dev, weights_only=False)
            state = ckpt.get('model', ckpt)
            model.load_state_dict(
                state if isinstance(state, dict) else state.state_dict(), strict=False)
            model.eval()
            print(f"[Client] Model loaded: {weights}  split_point={split_point}")

            self.inference_func(
                model=model,
                split_point=split_point,
                data=data,
                batch_size=batch_size,
                conf=conf,
                iou=iou,
                log_path=log_path,
                num_layers=num_layers,
            )
            return False

        elif action == 'STOP':
            print("[Client] Received STOP.")
            return False

        return True
