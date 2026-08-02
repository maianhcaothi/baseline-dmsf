import pika


def delete_old_queues(host, username='guest', password='guest', virtual_host='/'):
    try:
        credentials = pika.PlainCredentials(username, password)
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=host, credentials=credentials, virtual_host=virtual_host))
        ch = conn.channel()
        for q in ['rpc_queue', 'dmsf_intermediate_queue']:
            try:
                ch.queue_delete(queue=q)
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
