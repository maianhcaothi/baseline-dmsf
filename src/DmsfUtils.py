import pika


def delete_old_queues(host, username='guest', password='guest', virtual_host='/'):
    try:
        credentials = pika.PlainCredentials(username, password)
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=host, credentials=credentials, virtual_host=virtual_host))
        ch = conn.channel()
        # The measurement queues go too: a crashed run's stale completions and
        # device reports would otherwise be counted into the next run, and the
        # inflation is invisible in the charts because it looks like a fast start.
        for q in ['rpc_queue', 'dmsf_intermediate_queue',
                  'fps_queue', 'utilization_queue',
                  'freetime_queue', 'msgsize_queue', 'map_queue']:
            try:
                ch.queue_delete(queue=q)
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
