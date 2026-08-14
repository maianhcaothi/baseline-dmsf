"""Message size — implements ``guide/12-message-size.md`` for the DMSF baseline.

Every other measurement here describes *time*. This one describes *bytes*: how
large the intermediate feature map an edge hands to the broker is, recorded on
the edge, once per published message.

It is what makes three other numbers readable. Utilization says an edge was
busy; this says whether it was busy computing or busy shipping. The broker's
memory curve (`guide/11`) shows the queue filling; ``mean_mb x queue depth``
says whether that is the payload or something else. And for DMSF specifically
it is the quantity the split point is chosen *for* — the whole point of cutting
at layer N is the size of what crosses the wire.

Three rules, each of which has its own failure mode:

* **Exactly one worker measures, and the SERVER picks it** — the first edge to
  register, told so in its dispatch message. Never self-selected: one stale
  config file and either every edge measures or none does, and the summary line
  looks identical either way (`guide/12 §1`). Every edge in a cluster publishes
  the same payload shape from the same split point, so nine measuring workers
  produce one number nine times at nine times the cost.
* **The size is recorded BEFORE the publish call.** A broker at its high-water
  mark stops accepting and a saturated link stalls mid-write — exactly the runs
  this measurement exists to explain. Measuring after the call writes the
  stalled message's sample late, or never (`guide/12 §2`).
* **Serialized bytes**, the value handed to the transport, after our own
  pickling and compression and before the transport's own framing. A
  pre-serialization tensor size is a different quantity.
"""

import math
import os
import time


#: A long run must not turn the report into a multi-megabyte message. The series
#: is decimated EVENLY (never truncated) so it still spans the whole run; the
#: statistics are always computed over the full sample set.
MAX_SHIPPED_SAMPLES = 3000


def nearest_rank(sorted_values, q):
    if not sorted_values:
        return None
    k = max(1, math.ceil(q / 100.0 * len(sorted_values)))
    return sorted_values[k - 1]


class MessageSizeRecorder:
    """Records one edge's egress. Disabled instances cost one attribute lookup.

    On every worker that was not elected the flag arrives ``False`` and
    :meth:`record` returns immediately, which is why the call can sit
    unconditionally in the publish path.
    """

    def __init__(self, enabled=False, cluster="unknown", client_id="unknown",
                 role="edge", context=None, log_path="."):
        self.enabled = bool(enabled)
        self.cluster = cluster
        self.client_id = str(client_id)
        self.role = role
        self.context = dict(context or {})
        self.log_path = log_path
        self.sizes = []
        self.batch_ids = []
        self.offsets_s = []
        self._t0 = None
        self._warned = False
        self._file = None
        if self.enabled:
            tag = f"{_safe(cluster)}_{self.client_id.replace('-', '')[:12]}"
            self._file = os.path.join(log_path, f"message_size_{tag}.log")
            try:
                open(self._file, "w").close()
            except Exception as e:
                self._disable(e)

    def record(self, n_bytes, batch_id=None):
        """One published message. Call it **before** ``basic_publish``."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        offset = now - self._t0
        self.sizes.append(int(n_bytes))
        self.batch_ids.append(batch_id)
        self.offsets_s.append(offset)
        # Local copy, appended live: the worker's own record, and the only one
        # that survives a broker or server problem (guide/12 §3).
        try:
            with open(self._file, "a") as f:
                f.write(f"{time.time_ns()} i={len(self.sizes) - 1} "
                        f"t_offset_s={offset:.3f} "
                        f"batch_id={batch_id if batch_id is not None else -1} "
                        f"bytes={int(n_bytes)} mb={int(n_bytes) / 1e6:.3f}\n")
        except Exception as e:
            self._disable(e)

    def _disable(self, err):
        """One warning, then stop — never one warning per message."""
        if not self._warned:
            print(f"[MessageSize] Warning: recording failed ({err}); "
                  f"recorder disabled for the rest of the run")
            self._warned = True
        self.enabled = False

    def report(self):
        """Statistics over **all** samples, plus an evenly decimated series.

        Decimation may coarsen a plot; it must never coarsen a number, so the
        summary is computed here from ``self.sizes`` in full and only the
        shipped series is thinned (`guide/12 §3`).
        """
        if not self.sizes:
            return None
        ordered = sorted(self.sizes)
        n = len(self.sizes)
        span_s = (self.offsets_s[-1] - self.offsets_s[0]) if n > 1 else 0.0
        step = max(1, math.ceil(n / MAX_SHIPPED_SAMPLES))
        series = [(i, self.offsets_s[i], self.batch_ids[i], self.sizes[i])
                  for i in range(0, n, step)]
        return {
            "action": "MESSAGE_SIZE",
            "client_id": self.client_id,
            "role": self.role,
            "cluster": self.cluster,
            "context": self.context,
            "n": n,
            "total_bytes": sum(self.sizes),
            "mean_bytes": sum(self.sizes) / n,
            "p50_bytes": nearest_rank(ordered, 50),
            "p95_bytes": nearest_rank(ordered, 95),
            "max_bytes": ordered[-1],
            "min_bytes": ordered[0],
            "span_s": span_s,
            "series": series,
        }


def _safe(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))
