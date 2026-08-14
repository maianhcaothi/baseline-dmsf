"""Infrastructure-host RAM — implements ``guide/11-broker-ram.md``.

Every other measurement in this project is reported by a process we wrote. This
one is not: between the edges and the clouds sits the machine running RabbitMQ,
and every intermediate feature map crosses it and is buffered in its memory. It
is on the critical path, so it can be the bottleneck, and when it is **every
symptom shows up somewhere else** — a broker at its high-water mark does not
fail, it *blocks publishers*, which on a device looks like a stall with no local
cause. "The next stage is slow" and "the broker stopped accepting" produce
almost identical device-side telemetry; this curve is what separates them.

Two sources, and the difference between them is not cosmetic:

``ssh``
    Host memory from ``/proc/meminfo`` over **one long-lived session** running a
    bounded remote loop — never one connection per sample, which would make the
    machine whose load we are measuring pay a TCP handshake plus an
    authentication every second. ``used = MemTotal - MemAvailable``; using
    ``MemFree`` instead counts reclaimable page cache as used and reads ~90% on
    any machine that has touched a disk.

``rabbitmq_api``
    The management API, which reports the **broker process**, not the host. On
    this deployment the broker is on loopback, so it shares a machine with the
    server and every worker — host memory there would be dominated by our own
    processes and "what running the system cost this host" would be a false
    attribution. The broker process is the honest number, and ``total_mb`` is
    then its high-water *limit*: the level at which publishers actually get
    blocked, which is the wall worth measuring the distance to.

Which one produced a number is never left implicit: every line carries
``source=``, and a fallback is labelled, never silently substituted. An
unlabelled fallback is worse than no fallback — it is a plausible number that
means something other than the file name says.
"""

import base64
import json
import threading
import time
import urllib.request

LOOPBACK = ("127.0.0.1", "localhost", "::1", "")

#: Bound the remote loop so a sampler orphaned by a hard kill of the server
#: cannot outlive the run on someone else's machine (guide/11 §2).
DEFAULT_MAX_MINUTES = 180

_REMOTE_LOOP = r"""
i=0
while [ $i -lt {max_samples} ]; do
  i=$((i+1))
  r=`ps -eo rss=,comm= | awk '$2 ~ /beam|rabbit/ {{s+=$1}} END{{print s+0}}'`
  awk -v ts="`date +%s%N`" -v r="$r" '
    /^MemTotal:/{{t=$2}} /^MemFree:/{{f=$2}} /^MemAvailable:/{{a=$2}}
    /^Buffers:/{{b=$2}} /^Cached:/{{c=$2}}
    /^SwapTotal:/{{st=$2}} /^SwapFree:/{{sf=$2}}
    END{{print "RAM", ts, t, f, a, b, c, st, sf, r}}' /proc/meminfo
  sleep {interval}
done
"""


class BrokerRamSampler:
    """Samples the queue host for the whole life of the server.

    The window opens in the server's constructor — before any worker registers
    or anything is published — and closes a second or two after the drain. That
    leading ``idle`` stretch is the entire point: without the host at rest there
    is no denominator for "what did running the system cost it", and a meter
    that only runs while the system runs can report a level but never a cost.

    Marks *partition* the series; they never gate it. Sampling does not pause at
    a boundary, so a missing mark coarsens the split rather than leaving a gap,
    and a run that never dispatched is all ``idle`` — which is exactly what it was.
    """

    def __init__(self, config, rabbit_cfg, series_path, log=print):
        cfg = (config or {}).get("broker_ram") or {}
        self.enabled = bool(cfg.get("enable", False))
        self.host = str(cfg.get("host") or rabbit_cfg.get("address", "127.0.0.1"))
        self.user = cfg.get("user") or ""
        self.password = cfg.get("password") or ""
        self.ssh_port = int(cfg.get("ssh_port", 22))
        self.api_port = int(cfg.get("api_port", 15672))
        self.api_user = rabbit_cfg.get("username", "guest")
        self.api_password = rabbit_cfg.get("password", "guest")
        self.interval_s = max(float(cfg.get("interval_s", 1.0)), 0.1)
        self.tail_s = float(cfg.get("tail_s", 2.0))
        self.max_minutes = float(cfg.get("max_minutes", DEFAULT_MAX_MINUTES))
        self.series_path = series_path
        self.log = log

        self.samples = []
        self.source = None
        self.reason = ""
        self.phase = "idle"
        self.marks = {}
        self._stop = threading.Event()
        self._thread = None
        self._stderr_tail = []
        self._client = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="broker-ram",
                                        daemon=True)
        self._thread.start()

    def mark(self, phase):
        """Move the phase forward. Marks only ever move forward, so a line
        written before dispatch is ``idle`` and stays ``idle``."""
        if not self.enabled or phase == self.phase:
            return
        self.phase = phase
        self.marks[phase] = time.time_ns()

    def stop(self):
        """Close the window a couple of seconds past the drain.

        Not at the last collector: that is not the system being idle, it is the
        busiest moment of the shutdown. The drain is precisely when a backed-up
        host gives memory back, so a curve that does *not* fall there is the
        signal that something is still holding units.
        """
        if not self.enabled:
            return
        self.mark("tail")
        if self.tail_s > 0:
            time.sleep(self.tail_s)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.interval_s * 2))
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass

    # -- sampling ----------------------------------------------------------- #
    def _run(self):
        """Telemetry never kills the run: every failure here degrades to a
        warning and, at worst, a ``samples=0`` line naming the reason."""
        try:
            if self._should_try_ssh() and self._run_ssh():
                return
            self._run_api()
        except Exception as e:                                # pragma: no cover
            self.reason = f"sampler crashed: {e}"
            self.log(f"[BrokerRAM] Warning: {self.reason}")

    def _should_try_ssh(self):
        return bool(self.user and self.password and self.host not in LOOPBACK)

    def _run_ssh(self):
        try:
            import paramiko
        except Exception as e:
            self.reason = f"paramiko unavailable ({e})"
            self.log(f"[BrokerRAM] {self.reason} - falling back to the management API")
            return False
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.host, port=self.ssh_port, username=self.user,
                           password=self.password, timeout=10,
                           allow_agent=False, look_for_keys=False)
            self._client = client
            script = _REMOTE_LOOP.format(
                max_samples=int(self.max_minutes * 60 / self.interval_s),
                interval=self.interval_s)
            # The script crosses a local argv AND a remote login shell; base64
            # removes both quoting layers at once (guide/11 §2).
            payload = base64.b64encode(script.encode()).decode()
            _stdin, stdout, stderr = client.exec_command(
                f"echo {payload} | base64 -d | sh", timeout=15)
            threading.Thread(target=self._drain_stderr, args=(stderr,),
                             daemon=True).start()
        except Exception as e:
            # Host credentials are not the AMQP credentials; confusing the two
            # produces an auth failure that looks like a network problem.
            self.reason = f"ssh to {self.host} failed: {e}"
            self.log(f"[BrokerRAM] {self.reason} - falling back to the management API")
            return False

        self.source = "ssh"
        self.log(f"[BrokerRAM] sampling {self.host} over one ssh session "
                 f"every {self.interval_s:.1f}s")
        got = False
        for line in iter(stdout.readline, ""):
            if self._stop.is_set():
                break
            parts = line.split()
            if len(parts) != 10 or parts[0] != "RAM":
                continue
            try:
                _, _remote_ns, total, free, avail, _buf, cached, swap_t, swap_f, rss = parts
                total_kb, avail_kb = float(total), float(avail)
                sample = {
                    # The SERVER's clock, taken when the line is read — a shared
                    # result file never carries another machine's timestamp
                    # (guide/README invariant 1).
                    "ts_ns": time.time_ns(),
                    "total_mb": total_kb / 1e3,
                    "used_mb": (total_kb - avail_kb) / 1e3,
                    "avail_mb": avail_kb / 1e3,
                    "free_mb": float(free) / 1e3,
                    "cached_mb": float(cached) / 1e3,
                    "swap_used_mb": (float(swap_t) - float(swap_f)) / 1e3,
                    "rss_mb": float(rss) / 1e3,
                }
            except ValueError:
                continue
            got = True
            self._emit(sample)
        if not got and not self._stop.is_set():
            self.reason = ("ssh session produced no samples: "
                           + " | ".join(self._stderr_tail[-3:]))
            self.log(f"[BrokerRAM] {self.reason}")
            self.source = None
            return False
        return True

    def _drain_stderr(self, stderr):
        try:
            for line in iter(stderr.readline, ""):
                line = line.strip()
                if line:
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-5]
        except Exception:
            pass

    def _run_api(self):
        url = f"http://{self.host}:{self.api_port}/api/nodes"
        token = base64.b64encode(
            f"{self.api_user}:{self.api_password}".encode()).decode()
        self.source = "rabbitmq_api"
        self.log(f"[BrokerRAM] sampling {self.host} over the management API "
                 f"every {self.interval_s:.1f}s "
                 f"(used_mb is the BROKER PROCESS, not the host)")
        deadline = time.time() + self.max_minutes * 60
        while not self._stop.is_set() and time.time() < deadline:
            try:
                request = urllib.request.Request(url)
                request.add_header("Authorization", f"Basic {token}")
                node = json.load(urllib.request.urlopen(request, timeout=5))[0]
                used_mb = float(node.get("mem_used", 0)) / 1e6
                limit_mb = float(node.get("mem_limit", 0)) / 1e6
                self._emit({
                    "ts_ns": time.time_ns(),
                    "total_mb": limit_mb,
                    "used_mb": used_mb,
                    "avail_mb": max(limit_mb - used_mb, 0.0),
                    "rss_mb": used_mb,
                })
            except Exception as e:
                if not self.reason:
                    self.reason = f"management API unreachable: {e}"
                    self.log(f"[BrokerRAM] Warning: {self.reason}")
            self._stop.wait(self.interval_s)

    def _emit(self, sample):
        """Append one sample to the live series and keep it for the summary.

        Written live, like ``batch_done_ns.log`` and for the same reason: a run
        that dies still leaves the series behind, and the series is the part
        that cannot be reconstructed.
        """
        sample["phase"] = self.phase
        sample["source"] = self.source
        self.samples.append(sample)
        line = (f"{sample['ts_ns']} host={self.host} source={self.source} "
                f"phase={sample['phase']} "
                f"total_mb={sample['total_mb']:.1f} "
                f"used_mb={sample['used_mb']:.1f} "
                f"used={_pct(sample['used_mb'], sample['total_mb']):.2f}% "
                f"avail_mb={sample['avail_mb']:.1f}")
        for key, fmt in (("free_mb", "free_mb={:.1f}"),
                         ("cached_mb", "cached_mb={:.1f}"),
                         ("swap_used_mb", "swap_used_mb={:.1f}")):
            if key in sample:
                line += " " + fmt.format(sample[key])
        line += f" rabbit_rss_mb={sample['rss_mb']:.1f}"
        try:
            with open(self.series_path, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            if not self.reason:
                self.reason = f"series write failed: {e}"
                self.log(f"[BrokerRAM] Warning: {self.reason}")

    # -- for the summary writer -------------------------------------------- #
    def meta(self):
        return {"host": self.host, "source": self.source or "none",
                "interval_s": self.interval_s, "reason": self.reason}


def _pct(part, whole):
    return (part / whole * 100.0) if whole else 0.0
