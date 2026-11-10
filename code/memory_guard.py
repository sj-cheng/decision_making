"""Cap how many jobs run at once when the whole server is short on memory.

Used memory matches the 已用 column of ``free``: total minus free minus buffers
and cache. File cache does not count. Above 220 GiB the cap steps down. Below
200 GiB it steps back up, never past the number the caller asked for.
"""
import time

LIMIT_GIB = 220
RELEASE_GIB = 200
FLOOR = 8


def used_gib():
    info = {}
    with open('/proc/meminfo') as handle:
        for line in handle:
            key, rest = line.split(':', 1)
            info[key] = int(rest.split()[0])
    used = (
        info['MemTotal']
        - info['MemFree']
        - info.get('Buffers', 0)
        - info.get('Cached', 0)
        - info.get('SReclaimable', 0)
    )
    return used / 1024 / 1024


class Inflight:
    """Remember the caller's worker count and the smaller cap memory allows."""

    def __init__(self, configured, limit_gib=LIMIT_GIB, release_gib=RELEASE_GIB, floor=FLOOR):
        if configured < 1:
            raise ValueError('configured workers must be positive')
        self.configured = int(configured)
        self.limit = self.configured
        self.limit_gib = float(limit_gib)
        self.release_gib = float(release_gib)
        self.floor = max(1, min(int(floor), self.configured))
        self._checked = 0.0

    def poll(self, now=None, used=None):
        """Return the cap. Re-read memory at most once every five seconds."""
        now = time.monotonic() if now is None else now
        if now - self._checked < 5 and used is None:
            return self.limit
        self._checked = now
        used = used_gib() if used is None else used
        previous = self.limit
        if used >= self.limit_gib and self.limit > self.floor:
            step = max(1, self.configured // 8)
            self.limit = max(self.floor, self.limit - step)
        elif used <= self.release_gib and self.limit < self.configured:
            step = max(1, self.configured // 16)
            self.limit = min(self.configured, self.limit + step)
        if self.limit != previous:
            print(
                f'memory guard: used {used:.1f} GiB, parallel {previous} -> {self.limit} (configured {self.configured})',
                flush=True,
            )
        return self.limit
