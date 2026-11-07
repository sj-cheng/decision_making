"""Training-only execution helpers; preserve the exact public filter outputs."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import numpy as np


def clone(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return value


class ExactArrayCache:
    """Bounded per-process memoization by every argument's exact bytes.

    Return independent array copies: estimators cannot mutate cached objects.
    No posterior, random sample, weight, observation or hidden state is cached.
    """
    def __init__(self, function, capacity=16):
        self.function = function
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, *args):
        key = tuple((value.dtype.str, value.shape, value.tobytes()) for value in args)
        if key in self.entries:
            self.hits += 1
            value = self.entries.pop(key)
            self.entries[key] = value
            return clone(value)
        self.misses += 1
        value = self.function(*args)
        self.entries[key] = clone(value)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value


_CACHES = {}


@contextmanager
def public_geometry_cache():
    """Only wrap a training rollout; validation/test remain fully uncached."""
    import l45_corner_shared_20261025.corner_estimator as estimator
    originals = {name: getattr(estimator, name) for name in ('make_graph', 'map_probabilities')}
    try:
        for name, function in originals.items():
            cache = _CACHES.setdefault(name, ExactArrayCache(function))
            if cache.function is not function:
                raise RuntimeError('Unexpected estimator wrapper nesting')
            setattr(estimator, name, cache)
        yield
    finally:
        for name, function in originals.items():
            setattr(estimator, name, function)


def cache_statistics():
    return {name: dict(hits=cache.hits, misses=cache.misses, entries=len(cache.entries))
            for name, cache in _CACHES.items()}


def parallel_collect(pool, jobs, entries, request_builder, rollout_function):
    """Split the same logical streams at complete-episode boundaries.

    Every episode has at most 80 transitions. Schedule only episodes guaranteed
    to be consumed by the original stream, then extend streams if necessary.
    Thus there are no speculative episodes, changed minibatches or dropped
    trajectories. Concatenation preserves the original RNG and episode order.
    """
    streams = [dict(rng=np.random.default_rng(job['seed']), parts=[], steps=0, attempts=0)
               for job in jobs]
    while any(state['steps'] < job['steps'] for state, job in zip(streams, jobs)):
        work, owners = [], []
        for index, (state, job) in enumerate(zip(streams, jobs)):
            remaining = job['steps'] - state['steps']
            if remaining <= 0:
                continue
            if state['attempts'] >= 30:
                raise RuntimeError('Too many unresolved training episodes; no silent filtering')
            count = min((remaining - 1) // 80 + 1, 30 - state['attempts'])
            for _ in range(count):
                req = request_builder(job, state['rng'], entries)
                work.append(job | dict(steps=1, requests=[req]))
                owners.append(index)
                state['attempts'] += 1
        for index, result in zip(owners, pool.map(rollout_function, work)):
            if any(length > 80 for length in result['lengths']):
                raise RuntimeError('Episode exceeded the registered 80-step horizon')
            streams[index]['parts'].append(result)
            streams[index]['steps'] += sum(result['lengths'])
    results = []
    for state in streams:
        parts = state['parts']
        valid = [part for part in parts if part['lengths']]
        arrays = {key: np.concatenate([part['arrays'][key] for part in valid])
                  for key in valid[0]['arrays']}
        results.append(dict(arrays=arrays,
            lengths=[n for part in parts for n in part['lengths']],
            summaries=[item for part in parts for item in part['summaries']],
            invalid=[item for part in parts for item in part['invalid']],
            digest=hashlib.sha256(b''.join(arrays[key].tobytes() for key in sorted(arrays))).hexdigest()))
    return results
