"""Identity-preserving camera tracks; timestamps are sensor times, never relay times."""
from dataclasses import dataclass
import numpy as np


@dataclass
class TargetTrack:
    target_id: int
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    timestamp: float
    confidence: float
    source: int

    def payload(self):
        return dict(target_id=self.target_id, position=self.position.tolist(),
                    velocity=self.velocity.tolist(), covariance=self.covariance.tolist(),
                    timestamp=self.timestamp, confidence=self.confidence, source=self.source)


class TargetBelief:
    def __init__(self, agent_id, ttl=5):
        self.agent_id = int(agent_id)
        self.ttl = float(ttl)
        self.tracks = {}
        self.captured = set()
        self.now = 0.

    def update(self, reports, timestamp):
        self.now = float(timestamp)
        for tid, point in reports:
            if tid in self.captured:
                continue
            old = self.tracks.get(tid)
            point = np.asarray(point, float).copy()
            velocity = ((point - old.position) / (self.now - old.timestamp)
                        if old is not None and self.now > old.timestamp else np.zeros(2))
            speed = np.linalg.norm(velocity)
            if speed > 2.: velocity *= 2. / speed
            self.tracks[tid] = TargetTrack(int(tid), point, velocity, np.eye(2)*.04,
                                          self.now, 1., self.agent_id)
        self.expire()

    def expire(self):
        self.tracks = {k:v for k,v in self.tracks.items()
                       if k not in self.captured and 0 <= self.now-v.timestamp <= self.ttl}

    def merge(self, messages, timestamp):
        self.now = float(timestamp)
        for msg in messages:
            if not msg.valid_at(int(timestamp)): continue
            for item in msg.beliefs:
                if not isinstance(item, dict): continue
                tid, stamp = int(item['target_id']), float(item['timestamp'])
                old = self.tracks.get(tid)
                if tid in self.captured or not 0 <= self.now-stamp <= self.ttl: continue
                if old is not None and old.timestamp >= stamp: continue
                self.tracks[tid] = TargetTrack(tid, np.array(item['position']), np.array(item['velocity']),
                    np.array(item['covariance']), stamp, float(item['confidence']), int(item['source']))
        self.expire()

    def capture(self, target_id):
        self.captured.add(int(target_id))
        self.tracks.pop(int(target_id), None)

    @property
    def particles(self):
        return np.asarray([v.position + v.velocity * (self.now-v.timestamp)
                           for v in self.tracks.values()], float).reshape(-1, 2)

    @property
    def weights(self):
        return np.asarray([v.confidence*np.exp(-.3*(self.now-v.timestamp)) for v in self.tracks.values()])

    def payload(self):
        return tuple(v.payload() for v in self.tracks.values())
