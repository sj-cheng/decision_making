"""E52 public-observation NAR/NAE networks.

The networks score only features derived from AgentObservation.  They do not
receive simulator truth or a global map.  PyTorch is imported lazily so the
geometry and protocol tests remain usable in lightweight environments.
"""
from pathlib import Path
import numpy as np


SCHEMA = "e52_nar_nae_v2"


def observation_features(observation, endpoint, *, level=0):
    pose = np.asarray(observation.pose, float)
    endpoint = np.asarray(endpoint, float)
    belief = getattr(observation, "belief", None)
    particles = np.asarray(getattr(belief, "particles", np.zeros((0, 2))), float).reshape(-1, 2)
    values = [*(endpoint - pose), float(np.linalg.norm(endpoint - pose)), float(level),
              float(getattr(observation, "cycle", 0)) / 160.]
    if len(particles):
        nearest = particles[np.argmin(np.linalg.norm(particles - endpoint, axis=1))]
        values += [*(nearest - pose), float(np.linalg.norm(nearest - endpoint)), 1.]
    else:
        values += [0., 0., 0., 0.]
    peers = tuple(getattr(observation, "peer_messages", ()))
    if peers:
        peer = peers[0]
        values += [*(np.asarray(peer.pose, float) - pose), *(np.asarray(peer.velocity, float)), 1.]
    else:
        values += [0., 0., 0., 0., 0.]
    return np.asarray(values, dtype=np.float32)


class NARNAE:
    """Separate MLP pair for candidate ranking and node value."""
    def __init__(self, input_dim=14, hidden=128):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("E52 NAR/NAE requires the dm_env_py311 PyTorch environment") from exc
        self.torch = torch
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.nar = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.nae = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _batch(self, observations, endpoints, levels):
        x = np.asarray([observation_features(o, e, level=l)
                        for o, e, l in zip(observations, endpoints, levels)], np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("E52 feature schema mismatch")
        return self.torch.as_tensor(x)

    def score(self, observation, endpoint, level=0):
        x = self._batch([observation], [endpoint], [level])
        with self.torch.no_grad():
            return float(self.nar(x)[0, 0]), float(self.nae(x)[0, 0])

    def value_scores(self, observation, endpoints, levels):
        x = self._batch([observation] * len(endpoints), endpoints, levels)
        with self.torch.no_grad():
            return self.nae(x).flatten().numpy()

    def fit(self, observations, endpoints, nar_targets, nae_targets, *, epochs=200,
            learning_rate=1e-3, weight_decay=1e-4):
        x = self._batch(observations, endpoints, [0] * len(observations))
        nar_y = self.torch.as_tensor(np.asarray(nar_targets, np.float32).reshape(-1, 1))
        nae_y = self.torch.as_tensor(np.asarray(nae_targets, np.float32).reshape(-1, 1))
        if len(x) == 0 or len(x) != len(nar_y) or len(x) != len(nae_y):
            raise ValueError("invalid E52 training data")
        optimizer = self.torch.optim.Adam(list(self.nar.parameters()) + list(self.nae.parameters()),
                                          lr=float(learning_rate), weight_decay=float(weight_decay))
        loss_fn = self.torch.nn.MSELoss()
        self.nar.train(); self.nae.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            loss = loss_fn(self.nar(x), nar_y) + loss_fn(self.nae(x), nae_y)
            loss.backward(); optimizer.step()
        self.nar.eval(); self.nae.eval()
        return float(loss.detach().cpu())

    def fit_records(self, records, *, epochs=100, seed=52, progress=None):
        """Grouped visit-policy cross entropy and eight-cycle value regression."""
        self.torch.manual_seed(seed)
        optimizer = self.torch.optim.Adam(list(self.nar.parameters()) + list(self.nae.parameters()), lr=1e-3)
        if not records or epochs < 1:
            raise ValueError('need training records and positive epochs')
        width = max(len(r['features']) for r in records)
        features = np.zeros((len(records), width, self.input_dim), np.float32)
        policy = np.zeros((len(records), width), np.float32)
        mask = np.zeros((len(records), width), bool)
        chosen = np.zeros(len(records), np.int64)
        targets = np.zeros(len(records), np.float32)
        for index, record in enumerate(records):
            n = len(record['features'])
            features[index, :n] = record['features']; policy[index, :n] = record['policy']
            mask[index, :n] = True; chosen[index] = record['value_index']
            targets[index] = record['returns'][record['value_index']]
        x = self.torch.as_tensor(features); y = self.torch.as_tensor(policy)
        valid = self.torch.as_tensor(mask); selected = self.torch.as_tensor(chosen)
        values = self.torch.as_tensor(targets)
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            order = rng.permutation(len(records)); epoch_losses = []
            for start in range(0, len(order), 32):
                ids = self.torch.as_tensor(order[start:start+32]); batch = x[ids]
                logits = self.nar(batch).squeeze(-1).masked_fill(~valid[ids], -1e9)
                policy_loss = -(y[ids] * self.torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                action_x = batch[self.torch.arange(len(ids)), selected[ids]]
                value_loss = self.torch.nn.functional.mse_loss(self.nae(action_x).flatten(), values[ids])
                optimizer.zero_grad(); loss = policy_loss + value_loss
                loss.backward(); optimizer.step(); epoch_losses.append(float(loss.detach()))
            if progress is not None and (epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs):
                progress(dict(epoch=epoch+1, epochs=epochs, loss=float(np.mean(epoch_losses))))
        self.nar.eval(); self.nae.eval()
        return float(loss.detach())

    def save(self, path, metadata=None):
        self.torch.save(dict(schema=SCHEMA, input_dim=self.input_dim, hidden=self.hidden,
                             nar=self.nar.state_dict(), nae=self.nae.state_dict(), metadata=metadata or {}), Path(path))

    @classmethod
    def load(cls, path):
        import torch
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        if data.get("schema") != SCHEMA:
            raise ValueError("unsupported E52 NAR/NAE schema")
        model = cls(data["input_dim"], data["hidden"])
        model.nar.load_state_dict(data["nar"]); model.nae.load_state_dict(data["nae"])
        model.metadata = data.get("metadata", {})
        model.nar.eval(); model.nae.eval()
        return model
