"""W repair revision: finer public-history support, preserving the G normal path.

The original 0.5/0.25 filter is untouched.  Only a certified exhausted G filter
is reconstructed at 0.125, then 0.0625.  This is a discretization amendment,
not an exact continuous posterior. All negative reports since the most recent
identified measurement remain constraints; hidden environment state is absent.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from learning.e45g_ppo import TeamBelief as GTeamBelief
from solvers.e45f_support_refinement import refined_posterior
from solvers.frozen_d_search_interface import Observation
from solvers.e45_team_adapter import team_hidden_mask
from l45_corner_shared_20261025.corner_estimator import geometry_labels, export_regions

REVISION = "W-support-r1-public-history-1over8-1over16"
SPACINGS = (.125, .0625)
EVIDENCE = None


def finer_posterior(boxes, history, target):
    for spacing in SPACINGS:
        result = refined_posterior(boxes, history, target, spacing=spacing)
        if result is not None:
            return result, spacing
    raise RuntimeError("W finer public-history support exhausted; no lawful recovery")


class TeamBelief(GTeamBelief):
    def update(self, obs):
        rngs = [e.rng.copy() for e in self.estimators]
        try:
            return super().update(obs)
        except RuntimeError as exc:
            if not str(exc).startswith("G refinement also exhausted;"):
                raise
            import e45c_belief_diagnostics as diagnostic
            evidence_dir = EVIDENCE if EVIDENCE is not None else diagnostic.EVIDENCE
            if evidence_dir is not None:
                directory = Path(evidence_dir); directory.mkdir(parents=True, exist_ok=True)
                evidence = directory / f"support_{os.getpid()}_{time.time_ns()}.json"
                evidence.write_text(json.dumps(dict(revision=REVISION, error=str(exc),
                    boxes=self.boxes.tolist(), seed=self.diagnostic_seed,
                    history=self.public_history, recovered=False)))
            else:
                evidence = None
            recovered = []
            for j, e in enumerate(self.estimators):
                if not obs['active_e'][j]:
                    self.known[j] = None
                    continue
                positive = [(i, q) for i, q in enumerate(obs['reports'][j])
                            if obs['active_p'][i] and q is not None]
                if positive:
                    at, measurement = positive[0]
                    self.known[j] = np.asarray(measurement, float)
                    e.step = obs['step'] - 1
                    e.update(Observation(obs['step'], np.asarray(obs['positions'][at]),
                                         self.known[j], obs['remaining']))
                    continue
                # Reconstruct from the last positive and ALL later public
                # negatives, never from a uniform reset or target truth.
                (graph, weights, parents, observers), spacing = finer_posterior(
                    self.boxes, self.public_history, j)
                e.graph, e.weights, e.parents = graph, weights, parents
                e.reachable = weights.sum(1) > 0
                e.regions = geometry_labels(graph[0], self.boxes)
                e.observers = [o[0] for o in observers]
                e.geometry_means = None
                positives = [q for item in self.public_history
                             for i, q in enumerate(item['reports'][j])
                             if item['active_p'][i] and q is not None]
                e.last = np.asarray(positives[-1]); e.rng[:] = rngs[j]
                e.step = obs['step']
                e.ids, e.particle_weights, e.labels = export_regions(weights, e.regions, e.rng, 64)
                e.particles = graph[0][e.ids].copy(); self.known[j] = None
                if not team_hidden_mask(e.particles, np.asarray(obs['positions']),
                                        np.asarray(obs['active_p']), self.boxes).all():
                    raise RuntimeError("Recovered particles violate public negatives")
                e.last_stats.update(support_points=int(e.reachable.sum()),
                                    mode_mass=weights.sum(0).tolist(), refined_spacing=spacing)
                recovered.append(dict(step=obs['step'], target=j, spacing=spacing,
                                      support_points=int(e.reachable.sum()), revision=REVISION))
            self.refinements.extend(recovered); self.step = obs['step']
            self.last_stats = dict(adaptive_grid_refinement=recovered, min_conditioning_mass=0.)
            if evidence is not None:
                metadata = json.loads(evidence.read_text())
                metadata.update(recovered=True, refinements=recovered)
                evidence.write_text(json.dumps(metadata))


def install(evidence=None):
    """Process-local replacement; original source and running workers stay fixed."""
    global EVIDENCE
    EVIDENCE = evidence
    import learning.e45w_joint_env as env
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    import run_e45b_role_free as base
    env.TeamBelief = adapter.TeamBelief = joint.TeamBelief = base.TeamBelief = TeamBelief
