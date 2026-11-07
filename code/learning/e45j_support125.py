"""Explicit J amendment: public-history 0.125 recovery after G/F exhaustion."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from learning.e45g_ppo import TeamBelief as PreviousBelief
from solvers.e45f_support_refinement import refined_posterior
from solvers.e45_team_adapter import team_hidden_mask
from solvers.frozen_d_search_interface import Observation
from l45_corner_shared_20261025.corner_estimator import geometry_labels,export_regions

EVIDENCE=None


class TeamBelief125(PreviousBelief):
    def update(self,obs):
        rngs=[e.rng.copy() for e in self.estimators]
        try:return super().update(obs)
        except RuntimeError as exc:
            if not str(exc).startswith('G refinement also exhausted;'):raise
            original=str(exc)
        started=time.perf_counter();recovered=[]
        for j,e in enumerate(self.estimators):
            if not obs['active_e'][j]:self.known[j]=None;continue
            positive=[i for i,q in enumerate(obs['reports'][j]) if obs['active_p'][i] and q is not None]
            if positive:
                i=positive[0];self.known[j]=np.asarray(obs['reports'][j][i],float)
                e.step=obs['step']-1;e.rng[:]=rngs[j]
                e.update(Observation(obs['step'],np.asarray(obs['positions'][i]),self.known[j],obs['remaining']))
                continue
            result=refined_posterior(self.boxes,self.public_history,j,spacing=.125)
            if result is None:raise RuntimeError('J 0.125 recovery exhausted; '+original)
            graph,weights,parents,observers=result
            e.graph=graph;e.weights=weights;e.parents=parents;e.reachable=weights.sum(1)>0
            e.regions=geometry_labels(graph[0],self.boxes);e.geometry_means=None
            tail=self.public_history[len(self.public_history)-len(observers):] if observers else []
            e.observers=[o[j] if h['active_p'][j] else o[np.flatnonzero(h['active_p'])[0]] for o,h in zip(observers,tail)]
            positives=[q for h in self.public_history for i,q in enumerate(h['reports'][j]) if h['active_p'][i] and q is not None]
            e.last=np.asarray(positives[-1]);e.rng[:]=rngs[j];e.step=obs['step']
            e.ids,e.particle_weights,e.labels=export_regions(weights,e.regions,e.rng,64)
            e.particles=graph[0][e.ids].copy();self.known[j]=None
            assert team_hidden_mask(e.particles,np.asarray(obs['positions']),np.asarray(obs['active_p']),self.boxes).all()
            e.last_stats.update(support_points=int(e.reachable.sum()),mode_mass=weights.sum(0).tolist(),refined_spacing=.125)
            recovered.append(dict(step=obs['step'],target=j,spacing=.125,support_points=int(e.reachable.sum())))
        self.refinements.extend(recovered);self.step=obs['step']
        elapsed=1000*(time.perf_counter()-started)
        self.last_stats=dict(adaptive_grid_refinement=recovered,min_conditioning_mass=0.,support125_recovery_ms=elapsed)
        if EVIDENCE is not None:
            history=json.dumps(self.public_history,sort_keys=True);key=hashlib.sha256(history.encode()).hexdigest()[:16]
            p=Path(EVIDENCE);p.mkdir(parents=True,exist_ok=True)
            record=dict(seed=self.diagnostic_seed,boxes=self.boxes.tolist(),history=self.public_history,recovered=recovered,
                recovery_ms=elapsed,public_only=True,discretization_changed=True,scope='invocation; includes independent scoring replay')
            (p/f'{os.getpid()}_{key}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


def install(evidence):
    global EVIDENCE
    EVIDENCE=Path(evidence)
    import learning.e45i_official_mappo as neural
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    import run_e45b_role_free as runner
    neural.TeamBelief=adapter.TeamBelief=joint.TeamBelief=runner.TeamBelief=TeamBelief125
