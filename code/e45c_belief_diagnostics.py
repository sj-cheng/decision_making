"""C-only empty-support repair at the frozen-D/team sensor interface.

Normal arithmetic and RNG draws match the frozen TeamBelief. Only when its
team posterior is empty, use the SAME prior/graph/D kernel with the canonical
team negative likelihood. A genuinely empty canonical posterior stays fatal.
Every intervention retains the original failed arrays and full PUBLIC history.
"""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
from solvers.e45_team_adapter import TeamBelief as FrozenTeamBelief, team_hidden_mask
from solvers.frozen_d_search_interface import Observation, sensor_visible
from l45_corner_shared_20261025.corner_estimator import export_regions
from search_value_model import predict_joint

SUPPORT_ERRORS = ('Empty team-conditioned finite-grid support',
                  'Empty weighted public-history support')
EVIDENCE = None


class RecordedTeamBelief(FrozenTeamBelief):
    def __init__(self, boxes, seed):
        super().__init__(boxes, seed)
        self.public_history = []
        self.diagnostic_seed = int(seed)

    def update(self, obs):
        self.public_history.append(copy.deepcopy(obs))
        try:
            return self.update_team(obs)
        except RuntimeError as exc:
            if str(exc) not in SUPPORT_ERRORS or EVIDENCE is None:raise
            stem=self.record_failure(exc,obs)
            raise RuntimeError(str(exc)+'; public_history_evidence='+str(stem)) from exc

    def update_team(self,obs):
        # Keep normal-path operations and draw order equal to the frozen
        # wrapper; frozen per-target Estimator.update is never modified.
        if self.step>=0 and obs['step']!=self.step+1:raise ValueError('Nonconsecutive team history')
        if obs['remaining']!=80-obs['step']:raise ValueError('Bad team clock')
        positions=np.asarray(obs['positions']);ap=np.asarray(obs['active_p']);ae=obs['active_e']
        live=np.flatnonzero(ap)
        if not len(live):raise ValueError('No active observer')
        masses=[];repairs=[]
        for j,e in enumerate(self.estimators):
            if not ae[j]:self.known[j]=None;continue
            positive=[i for i in live if obs['reports'][j][i] is not None]
            if positive:
                at=int(positive[0]);target=np.asarray(obs['reports'][j][at],float)
                for i in positive:
                    if not np.array_equal(target,obs['reports'][j][i]):raise ValueError('Conflicting identified measurements')
                    if not sensor_visible(positions[i],target,self.boxes):raise ValueError('Invalid team measurement')
                self.known[j]=target.copy()
            else:
                at=j if ap[j] else int(live[0]);target=None;self.known[j]=None
            # Frozen update replaces weights and reachable, so retaining the
            # prior reference does not copy the large graph or posterior.
            prior=e.weights;prior_graph=e.graph;prior_rng=e.rng.copy()
            e.update(Observation(obs['step'],positions[at],target,obs['remaining']))
            if target is None:
                allowed=team_hidden_mask(e.graph[0],positions,ap,self.boxes)
                e.weights[~allowed]=0.;e.reachable &= allowed
                mass=float(e.weights.sum());masses.append(mass)
                if mass==0 and np.isfinite(e.weights).all():
                    stem=self.record_failure(RuntimeError(SUPPORT_ERRORS[0]),obs) if EVIDENCE is not None else None
                    if self.repair_boundary(e,prior,prior_graph,prior_rng,allowed):
                        mass=float(e.weights.sum());repairs.append(j)
                        if stem is not None:
                            path=stem.with_suffix('.json');metadata=json.loads(path.read_text())
                            metadata.update(recovered=True,repaired_target=j,posterior_mass=mass,
                                repair='same D prior/graph/kernel, canonical team negative likelihood')
                            temp=stem.with_suffix('.tmp.json');temp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2));temp.replace(path)
                if mass<=0 or not np.isfinite(mass):raise RuntimeError(SUPPORT_ERRORS[0])
                e.weights/=mass
                e.ids,e.particle_weights,e.labels=export_regions(e.weights,e.regions,e.rng,64)
                e.particles=e.graph[0][e.ids].copy()
                if not team_hidden_mask(e.particles,positions,ap,self.boxes).all():raise RuntimeError('Export violates team observation')
        self.step=obs['step'];self.last_stats=dict(min_conditioning_mass=min(masses,default=1.))
        if repairs:self.last_stats.update(canonical_sensor_support_repair=True,repaired_targets=repairs)

    @staticmethod
    def repair_boundary(e,prior,prior_graph,prior_rng,allowed):
        if e.arm!='D':return False
        g=e.graph
        if prior_graph is None:
            prior=np.zeros_like(e.weights);prior[g[5]]=1/9.
        if prior is None or not np.isfinite(prior).all():return False
        prediction=predict_joint(prior,g[1],g[2],g[3]);prediction[~allowed]=0.
        mass=float(prediction.sum())
        if not np.isfinite(mass) or mass<=0:return False
        # Zero original mass and positive canonical mass certify a filtering
        # discrepancy, not a missing path. Every revived node has a lawful
        # ancestral edge from the actual previous posterior on this D graph.
        posterior=prediction/mass;parents=np.full(len(posterior),-1,np.int32)
        support=posterior.sum(1)>0
        for i in np.flatnonzero(prior.sum(1)>0):
            for k in g[1][i,:g[2][i]]:
                if support[k] and parents[k]<0:parents[k]=i
        if np.any(support&(parents<0)):return False
        e.weights=posterior;e.reachable=support;e.parents[-1]=parents
        # Redo the frozen estimator's export from its pre-update RNG, then
        # update_team performs the normal second team-conditioned export.
        e.rng[:]=prior_rng
        e.ids,e.particle_weights,e.labels=export_regions(e.weights,e.regions,e.rng,64)
        e.particles=g[0][e.ids].copy()
        e.last_stats.update(support_points=int(support.sum()),mode_mass=e.weights.sum(0).tolist())
        return True

    def record_failure(self,exc,obs):
        stem=Path(EVIDENCE)/f'public_support_{os.getpid()}_{time.time_ns()}'
        stem.parent.mkdir(parents=True,exist_ok=True);arrays={'boxes':self.boxes}
        for j,e in enumerate(self.estimators):
            for name in ('weights','reachable','last','particles','particle_weights','ids','rng'):
                value=getattr(e,name,None)
                if isinstance(value,np.ndarray):arrays[f'target_{j}_{name}']=value
            if e.graph is not None:
                for k,value in enumerate(e.graph):arrays[f'target_{j}_graph_{k}']=np.asarray(value)
        path=stem.with_suffix('.npz');temp=stem.with_suffix('.tmp.npz')
        np.savez_compressed(temp,**arrays);temp.replace(path)
        metadata=dict(error=str(exc),seed=self.diagnostic_seed,boxes=self.boxes.tolist(),history=self.public_history,
            failed_public_step=obs['step'],arrays=str(path),recovered=False,
            scope='Public measurements and estimator internals only; no hidden environment state')
        temp=stem.with_suffix('.tmp.json');temp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2));temp.replace(stem.with_suffix('.json'))
        return stem


def install(evidence):
    """C-only process-local hook, including the independent public replay."""
    global EVIDENCE
    EVIDENCE=Path(evidence)
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    import run_e45b_role_free as runner
    adapter.TeamBelief=RecordedTeamBelief
    joint.TeamBelief=RecordedTeamBelief
    runner.TeamBelief=RecordedTeamBelief
