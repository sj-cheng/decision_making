"""E44 continuous public-belief neural expansion, with shared feedback rollout."""
import time
import numpy as np
from solvers.nte_d_local import Controller as Base
from solvers.frozen_d_search_interface import safe
from solvers.e44_routes import route_arrays, hand_probabilities
from learning.e44_policy import Runtime


class Controller(Base):
    def __init__(self,boxes,speed,seed,proposal='uniform',allocation='focused',policy=None,direct=False,budget_ms=200.):
        super().__init__(boxes,speed,seed,'M0',budget_ms=budget_ms,simulations=100000)
        self.e44_proposal,self.allocation,self.direct = proposal,allocation,bool(direct)
        self.network = None if policy is None else Runtime(policy)
        if proposal == 'neural' and self.network is None:
            raise ValueError('Neural expansion needs a checkpoint')
        self._in_direct = False

    def features(self,node):
        x = super().features(node)
        if 'routes' not in x:
            started = time.perf_counter()
            samples = x['particles'][:,:2]*20.+node.at
            if self.estimator.graph is None:
                labels = np.zeros(64,np.int64)
            else:
                # Match the frozen region half-space definition, not true route IDs.
                from l45_corner_shared_20261025.corner_estimator import geometry_labels
                labels = geometry_labels(samples,self.public.boxes)
            desc,mask,anchors,keys,omitted,queries = self._operation(route_arrays,node.at,
                samples,x['weights'],labels,self.public.speed,self.public.vertices,self.public.dist,
                self.public.boxes,self.motion_boxes)
            x.update(routes=desc,route_mask=mask,anchors=anchors)
            node.e44_route_keys,node.e44_omitted = keys,omitted
            self.queries += queries
            self.feature_ms += 1000*(time.perf_counter()-started)
        return x

    def propose(self,node,direct=False):
        started = time.perf_counter()
        neural = self.network is not None and (self._in_direct or self.proposal_rng.random()<.5)
        parameters = None
        x = None
        if neural or self.e44_proposal == 'route':
            x = self.features(node)
        if neural:
            if node.parameters is None:
                t = time.perf_counter(); node.parameters = self.network(x)
                self.inference_ms += 1000*(time.perf_counter()-t)
            parameters = node.parameters
        # Shared route proposal supplies one representative per legal mode before
        # drawing perturbations. Continuous support remains through uniform draws.
        candidate = None
        if not neural and self.e44_proposal == 'route' and not self._in_direct:
            active = np.flatnonzero(x['route_mask'])
            if len(node.actions) < len(active):
                candidate = x['anchors'][active[len(node.actions)]]*self.public.speed
        for attempt in range(64):
            self._allowed()
            if candidate is not None and attempt == 0:
                a = candidate.copy()
            elif neural:
                probs,location,logstd = parameters
                k = int(self.proposal_rng.choice(len(probs),p=probs))
                a = np.tanh(location[k]+np.exp(logstd[k])*self.proposal_rng.standard_normal(2))*self.public.speed
            elif self.e44_proposal == 'route' and self.proposal_rng.random()<.5:
                k = int(self.proposal_rng.choice(len(x['route_mask']),p=hand_probabilities(x)))
                base = x['anchors'][k]
                # Both full representatives and continuously variable amplitude.
                scale = self.proposal_rng.uniform(0.,1.)
                a = np.clip(scale*base+self.proposal_rng.normal(0.,.1,2),-1.,1.)*self.public.speed
            else:
                a = self.proposal_rng.uniform(-self.public.speed,self.public.speed,2)
            self.proposals += 1
            if safe(node.at,a,self.public.speed,self.motion_boxes,np.zeros(2,np.int64)):
                self.proposal_ms += 1000*(time.perf_counter()-started)
                return a
            self.rejections += 1
        self.stop_fallbacks += 1
        return np.zeros(2)

    def select(self,node):
        width = max(1,int(np.ceil(2.*np.sqrt(node.n+1.))))
        if self.allocation == 'focused':
            width = min(12,max(1,int(np.ceil((node.n+1.)**.25))))
        if len(node.actions) < width:
            action = self.propose(node)
            match = next((i for i,a in enumerate(node.actions) if np.max(np.abs(a-action))<1e-10),None)
            if match is None:
                node.actions.append(action);node.visits.append(0);node.totals.append(0.);node.squares.append(0.)
                return len(node.actions)-1
        unvisited = [i for i,n in enumerate(node.visits) if n == 0]
        if unvisited:
            return unvisited[0]
        v = np.asarray(node.visits)
        return int(np.argmax(np.asarray(node.totals)/v+np.sqrt(2.*np.log(node.n+1.)/v)))

    def act(self,obs,fixed_simulations=None):
        # Base's DIRECT path is selected without loading its incompatible old NN.
        self.mode = 'DIRECT' if self.direct else 'M0'
        self._in_direct = self.direct
        action = super().act(obs,fixed_simulations)
        self.last_stats.update(e44_proposal=self.e44_proposal,allocation=self.allocation,
            neural_kind=None if self.network is None else self.network.kind,direct=self.direct)
        return action
