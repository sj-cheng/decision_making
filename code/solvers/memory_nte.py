"""Private copy of frozen T simulation: neural proposals only at tree expansion.
Rollout, opponent, widening, rewards and backpropagation remain original.
Simulated target reaches the network exclusively through the sensor record.
"""
import copy
import time
import numpy as np
from numba import njit
from learning.memory_nte import inference, observation_array
from solvers.local_pursuit import (Controller,move,modal,uniform,safe,proposal,
    visible,predict,recover,guided)

@njit(cache=True)
def neural_proposal(at,speed,boxes,rng,count,records,n,memory,weights,stats):
    guided_branch=uniform(rng)<.5
    if guided_branch:
        mu,lv=inference(boxes,records,n,speed,memory,weights)
        stats[0]+=1
    for attempt in range(1000):
        if guided_branch:
            eps=np.empty(2)
            for axis in range(2):
                eps[axis]=np.sqrt(-2*np.log(uniform(rng)))*np.cos(2*np.pi*uniform(rng))
            result=speed*np.minimum(1.,np.maximum(-1.,mu+np.exp(.5*lv)*eps))
        else:result=speed*np.array([2*uniform(rng)-1,2*uniform(rng)-1])
        if safe(at,result,speed,boxes,count):return result,guided_branch
        stats[1]+=1
    stats[2]+=1
    return np.zeros(2),guided_branch

@njit(cache=True)
def neural_simulation(at0, particles0, last0, history0, speed, remaining, vertices, dist, boxes,
               root_actions, root_sources, root_n, rng, count, parents, actions, sources, visits, totals, tree_size, records0, memory, weights, neural_count):
    # Each node indexes an action history, never a sampled state or observation.
    records=np.zeros((161,6))
    records[:len(records0)]=records0
    rn=len(records0)
    particles = particles0.copy()
    target = particles[min(63, int(uniform(rng)*64))].copy()
    at = at0.copy()
    last = last0.copy()
    history = np.empty((len(history0)+remaining, 2))
    history[:len(history0)] = history0
    hn = len(history0)
    node, pn, expanded = 0, 1, False
    path = np.zeros(remaining+1, np.int64)
    rewards = np.zeros(remaining)
    steps = 0
    for depth in range(min(100, remaining)):
        if not expanded:
            width = max(1, int(np.ceil(2*np.sqrt(max(1, visits[node])))))
            children = 0
            for c in range(1, tree_size):
                children += int(parents[c] == node)
            if children < width:
                base = move(at, modal(particles), speed, vertices, dist, boxes, count)
                source = 0
                if depth == 0:
                    k = min(root_n-1, int(uniform(rng)*root_n))
                    base, source = root_actions[k], root_sources[k]
                action, isguided = neural_proposal(at, speed, boxes, rng, count, records, rn, memory, weights, neural_count)
                child = tree_size
                tree_size += 1
                parents[child], actions[child] = node, action
                sources[child] = source if isguided else 3
                expanded = True
            else:
                child, best = -1, -1e100
                for c in range(1, tree_size):
                    if parents[c] == node:
                        score = totals[c]/max(1, visits[c]) + np.sqrt(np.log(visits[node]+1)/max(1, visits[c]))
                        if score > best:
                            child, best = c, score
                action = actions[child].copy()
            node = child
            path[pn] = node
            pn += 1
        else:
            base = move(at, modal(particles), speed, vertices, dist, boxes, count)
            action, _ = proposal(at, base, speed, boxes, rng, count)
        # Simulated opponent has a public full-position sensor, matching the
        # information advantage of the real Search Only opponent. Its model
        # is an inexpensive evasive rollout, identical for T and V. It never
        # receives our already selected action.
        away = target-at
        away /= max(1., np.max(np.abs(away)))
        other, _ = proposal(target, away, 1., boxes, rng, count)
        d0 = np.linalg.norm(target-at)
        nxt, end = at+action, target+other
        steps += 1
        count[1] += 1
        if not safe(at, action, speed, boxes, count):
            rewards[depth] = -1.
            break
        if not safe(target, other, 1., boxes, count):
            break
        d1 = np.linalg.norm(end-nxt)
        if d1 < 1.:
            rewards[depth] = .5*(1.-(80-remaining+depth+1)/80.)
            break
        rewards[depth] = .03*min(1., max(-1., (d0-d1)/4.))
        at, target = nxt, end
        records[rn,0]=80-remaining+depth+1
        records[rn,1:3]=at
        seen=visible(at,target,boxes,count)
        if seen:
            records[rn,3]=1.
            records[rn,4:6]=target
        rn+=1
        # Only this sensor result, never target itself, reaches G's belief.
        if seen:
            particles[:] = target
            last, hn = target.copy(), 0
        else:
            history[hn] = at
            hn += 1
            if not predict(particles, at, boxes, rng, count):
                new, support = recover(last, history[:hn], boxes, rng, count)
                if support:
                    particles[:] = new
                else:
                    # An impossible/too rare simulated history has no value
                    # cutoff; retain no hidden-world truth and stop rollout.
                    break
    # Depth-specific returns; no value network, no information bonus.
    for j in range(pn):
        visits[path[j]] += 1
        totals[path[j]] += np.sum(rewards[max(0, j-1):steps])
    return tree_size


class NeuralController(Controller):
    def __init__(self,public,method,seed,weights,simulations=1000,deadline_ms=50.):
        super().__init__(public,'T',seed,simulations,deadline_ms)
        if method not in ('N','H'):raise ValueError(method)
        self.neural_method=method
        self.weights=weights

    def act(self,obs,fixed_simulations=None):
        started=time.perf_counter();p=self.public;count=np.zeros(2,np.int64)
        self.update(obs,count)
        base=move(obs.position,modal(self.particles),p.speed,p.vertices,p.dist,p.boxes,count)
        records=observation_array(self.observations)
        actions0,sources0,n=guided(obs.position,self.particles,p.speed,p.vertices,p.dist,p.edist,p.boxes,count,False)
        cap=self.simulations if fixed_simulations is None else fixed_simulations
        parents=np.full(cap+1,-1,np.int64);actions=np.zeros((cap+1,2));sources=np.zeros(cap+1,np.int64)
        visits=np.zeros(cap+1,np.int64);totals=np.zeros(cap+1);size=1;completed=0;neural_count=np.zeros(3,np.int64)
        for _ in range(cap):
            if fixed_simulations is None and self.deadline_ms>0 and 1000*(time.perf_counter()-started)>=self.deadline_ms:break
            size=neural_simulation(obs.position,self.particles,self.last,np.asarray(self.history).reshape(-1,2),p.speed,
                obs.remaining,p.vertices,p.dist,p.boxes,actions0,sources0,n,self.rng,count,
                parents,actions,sources,visits,totals,size,records,self.neural_method=='H',self.weights,neural_count)
            completed+=1
        children=[i for i in range(1,size) if parents[i]==0 and safe(obs.position,actions[i],p.speed,p.boxes,count)]
        source='fallback_G'
        if children:
            best=max(children,key=lambda i:totals[i]/max(1,visits[i]))
            base=actions[best].copy();source='neural' if sources[best]==0 else 'uniform'
        root=[dict(action=actions[i].tolist(),source='neural' if sources[i]==0 else 'uniform',visits=int(visits[i]),
                   value=float(totals[i]/max(1,visits[i]))) for i in range(1,size) if parents[i]==0]
        latency=1000*(time.perf_counter()-started)
        self.last_stats=dict(simulations=completed,transitions=int(count[1]),geometry_queries=int(count[0]),
            latency_ms=latency,overrun=bool(self.deadline_ms>0 and latency>self.deadline_ms),candidate_ms=0.,
            source=source,root=root,recoveries=self.recoveries,network_calls=int(neural_count[0]),
            proposal_rejections=int(neural_count[1]),proposal_exhaustions=int(neural_count[2]),
            normalized_action_amplitude=float(np.max(np.abs(base))/p.speed))
        if not safe(obs.position,base,p.speed,p.boxes,count):raise RuntimeError('Controller generated illegal real action')
        return base

