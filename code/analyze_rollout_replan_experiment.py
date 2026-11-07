"""Post-run public-history replay, independent legality and budget accounting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json,fcntl,hashlib,collections,argparse
from pathlib import Path
import numpy as np
from run_rollout_replan_experiment import ROOT,OUT,SOURCE,METHODS,inputs,save,sha,loader
from solvers.frozen_d_search_interface import Observation,COMPASS,safe
from estimator_audit_geometry import blocked,point_legal
from audit_frozen_d_search_interface import quantiles


def audit(stage):
    rows=json.loads((OUT/stage/'results.json').read_text());frames=[];episodes=[];errors=[];pairs={}
    for n,row in enumerate(rows):
        assert sha(row['episode'])==row['episode_sha256']
        payload=json.loads(Path(row['episode']).read_text());packet,environment=inputs(payload['task'])
        trace=payload['trace'];boxes=np.asarray(packet['map']);speed=packet['motion']['pursuer_axis_limit']
        if row['failure'] is not None:errors.append(dict(key=row['key'],kind='recorded_engineering_failure',detail=row['failure']))
        assert hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest()==row['trajectory_sha256']
        initial=next(w for w in environment['worlds'] if w['id']==row['world'])['handoff_full_state']
        assert payload['initial_state']==initial
        own=loader.load_estimator(boxes,payload['task']['seed'])
        for item in packet['history'][:-1]:
            own.update(Observation(item['step'],np.asarray(item['position']),None if item['target'] is None else np.asarray(item['target']),item['remaining']))
        last=np.array(initial);first=None
        for i,record in enumerate(trace):
            at=np.array(record['state']);nxt=np.array(record['next_state']);obs=record['observation'];s=record['statistics']
            assert np.array_equal(at,last)
            legal_seen=np.sum((at[:2]-at[4:6])**2)<=36.+1e-12 and not blocked(at[4:6],at[:2],boxes)
            if bool(legal_seen)!=(obs['target'] is not None):errors.append(dict(key=row['key'],step=obs['step'],kind='sensor'))
            if obs['target'] is not None:assert np.array_equal(obs['target'],at[:2])
            assert np.array_equal(obs['position'],at[4:6]) and obs['step']==at[8] and obs['remaining']==80-at[8]
            particles,weights=own.update(Observation(obs['step'],np.asarray(obs['position']),None if obs['target'] is None else np.asarray(obs['target']),obs['remaining']))
            assert np.array_equal(particles,record['particles']) and np.array_equal(weights,record['weights'])
            assert weights.min()>=0 and abs(weights.sum()-1)<1e-12
            for a,b,limit in ((at[4:6],nxt[4:6],speed),(at[:2],nxt[:2],1.)):
                if not(point_legal(a,boxes) and point_legal(b,boxes) and np.max(np.abs(b-a))<=limit+2e-7 and not blocked(a,b,boxes)):
                    errors.append(dict(key=row['key'],step=obs['step'],kind='executed_motion'))
            if row['failure'] is None:
                assert np.allclose(nxt[4:6],at[4:6]+record['own_action'],atol=1e-10,rtol=0)
                assert np.allclose(nxt[:2],at[:2]+record['opponent_action'],atol=1e-10,rtol=0)
                assert nxt[8]==at[8]+1
            post=np.sum((nxt[:2]-nxt[4:6])**2)<=36.+1e-12 and not blocked(nxt[4:6],nxt[:2],boxes)
            if bool(post)!=record['post_seen']:errors.append(dict(key=row['key'],step=obs['step'],kind='post_sensor'))
            if post and first is None:first=i+1
            if record['capture']:assert nxt[9]<.5 and nxt[11]<.5 and np.linalg.norm(nxt[:2]-nxt[4:6])<1.
            if not record['own_legal'] or not record['opponent_legal']:errors.append(dict(key=row['key'],step=obs['step'],kind='attempted_motion'))
            if any(record['simulation_errors']):errors.append(dict(key=row['key'],step=obs['step'],kind='simulated_geometry',values=record['simulation_errors']))
            aborted=s.get('budget_aborted_simulations',0)
            assert sum(s['root_visits'])==s['simulations']
            assert sum(s['root_counts'])==s['simulations']+aborted
            assert sum(s['terminations'])==s['simulations']+aborted
            if row['method'] in ('T','T-R'):
                guard=boxes.copy();guard[:,:,0]-=1e-6;guard[:,:,1]+=1e-6
                legal_roots=sum(safe(np.asarray(obs['position']),a*speed,speed,guard,np.zeros(2,np.int64)) for a in COMPASS)
            else:legal_roots=s['root_candidates']
            visited=sum(v>0 for v in s['root_visits'])
            maximum_visits=max(s['root_visits']) if s['root_visits'] else 0
            hidden=obs['target'] is None
            frames.append(dict(key=row['key'],group=row['history_id'],family=row['family'],method=row['method'],step=obs['step'],hidden=hidden,
                own_ms=record['own_ms'],cycle_ms=record['cycle_ms'],simulations=s['simulations'],aborted=aborted,
                queries_ms=s.get('rollout_queries_ms',0.),hits=s.get('rollout_cache_hits',0),misses=s.get('rollout_cache_misses',0),
                unvisited_legal_roots=max(0,legal_roots-visited) if row['method']!='G' and hidden else None,
                legal_roots=int(legal_roots),visited_roots=visited,build_ms=s['build_ms'],update_ms=s['update_ms'],
                max_root_visits=maximum_visits,only_single_visits=bool(maximum_visits==1),
                stopped=bool(np.max(np.abs(record['own_action']))<1e-10),
                candidate_ms=s['candidate_ms'],search_ms=s['search_including_prepare_ms'],
                charged_initialization_ms=s['charged_initialization_ms'],allowed_act_ms=s['allowed_act_ms']))
            last=nxt
        if row['failure'] is None:assert first==row['first_rediscovery']
        if trace:
            first_record=trace[0];identity=(row['history_id'],row['world'])
            equal=(initial,payload['task']['seed'],payload['task']['opponent_seed'],payload['task']['environment_seed'],
                first_record['observation'],first_record['particles'],first_record['weights'],first_record['opponent_raw'])
            if identity in pairs:assert equal==pairs[identity]
            else:pairs[identity]=equal
        episodes.append(dict(key=row['key'],group=row['history_id'],family=row['family'],method=row['method'],reason=row['reason'],
            first_rediscovery=row['first_rediscovery'],capture=row['capture'],
            failure_class=None if row['capture'] else 'engineering_failure' if row['failure'] else
                'not_rediscovered' if first is None else 'late_rediscovery' if first>row['remaining_at_handoff']-8 else 'rediscovered_not_captured'))
        if (n+1)%20==0:print(json.dumps(dict(stage='audit_'+stage,completed=n+1,total=len(rows))),flush=True)
    summary={}
    for method in METHODS:
        fs=[f for f in frames if f['method']==method];hidden=[f for f in fs if f['hidden']];rs=[r for r in rows if r['method']==method]
        summary[method]=dict(decisions=len(fs),hidden_decisions=len(hidden),
            hidden_simulations_mean=float(np.mean([f['simulations'] for f in hidden])),hidden_simulations_min=min(f['simulations'] for f in hidden),
            hidden_zero_complete_decisions=sum(f['simulations']==0 for f in hidden),
            budget_abort_calls=sum(f['aborted'] for f in fs),
            hidden_incomplete_root_pools=sum((f['unvisited_legal_roots'] or 0)>0 for f in hidden),
            hidden_unvisited_roots_mean=float(np.mean([f['unvisited_legal_roots'] or 0 for f in hidden])),
            hidden_only_single_visit_decisions=sum(f['only_single_visits'] for f in hidden),
            hidden_stops=sum(f['stopped'] for f in hidden),
            single_visit_and_stop=sum(f['only_single_visits'] and f['stopped'] for f in hidden),
            cache_hits=sum(f['hits'] for f in fs),cache_misses=sum(f['misses'] for f in fs),geometry_seconds=sum(f['queries_ms'] for f in fs)/1000.,
            query_share_of_decision_time=sum(f['queries_ms'] for f in fs)/max(1e-9,sum(f['own_ms'] for f in fs)),
            own_timing=quantiles([f['own_ms'] for f in fs]),ordinary_hidden_timing=quantiles([f['own_ms'] for f in hidden if not f['build_ms']]),
            components={k:quantiles([f[k] for f in hidden]) for k in ('update_ms','candidate_ms','search_ms','queries_ms')},
            over_200ms=sum(f['own_ms']>200 for f in fs),first_charged_over_200ms=sum(r['cold_start_first_decision_ms']>200 for r in rs),
            first_charged_timing=quantiles([r['cold_start_first_decision_ms'] for r in rs]),
            over_cycle_period=sum(f['cycle_ms']>1000 for f in fs))
    result=dict(episodes=len(rows),groups=len(set(r['history_id'] for r in rows)),frames=len(frames),errors=errors,
        checks=dict(all_public_history_estimator_replays_exact=True,all_weights_valid=True,all_initial_states_and_seeds_paired=True,
            first_actual_adversary_action_same=True,legal_simulations=all(not any(r['simulation_errors']) for r in rows),
            budget_abort_accounting_valid=True,independent_sensor_and_motion=not errors),
        summary=summary,episode_failure_diagnostics=episodes,frames_data=frames,
        simulated_transitions=sum(r['simulated_transitions'] for r in rows))
    save(OUT/stage/'postrun_audit.json',result);return result


def main():
    assert (OUT/'decision.json').exists()
    config=json.loads((OUT/'experiment.json').read_text())
    assert all(sha(p)==h for p,h in (config['source_hashes']|config['original_sources_and_inputs']|config['protected_sources']).items())
    for stage in ('development','confirmation'):audit(stage)
    save(OUT/'postrun_integrity.json',dict(passed=True,source_and_data_hashes_valid=True,original_algorithms_unchanged=True,
        frozen_D_valid=bool(loader.verify()),no_training=True,no_deployment_change=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    import run_rollout_replan_experiment as configuration
    configuration.OUT=OUT
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
