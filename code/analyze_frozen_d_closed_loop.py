"""Post-run audit and diagnostics; never alters frozen actors or outcomes.

Use only after the serial timing process exits. Outcome diagnosis is
descriptive and does not tune the confirmation bank or its controller.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import collections,fcntl,hashlib,json
from pathlib import Path
import numpy as np
from run_frozen_d_closed_loop import (ROOT,OUT,OLD,METHODS,sha,save,tasks,inputs,
    legal_action,original_model,statistics)
from solvers.frozen_d_search_interface import loader,Observation
from estimator_audit_geometry import blocked,point_legal
from audit_frozen_d_search_interface import quantiles


def positive_intervals(values,seed=81120261026):
    """History-block paired interval for descriptive secondary metrics."""
    values=np.asarray(values,float)
    draws=np.random.default_rng(seed).integers(0,len(values),(10000,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist())


def audit_stage(stage,replay_estimator=True):
    rows=json.loads((OUT/stage/'results.json').read_text())
    requests=tasks(stage)
    mapping={r['key']:r for r in rows}
    assert len(rows)==len(requests)==len(mapping)
    frames=[];episodes=[];errors=[];common={};groups=collections.defaultdict(list)
    for number,task in enumerate(requests):
        row=mapping[task['key']]
        assert sha(row['episode'])==row['episode_sha256']
        payload=json.loads(Path(row['episode']).read_text())
        assert payload['task']==task
        trace=payload['trace']
        assert hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest()==row['trajectory_sha256']
        packet,env=inputs(task);boxes=np.array(packet['map']);speed=packet['motion']['pursuer_axis_limit']
        world=next(w for w in env['worlds'] if w['id']==task['world'])
        assert payload['initial_state']==world['handoff_full_state']
        if replay_estimator:
            estimator=loader.load_estimator(boxes,task['seed'])
            for item in packet['history'][:-1]:
                estimator.update(Observation(item['step'],np.array(item['position']),
                    None if item['target'] is None else np.array(item['target']),item['remaining']))
        last=np.array(payload['initial_state']);all_legal=True;first_seen=None
        own_path=0.;other_path=0.;stops=0;repeat_states=collections.Counter();reversals=0
        nominal_budget=0
        support_masses=[]
        for i,frame in enumerate(trace):
            at=np.array(frame['state']);nxt=np.array(frame['next_state'])
            action=np.array(frame['own_action']);other=np.array(frame['opponent_action'])
            obs=frame['observation'];particles=np.array(frame['particles']);weights=np.array(frame['weights'])
            assert np.array_equal(at,last)
            assert obs['step']==at[8] and obs['remaining']==80-at[8]
            assert np.array_equal(obs['position'],at[4:6])
            # Sensor oracle independent of the runtime slab algorithm.
            seen=np.sum((at[:2]-at[4:6])**2)<=36.+1e-12 and not blocked(at[4:6],at[:2],boxes)
            if seen!=(obs['target'] is not None):errors.append(dict(key=task['key'],step=obs['step'],kind='observation_sensor'))
            if seen and not np.array_equal(obs['target'],at[:2]):errors.append(dict(key=task['key'],step=obs['step'],kind='measurement_truth'))
            lawful=legal_action(at[4:6],action,speed,boxes) and legal_action(at[:2],other,1.,boxes)
            all_legal=all_legal and lawful
            assert frame['own_legal'] and frame['opponent_legal'] and lawful
            if not np.allclose(nxt[4:6],at[4:6]+action,atol=1e-10,rtol=0):errors.append(dict(key=task['key'],kind='own_transition'))
            if not np.allclose(nxt[:2],at[:2]+other,atol=1e-10,rtol=0):errors.append(dict(key=task['key'],kind='target_transition'))
            assert nxt[8]==at[8]+1
            assert np.all(np.isfinite(weights)) and weights.shape==(64,) and weights.min()>=0 and abs(weights.sum()-1)<1e-12
            if replay_estimator:
                actual,importance=estimator.update(Observation(obs['step'],np.array(obs['position']),
                    None if obs['target'] is None else np.array(obs['target']),obs['remaining']))
                assert np.array_equal(actual,particles) and np.array_equal(importance,weights)
            post_seen=np.sum((nxt[:2]-nxt[4:6])**2)<=36.+1e-12 and not blocked(nxt[4:6],nxt[:2],boxes)
            if bool(post_seen)!=frame['post_seen']:errors.append(dict(key=task['key'],step=obs['step'],kind='post_sensor'))
            if post_seen and first_seen is None:first_seen=i+1
            if frame['capture']:
                assert nxt[9]<.5 and nxt[11]<.5 and np.linalg.norm(nxt[:2]-nxt[4:6])<1 and all_legal
            stats=frame['statistics']
            one_metre=float(weights[np.linalg.norm(particles-at[:2],axis=1)<=1.].sum())
            if obs['target'] is None:support_masses.append(one_metre)
            assert stats['simulations']<=1000 and stats['planning_depth']==min(8,obs['remaining'])
            if seen:assert stats['simulations']==0
            if task['method']=='G':assert stats['simulations']==0
            hidden=obs['target'] is None
            own_path+=float(np.linalg.norm(action));other_path+=float(np.linalg.norm(other))
            stops+=int(np.max(np.abs(action))<1e-10)
            repeat_states[tuple(np.round(at[4:6],5))]+=1
            if i and np.dot(action,np.array(trace[i-1]['own_action']))<-.5*speed*speed:reversals+=1
            if hidden and stats['simulations'] and stats['simulations']<1000:nominal_budget+=1
            region='visible' if seen else 'rebuild' if stats['build_ms']>0 else 'ordinary_hidden'
            frames.append(dict(key=task['key'],group=row['history_id'],method=row['method'],family=row['family'],
                step=obs['step'],region=region,own_ms=frame['own_ms'],cycle_ms=frame['cycle_ms'],
                **{k:stats[k] for k in ('update_ms','build_ms','candidate_ms','search_including_prepare_ms','simulations')},
                hidden=hidden,covered=frame['nearest_particle_error']<=1.,post_seen=bool(post_seen),
                weighted_mass_1m=one_metre,
                stop=bool(np.max(np.abs(action))<1e-10),
                all_root_returns_zero=bool(stats['root_totals'] and max(stats['root_totals'])==0.),
                root_children=len(stats['root_visits']),root_candidates=stats['root_candidates']))
            last=nxt
        assert first_seen==row['first_rediscovery']
        assert row['rediscovered']==int(first_seen is not None)
        assert np.array_equal(payload['final_state'],last)
        if row['capture']:assert trace[-1]['capture'] and row['failure'] is None
        if not trace or row['failure'] is not None:
            errors.append(dict(key=task['key'],kind='recorded_failure',reason=row['reason']))
        identity=(row['history_id'],row['world'])
        first=trace[0]
        # Actual adversary's first action must not depend on own current action.
        public_identity=(payload['initial_state'],task['seed'],task['opponent_seed'],task['environment_seed'],
            first['observation'],first['particles'],first['weights'],first['opponent_raw'])
        if identity in common:assert common[identity]==public_identity
        else:common[identity]=public_identity
        extra=dict(own_path_length=own_path,opponent_path_length=other_path,stop_steps=stops,
            repeated_pose_max=max(repeat_states.values()) if repeat_states else 0,
            reversal_steps=reversals,search_at_cap_or_time=nominal_budget,
            first_rediscovery=row['first_rediscovery'],lost_again=row['lost_again'],
            post_discovery_steps=None if first_seen is None else row['executed_steps']-first_seen)
        extra.update(hidden_mass_1m_mean=float(np.mean(support_masses)) if support_masses else None,
            failure_type=None if row['capture'] else 'engineering_failure' if row['failure'] is not None else
                'not_rediscovered' if first_seen is None else
                'rediscovered_with_less_than_8_steps_left' if first_seen>row['remaining_at_handoff']-8 else
                'rediscovered_but_not_captured_by_horizon')
        episodes.append(dict(key=row['key'],group=row['history_id'],world=row['world'],family=row['family'],method=row['method'],**extra))
        groups[identity].append(row)
        if (number+1)%12==0:print(json.dumps(dict(stage='postrun_audit_'+stage,completed=number+1,total=len(requests))),flush=True)
    summary={}
    for method in METHODS:
        fs=[f for f in frames if f['method']==method];rs=[r for r in rows if r['method']==method]
        summary[method]=dict(
            regions={kind:{field:quantiles([f[field] for f in fs if f['region']==kind]) for field in
                ('own_ms','cycle_ms','update_ms','build_ms','candidate_ms','search_including_prepare_ms')}
                for kind in ('visible','ordinary_hidden','rebuild') if any(f['region']==kind for f in fs)},
            decisions=len(fs),over_50ms=sum(f['own_ms']>50 for f in fs),over_200ms=sum(f['own_ms']>200 for f in fs),
            over_high_level_period=sum(f['cycle_ms']>1000 for f in fs),
            cold_start_over_200ms=sum(r['cold_start_first_decision_ms']>200 for r in rs),
            cold_start_over_1000ms=sum(r['cold_start_first_decision_ms']>1000 for r in rs),
            initialize_static=quantiles([r['initialization']['static_setup_ms'] for r in rs]),
            initialize_history=quantiles([r['initialization']['history_initialization_ms'] for r in rs]),
            initial_opponent=quantiles([r['initialization']['opponent_setup_ms'] for r in rs]),
            hidden_simulations_mean=float(np.mean([f['simulations'] for f in fs if f['hidden']])),
            hidden_at_simulation_cap=sum(f['simulations']==1000 for f in fs if f['hidden']),
            hidden_zero_simulations=sum(f['simulations']==0 for f in fs if f['hidden']),
            hidden_stop_steps=sum(f['stop'] for f in fs if f['hidden']),
            hidden_zero_root_return_decisions=sum(f['all_root_returns_zero'] for f in fs if f['hidden']),
            hidden_decisions=sum(f['hidden'] for f in fs),
            total_controller_seconds=sum(f['own_ms'] for f in fs)/1000.,
            hidden_coverage_history_equal=float(np.mean([r['hidden_covered']/max(1,r['hidden_steps']) for r in rs])))
    index={(r['history_id'],r['world'],r['method']):r for r in rows}
    descriptions={};history_ids=sorted({r['history_id'] for r in rows})
    marginal_intervals={m:{metric:positive_intervals([
        np.mean([index[(h,w,m)][metric] for w in ('A','B')]) for h in history_ids])
        for metric in ('capture','rediscovered','completion_cost')} for m in METHODS}
    for a,b in (('T','G'),('T+Geo','T'),('T+Geo','G')):
        detail=[];rediscovery_delta=[];post_delta=[]
        for h in history_ids:
            for w in ('A','B'):
                x,y=index[(h,w,a)],index[(h,w,b)]
                detail.append(dict(group=h,world=w,family=x['family'],a=a,b=b,
                    cost_delta=x['completion_cost']-y['completion_cost'],
                    rediscovery_delta=None if x['first_rediscovery'] is None or y['first_rediscovery'] is None else x['first_rediscovery']-y['first_rediscovery'],
                    a_capture=x['capture'],b_capture=y['capture'],a_key=x['key'],b_key=y['key']))
            if all(index[(h,w,m)]['first_rediscovery'] is not None for w in ('A','B') for m in (a,b)):
                rediscovery_delta.append(np.mean([index[(h,w,a)]['first_rediscovery']-index[(h,w,b)]['first_rediscovery'] for w in ('A','B')]))
                post_delta.append(np.mean([(index[(h,w,a)]['executed_steps']-index[(h,w,a)]['first_rediscovery'])-
                    (index[(h,w,b)]['executed_steps']-index[(h,w,b)]['first_rediscovery']) for w in ('A','B')]))
        descriptions[a+'-'+b]=dict(
            slower=sum(v['cost_delta']>0 for v in detail),faster=sum(v['cost_delta']<0 for v in detail),ties=sum(v['cost_delta']==0 for v in detail),
            first_rediscovery_delta=positive_intervals(rediscovery_delta) if rediscovery_delta else None,
            post_discovery_steps_delta=positive_intervals(post_delta) if post_delta else None,
            fastest=sorted(detail,key=lambda r:r['cost_delta'])[:3],slowest=sorted(detail,key=lambda r:r['cost_delta'],reverse=True)[:3])
    families={f:{m:dict(games=len([r for r in rows if r['family']==f and r['method']==m]),
        first_rediscovery_mean=float(np.mean([r['first_rediscovery'] for r in rows if r['family']==f and r['method']==m and r['first_rediscovery'] is not None])))
        for m in METHODS} for f in ('long_wall','wall_corner','multi_route')}
    # Bootstrap cannot create discordance absent from the empirical sample.
    # Report an exact zero-event bound on history-level discordance as a
    # diagnostic; it is NOT a capture-difference equivalence confidence band.
    zero_event_bound=1.-.025**(1./len(history_ids))
    result=dict(stage=stage,groups=len(history_ids),episodes=len(rows),decisions=len(frames),
        checks=dict(episode_hashes=True,initial_states_and_seeds_paired=True,first_adversary_action_identical=True,
            public_observation_independent_sensor=not any(e['kind'] in ('observation_sensor','post_sensor','measurement_truth') for e in errors),
            exact_estimator_public_history_replay=replay_estimator,all_actions_lawful=True,
            joint_motion_checked=True,weights_normalized=True,capture_independently_legal=True),
        errors=errors,summary=summary,descriptive_comparisons=descriptions,by_family=families,
        marginal_history_bootstrap_intervals=marginal_intervals,
        episode_diagnostics=episodes,frames=frames,zero_discordance_one_sided_97_5_bound=zero_event_bound,
        zero_discordance_bound_scope='probability that a public-history block has any discordant method outcomes; not an equivalence margin')
    save(OUT/stage/'postrun_audit.json',result)
    return result


def main():
    config=json.loads((OUT/'experiment.json').read_text())
    assert (OUT/'decision.json').exists(),'Wait until the serial timing run is complete'
    assert all(sha(p)==h for p,h in (config['sources_and_inputs']|config['protected_before']).items())
    loader.verify()
    for stage in ('development','confirmation'):
        audit_stage(stage)
    save(OUT/'postrun_integrity.json',dict(passed=True,source_and_input_hashes_valid=True,
        protected_deployment_files_unchanged=True,frozen_D_verified=bool(loader.verify()),
        no_retraining=True,no_policy_or_estimator_changes_after_confirmation=True))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
