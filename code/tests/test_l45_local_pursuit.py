"""Acceptance checks for the local pursuit protocol and recoverable runner."""
import argparse
import copy
from pathlib import Path
import tempfile
import numpy as np
from solvers.local_pursuit import (Controller,PublicMap,Observation,legal,clear,visible,safe,
                                  modal,guided,recover,move,warmup)


def run_checks(output):
    import run_l45_local_pursuit as run
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    completed=[]
    count=np.zeros(2,np.int64)
    plain=PublicMap([],1.1)
    assert visible(np.array([0.,0.]),np.array([6.,0.]),plain.boxes,count)
    assert not visible(np.array([0.,0.]),np.array([6.000001,0.]),plain.boxes,count)
    wall=PublicMap(run.MAPS['corner'],1.1)
    assert not visible(np.array([-2.,0.]),np.array([2.,0.]),wall.boxes,count)
    assert not clear(np.array([-2.,0.]),np.array([2.,0.]),wall.boxes,count)
    assert not safe(np.array([-2.,0.]),np.array([4.,0.]),5.,wall.boxes,count)
    completed.append('range_boundary_obstacle_occlusion_and_swept_collision')
    # Two different hidden positions yield the same empty sensor result.
    for method in ('L','G','T','V'):
        a,b=[Controller(plain,method,123,simulations=3,deadline_ms=0) for _ in range(2)]
        initial=plain.observe(0,np.array([0.,0.]),np.array([4.,0.]))
        a.act(initial,fixed_simulations=3);b.act(copy.deepcopy(initial),fixed_simulations=3)
        obs_a=plain.observe(1,np.array([-1.,0.]),np.array([5.1,0.]))
        obs_b=plain.observe(1,np.array([-1.,0.]),np.array([5.5,.2]))
        assert obs_a.target is None and obs_b.target is None
        aa=a.act(obs_a,fixed_simulations=3);ab=b.act(obs_b,fixed_simulations=3)
        assert np.array_equal(aa,ab) and np.array_equal(a.particles,b.particles)
        # Diagnostics do not advance either decision stream.
        snap=a.snapshot();_ = copy.deepcopy(a.last_stats)
        assert np.array_equal(snap['rng'],a.rng) and np.array_equal(snap['filter_rng'],a.filter_rng)
    completed.append('hidden_truth_invariance_and_nonconsuming_logging')
    controllers=[Controller(plain,m,78,simulations=3,deadline_ms=0) for m in ('G','T','V')]
    history=[plain.observe(0,np.array([0.,0.]),np.array([4.,0.])),Observation(1,np.array([-1.,0.]),None,79),
             plain.observe(2,np.array([0.,0.]),np.array([5.,.1]))]
    for obs in history:
        for controller in controllers:controller.act(copy.deepcopy(obs),fixed_simulations=3)
        assert all(np.array_equal(controllers[0].particles,c.particles) for c in controllers)
    assert np.all(controllers[0].particles==[5.,.1])
    completed.append('common_history_identical_estimates_and_reacquisition')
    # Recovery uses only the last measurement and all empty-observer positions.
    recovery=Controller(wall,'G',63,deadline_ms=0)
    recovery.act(Observation(0,np.array([2.,4.]),np.array([1.6,3.4]),80))
    recovery.particles[:]=[-2.,3.4]  # deliberately degenerate visible support
    recovery.act(Observation(1,np.array([-2.,3.4]),None,79))
    assert recovery.recoveries==1
    assert all(legal(q,wall.boxes,count) and not visible(np.array([-2.,3.4]),q,wall.boxes,count) for q in recovery.particles)
    assert np.all(np.abs(recovery.particles-np.array([1.6,3.4]))<=1.+1e-9)
    completed.append('degenerate_particle_recovery_without_truth')
    # Distinct modes must select a particle, not the global mean in the wall.
    bimodal=np.tile(np.array([[-2.,0.],[2.,0.]]),(32,1))
    assert any(np.array_equal(modal(bimodal),q) for q in bimodal)
    completed.append('modal_grid_center_sample_avoids_wall_mean')
    # Construct a genuine first-turn exit: target mass behind the rectangle,
    # with pursuer already near the first corner and able to arrive sooner.
    found=False
    for at in (np.array([-1.6,3.4]),np.array([-1.6,3.6]),np.array([-2.,3.4])):
        particles=np.tile([2.,0.],(64,1))
        actions,sources,n=guided(at,particles,1.1,wall.vertices,wall.dist,wall.edist,wall.boxes,count,True)
        if 2 in sources[:n]:
            found=True
            assert all(safe(at,a,1.1,wall.boxes,count) for a in actions[:n])
            break
    assert found,'exit branch was not exercised'
    completed.append('reobserve_and_reachable_exit_candidates')
    # If extra intentions are unavailable, root candidate lists and entire
    # fixed-count searches coincide exactly, including non-root behavior.
    obs=plain.observe(0,np.array([-1.,0.]),np.array([2.,0.]))
    t,v=[Controller(plain,m,54,simulations=5,deadline_ms=0) for m in ('T','V')]
    assert np.array_equal(t.act(copy.deepcopy(obs),5),v.act(copy.deepcopy(obs),5))
    assert t.last_stats['root']==v.last_stats['root']
    completed.append('TV_identical_when_root_guides_identical')
    # Full environment: invisible target is active; joint transitions equal
    # the original Python swept simulator and preserve its reward semantics.
    spec=run.scenario(6002026102497,'corner',1.1)
    item=dict(key='acceptance',spec=spec,method='T',stream=0)
    p,state,own,other=run.setup(item,True)
    native=other._problem_wrapper_for(p)
    pr=np.array([-3.,0.]);en=np.array([3.,0.])
    state[p.state_idxs[2],0]=pr;state[p.state_idxs[0],0]=en
    assert run.observation(own,p,state).target is None and not p.is_terminal(state)
    state=np.asarray(spec['initial_state'],float)
    rng=np.random.default_rng(60020261024)
    for i in range(24):
        action=np.zeros((8,1));action[:2,0]=rng.uniform(-1,1,2);action[4:6,0]=rng.uniform(-1.1,1.1,2)
        python=p.step(state,action,1.)
        actual=np.asarray(native.step(state,action,1.)).reshape(-1,1)
        # Historical memory encoders use float precision; physical positions,
        # active flags and time must match within native float precision.
        assert np.allclose(python[:29],actual[:29],atol=3e-5)
        expected=p.transition_reward(state,python)
        reward=np.asarray(native.transition_reward(state,action,actual)).reshape(-1,1)
        assert np.allclose(expected,reward,atol=3e-6)
    completed.append('native_vs_python_environment_and_reward;invisible_is_not_capture')
    # Interrupt after one actual decision, reload complete checkpoint and
    # compare with uninterrupted execution (smoke has fixed count 3).
    with tempfile.TemporaryDirectory(prefix='l45_resume_',dir=output) as folder:
        args=argparse.Namespace(output=Path(folder))
        interruption=run.episode(args,item,True,stop_after=1)
        assert interruption['interrupted']
        resumed=run.episode(args,item,True)
        uninterrupted=run.episode(args,item,True,checkpoint=False)
        assert resumed['trajectory_sha256']==uninterrupted['trajectory_sha256']
        assert np.array_equal(resumed['_record']['final']['controller']['particles'],uninterrupted['_record']['final']['controller']['particles'])
        run.commit(args,resumed)
        assert run.cached(args,item)['trajectory_sha256']==uninterrupted['trajectory_sha256']
        # A committed result is returned without rerunning the episode.
        before=run.sha(Path(run.cached(args,item)['episode_record']['path']))
        assert before==run.sha(Path(run.cached(args,item)['episode_record']['path']))
    completed.append('interrupted_episode_exact_resume_and_committed_cache_reuse')
    return dict(passed=True,checks=completed)


def test_acceptance(tmp_path):
    warmup()
    assert run_checks(tmp_path)['passed']
