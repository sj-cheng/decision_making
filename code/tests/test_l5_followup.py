"""Scientific interface checks for same-pool causal comparisons."""
import copy
import json
from pathlib import Path

import numpy as np
import tempfile
import unittest

import run_l5_followup as run
from learning import l5_followup as impl
from learning.e46_geometry_teacher import legal, prepare_public as legacy_public


def packet():
    run.configure(Path(tempfile.mkdtemp(prefix='l5-followup-acceptance-')))
    cfg = run.read(run.OLD/'experiment.json')
    entry = next(e for e in cfg['entries'] if e['split']=='development' and e['suite']=='ID')
    return run.load_pickle(run.OLD/'packets/development'/(entry['id']+'.pkl.gz'))[0]


def model_spec(arm, packet, seed=17):
    spec = dict(arm=arm,seed=packet['seed'],train_seed=seed)
    if arm in ('A3','A5'):
        spec['checkpoint'] = str(run.OLD/f'training/L5_{seed}_base/selected.npz')
    return spec


def test_legacy_fixed_simulation_is_replayable(packet):
    spec=dict(arm='POOL_LEARNED',seed=packet['seed'],checkpoint=model_spec('A3',packet)['checkpoint'])
    a,x,_=legacy_public(packet,spec,4)
    b,y,_=legacy_public(packet,spec,4)
    np.testing.assert_array_equal(x,y)
    assert a.last_stats['root_visits']==b.last_stats['root_visits']
    assert a.last_stats['simulations']==b.last_stats['simulations']==4


def test_all_orders_share_identical_public_root_candidates(packet):
    hashes=[]
    for arm in ('A1','A2','A3','A4','A5'):
        own,action,_=impl.prepare_public(packet,model_spec(arm,packet),4)
        hashes.append(own.last_stats['root_pool_sha256'])
        assert legal(own,own.root,action)
        assert own.last_stats['assignment_used'] is False
        assert own.last_stats['roles_used'] is False
    assert len(set(hashes))==1


def test_search_and_network_seed_cannot_change_root_menu(packet):
    menus=[]
    for seed,search in ((17,301),(29,601),(43,901)):
        spec=dict(model_spec('A3',packet,seed),search_seed=search)
        own,_,_=impl.prepare_public(packet,spec,4)
        menus.append(np.asarray(own.last_stats['root_pool_actions']))
    for other in menus[1:]:np.testing.assert_array_equal(menus[0],other)


def test_generation_does_not_consume_ordering_or_rollout_rng(packet):
    own,_,_=impl.prepare_public(packet,model_spec('A1',packet),0)
    before=copy.deepcopy(own.proposal_rng.bit_generator.state)
    world=copy.deepcopy(own.world_rng.bit_generator.state)
    own.deadline=float('inf');own.pool(own.root)
    assert own.proposal_rng.bit_generator.state==before
    assert own.world_rng.bit_generator.state==world


def test_direct_controls_return_rank_first_without_simulations(packet,arm):
    own,action,_=impl.prepare_public(packet,model_spec(arm,packet),16)
    assert own.last_stats['simulations']==0
    assert own.last_stats['joint_action_search'] is False
    np.testing.assert_array_equal(action,own.root.pool_actions[own.root.pool_order[0]])
    assert legal(own,own.root,action)


def test_injected_menu_exactly_reused_for_root_search(packet):
    menu=run.root_menu(packet)
    signatures=[]
    for arm in ('A1','A2','A3'):
        own,_,_=impl.prepare_public(packet,dict(model_spec(arm,packet),injected_root=menu.tolist()),4)
        np.testing.assert_array_equal(menu,own.root.pool_actions)
        signatures.append(own.last_stats['root_pool_sha256'])
    assert len(set(signatures))==1


def test_public_node_key_ignores_private_search_bookkeeping(packet):
    own,_,_=impl.prepare_public(packet,model_spec('A1',packet),0)
    node=copy.deepcopy(own.root)
    first=impl.public_node_key(node)
    if hasattr(node,'followup_public_key'):del node.followup_public_key
    node.n=100;node.visits=[33,67];node.totals=[-5.,5.]
    assert impl.public_node_key(node)==first


def test_registered_method_instances_and_protected_model_files():
    spec=run.specs()
    assert len(spec)==13
    assert sum(s['arm'] in ('A1','A2','A4','R1') for s in spec)==4
    assert len({s['checkpoint'] for s in spec if s.get('checkpoint')})==6
    for seed in run.SEEDS:
        record=run.read(run.OLD/f'training/L5_{seed}_base/result.json')
        assert run.sha(record['path'])==record['sha256']


class FollowupContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public_packet=packet()

    def test_legacy_replay(self):test_legacy_fixed_simulation_is_replayable(self.public_packet)
    def test_same_public_root(self):test_all_orders_share_identical_public_root_candidates(self.public_packet)
    def test_independent_seeds(self):test_search_and_network_seed_cannot_change_root_menu(self.public_packet)
    def test_independent_streams(self):test_generation_does_not_consume_ordering_or_rollout_rng(self.public_packet)
    def test_geometric_direct(self):test_direct_controls_return_rank_first_without_simulations(self.public_packet,'A4')
    def test_learned_direct(self):test_direct_controls_return_rank_first_without_simulations(self.public_packet,'A5')
    def test_exact_injection(self):test_injected_menu_exactly_reused_for_root_search(self.public_packet)
    def test_public_identity(self):test_public_node_key_ignores_private_search_bookkeeping(self.public_packet)
    def test_models_and_instance_counts(self):test_registered_method_instances_and_protected_model_files()

    def test_cpp_retirement_intervention_regression(self):
        fixture=run.read(Path(__file__).parent/'fixtures/l5_retirement_regression_20261107.json')
        path=Path(tempfile.mkdtemp(prefix='l5-retirement-regression-'))/'packet.pkl.gz'
        run.save_pickle(path,fixture['packet'])
        task=dict(packet=str(path),action=fixture['action'],opponent=fixture['opponent'],
                  posterior_world=fixture['posterior_world'],kind='intervention',stage='regression')
        result=run.intervention_job(task,dict(smoke=False))
        trace=run.load_pickle(result['trace'])
        assert trace and any(sum(r['observation']['active_e'])<2 for r in trace)
        assert result['decisions']==len(trace)

    def test_multilevel_calibration_continues_across_empty_queues(self):
        out=Path(tempfile.mkdtemp(prefix='l5-calibration-lifecycle-'))
        db=run.e46.connect(out)
        run.write(out/'execution_version_frozen.json',dict(test=True))
        run.write(out/'timing_serial_reference.json',dict(median=100.,p95=100.))
        for stage in ('P3_normal_profiles','P3_anomaly_replay'):
            run.e46.enqueue(db,dict(kind='profile',stage=stage,group='test'))
            db.execute('update jobs set status="done",result=? where stage=?',(json.dumps(dict(valid=True,result={})),stage))
        db.commit()
        cfg=dict(smoke=False,roots=[],development=[],methods=run.specs())
        levels=[]
        for expected in (1,2,4,8):
            gate=run.timing_gate(db,out,cfg,False)
            stage=gate['stage']
            assert gate['workers']==expected
            assert db.execute('select count(*) from jobs where stage=? and status="pending"',(stage,)).fetchone()[0]==expected
            payload=json.dumps(dict(valid=True,result=dict(rows=[dict(samples_ms=[100.,100.,100.])])) )
            db.execute('update jobs set status="done",result=? where stage=?',(payload,stage));db.commit()
            gate=run.timing_gate(db,out,cfg,False)
            levels.append(expected)
            if expected<8:
                assert db.execute('select count(*) from jobs where status="pending"').fetchone()[0]==0
                assert run.idle_next_action(db,gate)=='wait'
        assert levels==[1,2,4,8] and gate['passed']
        entry=dict(id='public_test',path='/unused/public_case.json',sha256='test',split='development',family='long_wall',suite='ID')
        cfg['development']=[entry]
        run.advance(db,out,cfg)
        assert db.execute('select count(*) from jobs where stage="P2_development_wall" and status="pending"').fetchone()[0]==52
        assert run.idle_next_action(db,gate)=='wait'
        db.close()

    def test_idle_calibration_does_not_hide_engineering_failure(self):
        out=Path(tempfile.mkdtemp(prefix='l5-calibration-failure-'));db=run.e46.connect(out)
        run.e46.enqueue(db,dict(kind='calibration',stage='P6a_calibration_0',group='test'))
        db.execute('update jobs set status="failed"');db.commit()
        assert run.idle_next_action(db,dict(passed=False))=='stopped_incomplete'
        db.close()


if __name__=='__main__':unittest.main()
