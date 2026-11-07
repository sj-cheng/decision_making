import copy
import numpy as np
from run_e45y_compute_allocation import (entropy_positions,decision_scores,query,
    threshold_mask,top_k_mask,probe,evaluate)
from test_e45x_event_diagnostic import actor,example
from learning.e45x_event_diagnostic import prepare_public,root_candidates


def test_entropy_merges_duplicate_particles_and_known_target_is_zero():
    assert entropy_positions(np.zeros((64,2)),np.ones(64)/64)==0
    points=np.r_[np.zeros((32,2)),np.ones((32,2))]
    assert np.isclose(entropy_positions(points,np.ones(64)/64),np.log(2))


def test_sampling_disagreement_is_zero_for_stable_ranking_and_positive_when_ambiguous():
    stable=np.tile([3.,1.,0.],(16,1))
    a=decision_scores(stable,17)
    assert a['decision_score']==0 and a['sampling_rank_disagreement']==0
    ambiguous=np.tile([[1.,0.],[0.,1.]],(8,1))
    b=decision_scores(ambiguous,17)
    assert b['decision_score']>0 and b['sampling_rank_disagreement']>0


def test_query_charges_signal_cost_without_reading_future_snapshots():
    record=dict(stop_index=4,snapshots=[dict(at_ms=0.,selected_candidate=4),
        dict(at_ms=40.,selected_candidate=0),dict(at_ms=80.,selected_candidate=1),dict(at_ms=180.,selected_candidate=2)])
    assert query(record,50)==0
    assert query(record,50,15)==4
    assert query(record,100,15)==1
    assert query(record,200,30)==1
    assert query(record,50,100)==4


def test_probe_does_not_consume_tree_rng_or_mutate_posterior(actor):
    root,case=example();prepared=prepare_public(root,case,actor)
    actions,_=root_candidates(root,prepared,actor);root['candidates']=actions.tolist()
    c=prepared.controller
    before=(copy.deepcopy(c.world_rng.bit_generator.state),copy.deepcopy(c.proposal_rng.bit_generator.state),c.rng.copy(),
            [w.copy() for w in c.root.joints])
    result=probe(c,root,0)
    assert result['signal_ms']>=result['entropy_ms']>=0
    assert before[0]==c.world_rng.bit_generator.state and before[1]==c.proposal_rng.bit_generator.state
    np.testing.assert_array_equal(before[2],c.rng)
    for a,b in zip(before[3],c.root.joints):np.testing.assert_array_equal(a,b)


def test_budget_allocators_have_exact_high_count_and_strict_threshold():
    assert top_k_mask([3,1,2],2).tolist()==[True,False,True]
    assert top_k_mask([1,1,1],1).tolist()==[True,False,False]
    assert not threshold_mask([0,0],0).any()


def synthetic_rows():
    rows=[];references={}
    for opponent in ('fixed_D_kernel','rootpuct'):
        for i,score in enumerate([0.,1.,2.]):
            rows.append(dict(split='train',opponent=opponent,signals=dict(decision_score=score)))
        for i,score in enumerate([2.,0.]):
            rid=f'{opponent}-{i}'
            rows.append(dict(root_id=rid,group=f'map{i}',split='development',repeat=0,opponent=opponent,
                stratum='handover',stop_index=4,signals=dict(decision_score=score,entropy=1-score,signal_ms=5.,entropy_ms=.1),
                snapshots=[dict(at_ms=0.,selected_candidate=4),dict(at_ms=40.,selected_candidate=0),
                    dict(at_ms=100.,selected_candidate=1),dict(at_ms=180.,selected_candidate=2)]))
            references[rid]={str(a):[dict(stream=s,utility=a/10.+s/100.) for s in range(8)] for a in range(6)}
    return rows,references


def test_outcome_labels_do_not_change_primary_gate_and_all_controls_match_budget():
    rows,references=synthetic_rows();a=evaluate(rows,references)
    changed=copy.deepcopy(references)
    for cells in changed.values():
        for key,values in cells.items():
            for value in values:value['utility']=1.-float(key)/10.
    b=evaluate(rows,changed)
    assert a['thresholds']==b['thresholds']
    assert [r['high_assigned'] for r in a['allocations']]==[r['high_assigned'] for r in b['allocations']]
    assert all(r['charged_mean_ms']==125. for r in a['policy_statistics'])
