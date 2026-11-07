import copy
import json
import numpy as np
import pytest
from run_e45z_confirmation import select_maps,reference_jobs,evaluate,NAMESPACE


def test_map_selection_excludes_prior_geometry_as_well_as_id(tmp_path):
    def row(name,boxes):
        p=tmp_path/f'{name}.json';p.write_text(json.dumps(dict(map=boxes)))
        return dict(id=name,path=str(p))
    old=row('old',[[[0,1],[0,1]]]);same=row('different-name',[[[0,1],[0,1]]])
    fresh=row('fresh',[[[2,3],[2,3]]])
    assert [r['id'] for r in select_maps([same,fresh],[old])]==['fresh']


def test_reference_streams_are_paired_and_identical_actions_are_deduplicated():
    rows=[dict(root_id='a',chosen_D_proxy_action=1,chosen_Root_proxy_action=1),
          dict(root_id='b',chosen_D_proxy_action=0,chosen_Root_proxy_action=3)]
    jobs=reference_jobs(rows,16)
    assert len(jobs)==48 and all(r['namespace']==NAMESPACE for r in jobs)
    assert {r['stream'] for r in jobs if r['root_id']=='b' and r['candidate']==0}==set(range(16))
    assert {r['stream'] for r in jobs if r['root_id']=='b' and r['candidate']==3}==set(range(16))


def fixture_data(gains,streams=2):
    selections=[];rows=[]
    for i,gain in enumerate(gains):
        rid=f'root-{i}'
        selections.append(dict(root_id=rid,group=f'map-{i}',family=f'family-{i%3}',stratum='public_imminent_first',
            chosen_D_proxy_action=0,chosen_Root_proxy_action=1))
        for candidate in (0,1):
            for stream in range(streams):
                rows.append(dict(root_id=rid,candidate=candidate,stream=stream,utility=.1+candidate*gain,
                    double_capture=False,initial_world_sha256=f'{rid}-{stream}'))
    return selections,rows


def test_evaluation_rejects_world_mismatch_between_actions():
    selections,rows=fixture_data([.1])
    rows[-1]['initial_world_sha256']='wrong-world'
    with pytest.raises(RuntimeError,match='Unpaired'):evaluate(selections,rows,2)


def test_map_weighting_does_not_count_source_repeats_as_new_maps():
    selections,rows=fixture_data([0.,1.])
    # A second public root on map zero should not double that map's weight.
    duplicate=copy.deepcopy(selections[0]);duplicate['root_id']='extra'
    selections.append(duplicate)
    rows.extend([r|dict(root_id='extra') for r in copy.deepcopy(rows[:4])])
    result=evaluate(selections,rows,2)['summaries'][1]
    assert result['maps']==2 and result['roots']==3
    assert np.isclose(result['metrics']['utility_gain']['mean'],.5)


def test_progression_gate_requires_cross_map_and_family_evidence():
    selections,rows=fixture_data([.1]*6)
    assert evaluate(selections,rows,2)['advance_to_one_cheap_predictor_pilot']
    for row in selections:row['family']='one-family'
    assert not evaluate(selections,rows,2)['advance_to_one_cheap_predictor_pilot']
    selections,rows=fixture_data([0.,0.,0.,0.,0.,.1])
    assert not evaluate(selections,rows,2)['advance_to_one_cheap_predictor_pilot']


def test_equal_selected_action_has_zero_paired_gain():
    selections,rows=fixture_data([.1])
    selections[0]['chosen_Root_proxy_action']=0
    result=evaluate(selections,rows[:2],2)
    primary=result['summaries'][1]['metrics']['utility_gain']
    assert primary['mean']==0 and primary['descriptive_95_interval']==[0.,0.]
