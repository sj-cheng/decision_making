"""Bank acceptance tests independent of any controller or method outcome."""
import copy
from collections import Counter

import numpy as np
import pytest

from e49_bank import (ID_FAMILIES, OOD_FAMILIES, TIERS, allocation, construct,
                      geometry_audit, geometry_key, make_observation, replay)


def test_registered_allocation_and_id_only_development():
    totals = Counter()
    for split in ('train', 'validation', 'development', 'test'):
        for tier in TIERS:
            counts = allocation(split, tier)
            totals[split] += sum(counts.values())
            if split != 'test':
                assert set(counts) == set(ID_FAMILIES)
                assert max(counts.values())-min(counts.values()) <= 1
            else:
                assert counts == dict.fromkeys((*ID_FAMILIES, *OOD_FAMILIES), 4)
    assert totals == dict(train=240, validation=24, development=24, test=84)


def test_complete_rectangle_geometry_and_free_space_connectivity():
    assert geometry_audit([[[-1, 1], [-8, 8]]])['connected']
    with pytest.raises(ValueError, match='interiors overlap'):
        geometry_audit([[[-1, 1], [-1, 1]], [[0, 2], [0, 2]]])
    with pytest.raises(ValueError, match='boundary'):
        geometry_audit([[[-1, 1], [-10, 8]]])
    ring = [[[-2, -1], [-2, 2]], [[1, 2], [-2, 2]],
            [[-1, 1], [-2, -1]], [[-1, 1], [1, 2]]]
    with pytest.raises(ValueError, match='Disconnected'):
        geometry_audit(ring)


def test_dedup_translation_rotation_and_order():
    boxes = np.array([[[-1, 1], [-7, 7]], [[3, 4], [-5, 5]]], float)
    rotated = np.stack((-boxes[:, 1, ::-1], boxes[:, 0]), axis=1)
    assert geometry_key(boxes) == geometry_key(rotated[::-1] + np.array([3., 3.])[None, :, None])
    changed = boxes.copy(); changed[1, 0, 1] += .01
    assert geometry_key(changed) != geometry_key(boxes)
    # Joined wall pieces put bounds at the canonical origin; signed zero must
    # not distinguish 180-degree rotations.
    joined = np.array([[[-1, 1], [-6, 6]], [[1, 3], [4.6, 6]]], float)
    assert geometry_key(joined) == geometry_key(-joined[:, :, ::-1])


@pytest.fixture(scope='module', params=TIERS)
def paired_case(request):
    return construct('long_wall', 0, 'test', request.param, horizon=120)


def test_rebuilt_paired_history_and_motion_law(paired_case):
    case = paired_case
    if case['obstacle_tier'] == 'sparse': assert len(case['map']) <= 8
    else: assert len(case['map']) == int(case['obstacle_tier'])
    assert case['speed'] == 1. and case['evader_speed'] == 1.5
    assert [world['id'] for world in case['worlds']] == ['AB', 'BA']
    assert case['worlds'][0]['initial'][:4] != case['worlds'][1]['initial'][:4]
    assert case['history'][-1]['step'] == 0
    assert case['history'][-1]['remaining'] == 120
    assert all(any(q is not None for q in row) for row in case['history'][0]['reports'])
    assert all(q is None for row in case['history'][-1]['reports'] for q in row)
    for world in case['worlds']:
        assert world['history'] == case['history']
        states = np.asarray(world['scripted_states'])
        assert np.all(states[:, 9:13] == 1)
        assert np.abs(np.diff(states[:, 4:8], axis=0)).max() <= 1.+1e-9
        assert np.abs(np.diff(states[:, :4], axis=0)).max() <= 1.5+1e-9
        assert [make_observation(s, case['map'], 120) for s in states] == case['history']


def test_replay_rejects_illegal_script_instead_of_speed_relabel(paired_case):
    case = copy.deepcopy(paired_case)
    case['worlds'][0]['scripted_states'][1][4] += 3.
    with pytest.raises(ValueError, match='role speed'):
        replay(case, 120)
