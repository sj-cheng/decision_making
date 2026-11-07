import json
from pathlib import Path
import numpy as np

from learning.e45w_support_recovery import TeamBelief, finer_posterior
from learning.e45g_ppo import TeamBelief as Original
from solvers.e45_team_adapter import team_hidden_mask
from solvers.local_pursuit import clear
from run_e45w_repaired_training import budget, CAP, BLOCK

FIXTURE = Path(__file__).parent / 'fixtures/e45w_fine_support_public.json'


def test_real_failed_history_replays_and_normal_prefix_is_identical():
    data = json.loads(FIXTURE.read_text())
    repaired = TeamBelief(data['boxes'], data['seed'])
    original = Original(data['boxes'], data['seed'])
    for obs in data['history'][:-1]:
        repaired.update(obs); original.update(obs)
        assert repaired.signature() == original.signature()
    failed = False
    try:
        original.update(data['history'][-1])
    except RuntimeError as exc:
        failed = str(exc).startswith('G refinement also exhausted;')
    assert failed, 'Fixture must reproduce the original scientific failure'
    repaired.update(data['history'][-1])
    assert any(r.get('spacing') == .125 for r in repaired.refinements)
    assert repaired.step == data['history'][-1]['step']
    for j, e in enumerate(repaired.estimators):
        if data['history'][-1]['active_e'][j]:
            assert np.isfinite(e.particle_weights).all()
            assert np.isclose(e.particle_weights.sum(), 1)


def test_recovered_support_has_legal_full_history_witnesses():
    data = json.loads(FIXTURE.read_text()); history = data['history']; boxes = np.asarray(data['boxes'])
    (graph, weights, parents, _), spacing = finer_posterior(boxes, history, 1)
    assert spacing == .125
    start, last = [(k, np.asarray(q)) for k, o in enumerate(history)
                   for i, q in enumerate(o['reports'][1]) if o['active_p'][i] and q is not None][-1]
    points = graph[0]
    for end in np.flatnonzero(weights.sum(1) > 0):
        current = int(end)
        for t in range(len(parents)-1, -1, -1):
            obs = history[start+1+t]
            assert team_hidden_mask(points[current:current+1], np.asarray(obs['positions']),
                                    np.asarray(obs['active_p']), boxes).all()
            parent = int(parents[t][current]); assert parent >= 0
            assert np.max(np.abs(points[current]-points[parent])) <= 1+1e-10
            assert clear(points[current], points[parent], boxes, np.zeros(2, np.int64))
            current = parent
        np.testing.assert_allclose(points[current], last, atol=1e-10)


def test_failed_sampling_is_charged_to_original_cap():
    for reported, base in [(350464,204800),(688072,614400)]:
        result = budget({'counts':{'team_transitions':reported}}, base)
        assert result['maximum_lifetime_team_steps'] <= CAP
        assert result['charged_attempt_steps'] >= reported+8
        assert result['target_logical_steps'] % BLOCK == 0
        assert result['target_logical_steps'] > base
    first = budget({'counts':{'team_transitions':350464}},204800)
    second = budget({'counts':{'team_transitions':600000},'repair_budget':first},409600)
    assert second['charged_lost_steps'] > first['charged_lost_steps']
    assert second['maximum_lifetime_team_steps'] <= CAP
