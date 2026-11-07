"""Read-only, posthoc PRIOR feasibility review; no controllers or fits run."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from learning.e45b_joint_network import Runtime, KEYS, log_density

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'current/e45c_budget_scaling_20261028'
OUT = ROOT / 'current/prior_feasibility_review_20261029'
SEEDS = (17, 29, 43)
KINDS = ('single', 'factorized', 'joint')
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    INPUTS[str(path)] = digest
    return json.loads(path.read_text())


def family(row):
    return row['method'].rstrip('0123456789')


def lse(x, axis=-1):
    top = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.exp(x - top).sum(axis=axis))


def joint_and_product_z(parameters, z, active):
    """Same learned marginals, with shared vs independent mixture component."""
    p, mu, logs = parameters
    terms = -.5 * ((z[:, None] - mu) * np.exp(-logs)) ** 2 - logs - .5 * np.log(2 * np.pi)
    per_agent = terms.reshape(len(z), len(p), 2, 2).sum(-1)
    logp = np.log(p)
    joint = lse(logp + (per_agent * active).sum(-1))
    product = (lse(logp[None, :, None] + per_agent, axis=1) * active).sum(-1)
    return joint, product


def density(runtime, parameters, actions, active):
    a = np.clip(actions, -1 + 1e-4, 1 - 1e-4)
    z = np.arctanh(a)
    p, mu, logs = parameters
    jacobian = (np.log1p(-a.reshape(-1, 2, 2) ** 2).sum(-1) * active).sum(-1)
    if runtime.kind != 'factorized':
        joint, _ = joint_and_product_z(parameters, z, active)
        return joint - jacobian
    z = z.reshape(-1, 2, 2)
    terms = (-.5 * ((z[:, :, None] - mu) * np.exp(-logs)) ** 2 - logs - .5 * np.log(2 * np.pi)).sum(-1)
    return (lse(np.log(p)[None] + terms, axis=-1) * active).sum(-1) - jacobian


def audit_episodes(rows):
    cells = collections.defaultdict(list)
    total = 0
    for row in rows:
        if Path(row['_output']).name != 'x20':
            continue
        assert row.get('valid_episode', True)
        episode = read(row['episode'], row['episode_sha256'])
        trace = episode['trace']
        assert len(trace) == row['decisions']
        assert abs(row['utility'] - (float(row['double_capture']) - .0025 * row['decisions'])) < 1e-10
        assert all(trace[0]['observation']['active_p']) and all(trace[0]['observation']['active_e'])
        assert row['first_capture'] == next((i + 1 for i, f in enumerate(trace) if f['new_captures']), None)
        two = sum(sum(f['observation']['active_p']) == 2 for f in trace)
        one = sum(sum(f['observation']['active_p']) == 1 for f in trace)
        assert two + one == len(trace)
        assert two == (row['first_capture'] or row['decisions'])
        total += 1
        cells[row['opponent'], family(row)].append((row, two, one))
    result = []
    for (op, arm), rr in sorted(cells.items()):
        first = [x[0]['first_capture'] for x in rr if x[0]['first_capture'] is not None]
        pre = [x[0]['first_capture'] or x[0]['completion_cost'] for x in rr]
        post = [x[0]['completion_cost'] - v for x, v in zip(rr, pre)]
        result.append(dict(opponent=op, method=arm, games=len(rr), groups=len({x[0]['group'] for x in rr}),
            double_capture_rate=np.mean([x[0]['double_capture'] for x in rr]),
            first_capture_rate=len(first) / len(rr),
            mean_first_capture_steps_conditional=np.mean(first) if first else None,
            mean_penalized_pre_first_capture_steps=np.mean(pre),
            mean_penalized_post_first_capture_steps=np.mean(post),
            mean_cost=np.mean([x[0]['completion_cost'] for x in rr]),
            mean_utility=np.mean([x[0]['utility'] for x in rr]),
            actual_two_v_two_decisions=sum(x[1] for x in rr), actual_one_v_one_decisions=sum(x[2] for x in rr)))
    return dict(verified_episode_files=total, summaries=result,
        limitation='Stage summaries are descriptive, policy-dependent and affected by survivor selection; no stage-specific causal claim.')


def contrasts(rows):
    selected = [r for r in rows if Path(r['_output']).name == 'x20']
    groups = sorted({r['group'] for r in selected})
    cells = collections.defaultdict(list)
    for r in selected:
        cells[r['opponent'], family(r), r['group']].append(r['utility'])
    comparisons = [(op, a, b) for op in ('fixed_D_kernel', 'rootpuct') for a, b in (
        ('PRIOR', 'U'), ('PRIOR', 'DIRECT'), ('PRIOR', 'AG'), ('PRIOR', 'IG'),
        ('JOINT', 'PRIOR'), ('PRIOR', 'FACT'), ('PRIOR', 'SINGLE'))]
    delta = np.array([[np.mean(cells[o, a, g]) - np.mean(cells[o, b, g]) for o, a, b in comparisons] for g in groups])
    center = delta.mean(0)
    rng = np.random.default_rng(945420261029)
    bootstrap = delta[rng.integers(len(groups), size=(10000, len(groups)))].mean(1)
    radius = float(np.quantile(np.max(np.abs(bootstrap - center), axis=1), .95))
    return dict(scope='Exploratory, posthoc 14-contrast family, not preregistered confirmation',
        groups=len(groups), draws=10000, seed=945420261029, simultaneous_radius=radius,
        rows=[dict(opponent=o, treatment=a, reference=b, utility_difference=center[i],
            simultaneous95=[center[i] - radius, center[i] + radius]) for i, (o, a, b) in enumerate(comparisons)])


def audit_proposals():
    model_manifest = read(SOURCE / 'models_frozen.json')
    for path, digest in model_manifest['files'].items():
        assert sha(path) == digest, path
        INPUTS[path] = digest
    folder = SOURCE / 'learning/x20/round_12'
    collection = read(folder / 'collection_results.json')
    output = []
    implementation_max_error = 0.
    for seed in SEEDS:
        runtimes = {kind: Runtime(folder / f'models/{kind}_{seed}.npz') for kind in KINDS}
        records = {kind: read(folder / f'training/{kind}_{seed}.json') for kind in KINDS}
        assert all(records[k]['validation_manifest'] == records['joint']['validation_manifest'] for k in KINDS)
        val = records['joint']['validation_manifest']['files']
        row_by_data = {r['data']: r for r in collection}
        rng = np.random.default_rng(945420261029 + seed)
        for path, digest in sorted(val.items()):
            assert sha(path) == digest
            INPUTS[path] = digest
            with np.load(path, allow_pickle=False) as data:
                data = {k: data[k].copy() for k in data.files}
            source = row_by_data[path]
            for i in range(len(data['public'])):
                active = data['agents'][i, :, 4]
                features = {k: data[k][i] for k in KEYS}
                actions = data['label_actions'][i]
                weights = data['label_weights'][i]
                assert abs(weights.sum() - 1) < 1e-10
                nll = {}
                for kind, runtime in runtimes.items():
                    parameters, _ = runtime(features)
                    ld = density(runtime, parameters, actions, active)
                    p, mu, logs = parameters
                    reference = log_density(tuple(torch.from_numpy(v)[None] for v in (np.log(p), mu, logs)),
                        torch.from_numpy(actions)[None], torch.from_numpy(active)[None], kind).numpy()[0]
                    error = float(np.max(np.abs(ld - reference)))
                    implementation_max_error = max(error, implementation_max_error)
                    assert error < 1e-8, (kind, error)
                    nll[kind] = float(-weights @ ld)
                parameters, _ = runtimes['joint'](features)
                j, product = joint_and_product_z(parameters, np.arctanh(np.clip(actions, -1+1e-4, 1-1e-4)), active)
                delta = float(weights @ (j - product))
                if active.sum() == 1:
                    assert np.max(np.abs(j - product)) < 1e-10
                    mi = 0.
                else:
                    p, mu, logs = parameters
                    component = rng.choice(len(p), size=512, p=p)
                    z = mu[component] + np.exp(logs[component]) * rng.standard_normal((512, 4))
                    jj, pp = joint_and_product_z(parameters, z, active)
                    mi = float(np.mean(jj - pp))
                output.append(dict(seed=seed, group=source['group'], opponent=source['opponent'], state_index=i,
                    active_agents=int(active.sum()), weighted_teacher_nll=nll,
                    teacher_log_density_gain_joint_over_own_product_marginals=delta,
                    raw_proposal_mutual_information_mc_nats=mi))
    summaries = []
    for seed in SEEDS:
        for active in (1, 2):
            selected = [x for x in output if x['seed'] == seed and x['active_agents'] == active]
            if not selected:
                continue
            gain = [x['teacher_log_density_gain_joint_over_own_product_marginals'] for x in selected]
            summaries.append(dict(seed=seed, active_agents=active, states=len(selected),
                mean_teacher_nll={k: np.mean([x['weighted_teacher_nll'][k] for x in selected]) for k in KINDS},
                mean_teacher_log_density_gain_joint_over_own_product_marginals=np.mean(gain),
                median_teacher_log_density_gain_joint_over_own_product_marginals=np.median(gain),
                fraction_states_positive_teacher_gain=np.mean(np.array(gain) > 1e-10),
                mean_raw_proposal_mutual_information_mc_nats=np.mean([x['raw_proposal_mutual_information_mc_nats'] for x in selected])))
    return dict(validation_only=True, no_action_or_episode_executed=True, final_models_verified=len(model_manifest['files']),
        density_crosscheck_max_error=implementation_max_error, summaries=summaries, states=output,
        limitation='Validation was used for checkpoint selection. Teacher visits are not independent true returns. MI is Monte Carlo on the raw pre-rejection proposal; 75/25 uniform mixing and joint legal rejection are excluded. Positive dependence is not evidence of beneficial coordination.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    read(SOURCE / 'experiment.json')
    rows = read(SOURCE / 'evaluation/results_with_engineering_failures.json')
    print('Verifying x20 episodes and phase accounting', flush=True)
    episodes = audit_episodes(rows)
    print('Computing posthoc grouped comparisons', flush=True)
    comparison = contrasts(rows)
    print('Checking frozen validation proposal densities and dependence', flush=True)
    proposals = audit_proposals()
    result = dict(date='2026-10-29 Asia/Shanghai', scope='Read-only posthoc feasibility audit; no new training or closed-loop evaluation',
        episodes=episodes, comparisons=comparison, proposals=proposals, input_sha256=INPUTS,
        script_sha256=sha(__file__))
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(episodes_verified=episodes['verified_episode_files'],
        density_crosscheck_max_error=proposals['density_crosscheck_max_error'], proposal_summaries=proposals['summaries']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
