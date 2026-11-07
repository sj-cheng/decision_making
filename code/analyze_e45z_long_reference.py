"""Exploratory read-only link to independently seeded, already archived X returns.

Root one-step sample means select a geometry-proxy action; X's original full
continuations score it. These are NOT newly executed model-aware trajectories.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45z_model_calibration_20261104'
REFERENCE=ROOT/'current/e45x_first_capture_diagnostic_20261104/root_ranking_audit.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def grouped(rows,names):
    result={}
    for name in names:
        groups={}
        for r in rows:groups.setdefault(r['group'],[]).append(r[name])
        values=np.asarray([np.mean(v) for v in groups.values()])
        rng=np.random.default_rng(20261104)
        ci=np.percentile(rng.choice(values,(10000,len(values)),replace=True).mean(1),[2.5,97.5])
        result[name]=dict(mean=float(values.mean()),descriptive_95_interval=ci.tolist(),maps=len(values))
    return result


def main():
    paths=[OUT/'results.json',REFERENCE,Path(__file__)]
    hashes={str(p):sha(p) for p in paths}
    rows=json.loads(paths[0].read_text())['root_summaries']
    references={r['root_id']:r for r in json.loads(REFERENCE.read_text())['roots']}
    records=[]
    for r in rows:
        if r['split']!='development' or r['source_opponent']!='rootpuct':continue
        ref=references[r['root_id']];a=r['chosen_D_proxy_action'];b=r['chosen_Root_proxy_action']
        records.append(dict(root_id=r['root_id'],group=r['group'],stratum=r['stratum'],
            D_proxy_action=a,Root_proxy_action=b,changed=a!=b,
            D_action_reference_utility=ref['reference_utilities'][a],
            Root_action_reference_utility=ref['reference_utilities'][b],
            utility_gain=ref['reference_utilities'][b]-ref['reference_utilities'][a],
            double_capture_gain=ref['reference_success'][b]-ref['reference_success'][a],
            reference_worlds_per_action=ref['streams_per_candidate']))
    summary=[]
    for stratum in ('handover','public_imminent_first','both_state_types'):
        subset=[r for r in records if stratum=='both_state_types' or r['stratum']==stratum]
        summary.append(dict(stratum=stratum,roots=len(subset),changed=sum(r['changed'] for r in subset),
            metrics=grouped(subset,['D_action_reference_utility','Root_action_reference_utility','utility_gain','double_capture_gain'])))
    uncertainty=[]
    for stratum in ('handover','public_imminent_first'):
        for known in (False,True):
            targets=[m|dict(group=r['group']) for r in rows if r['split']=='development' and r['stratum']==stratum
                for m in r['target_metrics'] if m['known']==known]
            if targets:
                uncertainty.append(dict(stratum=stratum,currently_visible=known,targets=len(targets),
                    metrics=grouped(targets,['D_model_current_position_variance','D_model_within_future_variance',
                        'D_model_between_position_future_variance','conditional_excess_mean_bias_squared'])))
    result=dict(exploratory_followup=True,input_and_analysis_hashes=hashes,records=records,summary=summary,
        uncertainty_details=uncertainty,scope='Only original Root-source development roots; no new episodes. Independent archived X reference streams, same reused maps.',
        caveats=['Root sample-based selector is an expensive simulator diagnostic, not a deployed public predictor.',
            'One action intervention followed by frozen Single Actor; not an integrated model-aware planner.',
            'Six reused development maps; five have an imminent Root state; eight paired worlds per action.',
            'Post-hoc exploratory analysis after one-step calibration, no confirmation or novelty claim.'])
    for p,h in hashes.items():assert sha(p)==h,p
    (OUT/'long_reference_link.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    for r in summary:print(r)


if __name__=='__main__':main()
