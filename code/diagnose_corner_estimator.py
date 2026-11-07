"""Read-only diagnosis of the previously viewed confirmation histories."""
from pathlib import Path
import json
import numpy as np

from audit_l45_rollout_reliability import observation,save
from solvers.multimodal_estimator import Estimator as Historical
from solvers.corner_estimator import Estimator
from run_multimodal_estimator_validation import score

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'current/l45_multimodal_estimator_20261025'
OUT=ROOT/'current/l45_corner_turn_estimator_20261025'


def main():
    entries=json.loads((SOURCE/'bank/manifest.json').read_text())['entries']
    rows=[]
    for entry in entries:
        packet=json.loads((SOURCE/'bank'/entry['public']).read_text())
        environment=json.loads((SOURCE/'bank'/entry['environment']).read_text())
        methods={'A':Estimator(packet['map'],'A',packet['controller_seeds'][0]),
                 'B':Estimator(packet['map'],'B',packet['controller_seeds'][0])}
        historical=Historical(packet['map'],'persistent',packet['controller_seeds'][0])
        outputs=[]
        for item in packet['history']:
            obs=observation(item)
            snapshots={}
            historical.update(obs)
            for name,own in methods.items():
                own.update(obs)
                snapshots[name]=(own.particles.copy(),own.distribution())
            np.testing.assert_array_equal(historical.particles,methods['A'].particles)
            if historical.weights is not None:
                np.testing.assert_array_equal(historical.weights,methods['A'].weights)
                np.testing.assert_array_equal(methods['A'].weights,methods['B'].weights)
            outputs.append(snapshots)
        # Targets and velocities are read by this independent scorer only.
        for world in environment['worlds']:
            trace=world['history_states'];states=np.array([s['evader'] for s in trace])
            velocity=np.diff(states,axis=0);turns=[]
            for t in range(1,len(velocity)):
                den=np.linalg.norm(velocity[t-1])*np.linalg.norm(velocity[t])
                if den>1e-9:
                    angle=float(np.degrees(np.arccos(np.clip(velocity[t-1]@velocity[t]/den,-1,1))))
                    if angle>=45:turns.append(t+1)
            timeline=[]
            for t,item in enumerate(packet['history']):
                if item['target'] is not None:continue
                measurements={}
                for name,(samples,(points,weights)) in outputs[t].items():
                    measurements[name]=score(points,weights,samples,states[t])
                timeline.append(dict(step=t,after_turn=t in turns,metrics=measurements))
            last=timeline[-1]['metrics'];mass=last['A']['mass_1m']
            rows.append(dict(group=entry['id'],family=environment['family'],world=world['id'],
                turns=turns,root=last,timeline=timeline,
                missed_by_old_export=not bool(last['A']['sample_coverage']),
                low_probability_mass=mass<.01,
                rescued_by_export_only=bool(last['B']['sample_coverage'] and not last['A']['sample_coverage']),
                probability_iid_64_would_hit=1.-(1.-mass)**64))
    summary={}
    for family in ('long_wall','wall_corner','multi_route'):
        rs=[r for r in rows if r['family']==family];miss=[r for r in rs if r['missed_by_old_export']]
        summary[family]=dict(distinct_handoff_states=len(rs),old_covered=sum(r['root']['A']['sample_coverage'] for r in rs),
            export_only_covered=sum(r['root']['B']['sample_coverage'] for r in rs),
            old_misses=len(miss),old_misses_with_mass_below_1pct=sum(r['low_probability_mass'] for r in miss),
            rescued_by_export_only=sum(r['rescued_by_export_only'] for r in rs),
            mean_root_mass=float(np.mean([r['root']['A']['mass_1m'] for r in rs])),
            mean_export_only_nll_delta=float(np.mean([r['root']['B']['nll']-r['root']['A']['nll'] for r in rs])),
            turn_count_mean=float(np.mean([len(r['turns']) for r in rs])))
    save(OUT/'diagnosis.json',dict(source=str(SOURCE),scope='previous confirmation is development now; no new independent claims',
        density_threshold=.01,original_A_exact=True,export_does_not_change_density=True,summary=summary,rows=rows))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
