"""Descriptive evaluator-only support geometry; not posterior calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45z_independent_confirmation_20261104'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=OUT/'selection_manifest.json';before=sha(path)
    rows=json.loads(path.read_text())['rows'];result=[]
    for stratum in ('handover','public_imminent_first'):
        subset=[r for r in rows if r['stratum']==stratum]
        for known in (False,True):
            targets=[m for r in subset for m in r['support_diagnostic'] if m['currently_visible']==known]
            if not targets:continue
            near=np.array([t['nearest_positive_support_distance'] for t in targets])
            if known:assert np.all(near<1e-10),'Visible target missing from support'
            result.append(dict(stratum=stratum,currently_visible=known,targets=len(targets),
                nearest_support_distance_median=float(np.median(near)),nearest_support_distance_max=float(near.max()),
                nearest_support_over_1=int(sum(near>1)),nearest_support_over_2=int(sum(near>2)),
                posterior_local_mass_medians={str(radius):float(np.median([m[f'posterior_mass_within_{radius}'] for m in targets]))
                    for radius in (1,2,4)}))
    assert sha(path)==before
    output=dict(input_sha256=before,analysis_sha256=sha(__file__),rows=result,
        scope='Actual collected root worlds only; descriptive support distances, no nominal coverage or causal claim.')
    (OUT/'support_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
