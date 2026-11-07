"""Descriptive posthoc E45-B diagnostics; leaves frozen outputs unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from learning.e45b_joint_network import Network,KEYS
from run_frozen_d_closed_loop import sha,save
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45b_role_free_joint_nte_20261027'
DEST=OUT/'posthoc_20261028'

def read(p):return json.loads(Path(p).read_text())
def family(m):
    return next((name for name in ('VALUE_U','SINGLE','FACT','JOINT','DIRECT','PRIOR') if m.startswith(name)),m)
def desc(a):
    a=np.asarray(a,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None

def main():
    torch.set_num_threads(1);DEST.mkdir(exist_ok=True)
    rows=read(OUT/'evaluation/results.json');groups=defaultdict(list);frames=defaultdict(lambda:defaultdict(list));phases=defaultdict(lambda:defaultdict(list))
    breakdown=defaultdict(list)
    for row in rows:
        key=row['opponent']+'/'+family(row['method']);groups[key].append(row)
        breakdown[key+'/'+row['family']].append(row)
        payload=read(row['episode'])
        for f in payload['trace']:
            stats=f['stats'];obs=f['observation'];ap=np.asarray(obs['active_p'],bool);ae=np.asarray(obs['active_e'],bool)
            phase='2v2' if ap.sum()==2 and ae.sum()==2 else '1v1';phases[key][phase].append(1)
            a=np.asarray(f['action'])[ap];amp=np.max(np.abs(a),axis=1)/read_case_speed(row['case'])
            values=frames[key];values['amplitude'].extend(amp);values['near_full'].extend(amp>=.95);values['nearly_stop'].extend(amp<.1)
            values['latency_ms'].append(f['latency_ms']);values['cpu_ms'].append(f['cpu_ms'])
            values['new_captures'].append(f['new_captures'])
            sims=stats.get('simulations')
            if sims is not None and family(row['method'])!='DIRECT':
                values['simulations'].append(sims);values['zero_simulations'].append(sims==0)
                values['root_candidates'].append(len(stats.get('root_actions',[])))
                values['rejections'].append(stats.get('rejections',0));values['fallbacks'].append(stats.get('fallbacks',0))
                if sims:values['transitions_per_complete_simulation'].append(stats['simulated_transitions']/sims)
                visits=np.asarray(stats.get('root_visits',[]),float)
                if visits.sum():
                    p=visits[visits>0]/visits.sum();values['root_max_visit_share'].append(p.max())
                    if len(p)>1:values['root_normalized_entropy'].append(-sum(p*np.log(p))/np.log(len(p)))
                phases[key][phase+'_simulations'].append(sims)
            phases[key][phase+'_steps'].append(1)
    result=dict(scope='Posthoc/descriptive. Same 18 history groups; not independent confirmatory experiments.',source_results_sha256=sha(OUT/'evaluation/results.json'),aggregate={},breakdown={},validation=[],training=[],phases={})
    for key,rr in groups.items():
        result['aggregate'][key]=dict(games=len(rr),double_captures=sum(r['double_capture'] for r in rr),capture_rate=np.mean([r['double_capture'] for r in rr]),
            utility=np.mean([r['utility'] for r in rr]),cost=np.mean([r['completion_cost'] for r in rr]),no_capture=sum(r['captures']==0 for r in rr),one_capture=sum(r['captures']==1 for r in rr),
            mean_first_capture_conditional=desc([r['first_capture'] for r in rr if r['first_capture'] is not None]),
            online_overruns=sum(r['online_overruns'] for r in rr),decisions=sum(r['decisions'] for r in rr),metrics={k:desc(v) for k,v in frames[key].items()})
        result['phases'][key]={k:desc(v) for k,v in phases[key].items()}
    for key,rr in breakdown.items():result['breakdown'][key]=dict(games=len(rr),double_captures=sum(r['double_capture'] for r in rr),utility=float(np.mean([r['utility'] for r in rr])),cost=float(np.mean([r['completion_cost'] for r in rr])))
    for round_ in range(1,13):
        folder=OUT/f'round_{round_:02d}';rr=read(folder/'collection_results.json')
        for split in ('train','validation'):
            for opp in ('fixed_D_kernel','rootpuct'):
                selected=[r for r in rr if r['split']==split and r['opponent']==opp]
                if selected:result['training'].append(dict(round=round_,split=split,opponent=opp,games=len(selected),capture_rate=float(np.mean([r['double_capture'] for r in selected])),utility=float(np.mean([r['utility'] for r in selected]))))
        if round_ not in (1,2,4,8,12):continue
        for seed in (17,29,43):
            selected=[r for r in rr if r['split']=='validation' and r['method']==f'JOINT{seed}']
            arrays=[];raw=[];phi=[]
            for row in selected:
                with np.load(row['data']) as data:arrays.append({k:data[k].copy() for k in data.files})
                trace=read(row['episode'])['trace'];chosen=np.unique(np.linspace(0,len(trace)-1,min(32,len(trace))).astype(int))
                returns=np.cumsum([f['reward'] for f in trace][::-1])[::-1]
                raw.extend(returns[chosen]);phi.extend([trace[i]['stats']['root_potential'] for i in chosen])
            x={k:torch.from_numpy(np.concatenate([a[k] for a in arrays])).float() for k in KEYS}
            y=np.concatenate([a['label_value'] for a in arrays]);raw=np.asarray(raw);phi=np.asarray(phi)
            np.testing.assert_allclose(y,raw-.05*phi,atol=1e-9,rtol=0)
            for kind in ('single','factorized','joint'):
                m=Network(kind);m.load_state_dict(torch.load(folder/'models'/f'{kind}_{seed}.pt',weights_only=True,map_location='cpu'));m.eval()
                with torch.no_grad():params,v=m(x);pred=v.numpy()
                residual=pred+.05*phi;error=residual-raw
                def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-8 and np.std(b)>1e-8 else None
                result['validation'].append(dict(round=round_,seed=seed,kind=kind,states=len(raw),raw_mae=float(abs(error).mean()),raw_bias=float(error.mean()),
                    raw_rmse=float(np.sqrt(np.mean(error**2))),raw_correlation=corr(residual,raw),shaped_correlation=corr(pred,y),raw_prediction=desc(residual),raw_truth=desc(raw),
                    outside_feasible_raw_range=float(np.mean((residual<-.2)|(residual>1))),
                    expected_potential=desc(-.05*phi),shaped_mae=float(np.mean(abs(pred-y)))))
    save(DEST/'diagnostics.json',result)
    print(json.dumps(dict(aggregate={k:{f:v for f,v in val.items() if f!='metrics'} for k,val in result['aggregate'].items()},validation_joint_final=[r for r in result['validation'] if r['round']==12 and r['kind']=='joint']),indent=2))
    plot(result)

_case_speeds={}
def read_case_speed(path):
    if path not in _case_speeds:_case_speeds[path]=read(path)['speed']
    return _case_speeds[path]

def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['AG','IG','U','SINGLE','FACT','JOINT','PRIOR','VALUE_U','DIRECT'];fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for j,opp in enumerate(('fixed_D_kernel','rootpuct')):
        vals=[result['aggregate'][opp+'/'+m] for m in methods]
        axes[j].bar(np.arange(len(methods)),[v['capture_rate']*100 for v in vals],color=['#607D8B' if m not in ('JOINT','PRIOR') else '#D55E00' if m=='JOINT' else '#009E73' for m in methods])
        axes[j].set_xticks(np.arange(len(methods)),methods,rotation=40,ha='right');axes[j].set_ylim(0,110);axes[j].set_ylabel('Double-capture rate (%)');axes[j].set_title(opp)
        for i,v in enumerate(vals):axes[j].text(i,v['capture_rate']*100+2,f"{v['double_captures']}/{v['games']}",ha='center',fontsize=8)
    fig.suptitle('E45-B frozen test results (descriptive; 18 independent history groups)');fig.tight_layout();fig.savefig(DEST/'capture_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
