"""Read-only E45-C result analysis; exploratory outputs stay separate."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import collections
import concurrent.futures as futures
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import run_e45c_budget as r
import run_e45c_accelerated as acceleration
import run_e45c_evaluation_audit as audit

OUT=r.OUT/'posthoc_analysis_20261028'
OPPONENTS=('fixed_D_kernel','rootpuct')
FAMILIES=('VALUE_U','SINGLE','FACT','JOINT','DIRECT','PRIOR')


def family(row):return next((x for x in FAMILIES if row['method'].startswith(x)),row['method'])
def dose(row):return Path(row['_output']).name

def grouped_compare(rows,pairs,seed):
    groups=sorted({x['group'] for x in rows});cells=collections.defaultdict(list)
    for x in rows:cells[(dose(x),x['opponent'],family(x),x['budget_ms'],x['group'])].append(x['utility'])
    def mean(g,op,arm,d,budget):
        selected=[v for (dd,oo,aa,bb,gg),values in cells.items() if gg==g and oo==op and aa==arm
            and (d is None or dd==d) and (budget is None or bb==budget) for v in values]
        if not selected:raise RuntimeError(('Missing inferential cell',g,op,arm,d,budget))
        return float(np.mean(selected))
    delta=np.array([[mean(g,o,a,d1,b1)-mean(g,o,b,d0,b0) for o,a,b,d1,d0,b1,b0 in pairs] for g in groups])
    center=delta.mean(0);rng=np.random.default_rng(seed)
    bs=delta[rng.integers(len(groups),size=(10000,len(groups)))].mean(1)
    radius=float(np.quantile(np.max(np.abs(bs-center),axis=1),.95))
    return dict(groups=len(groups),bootstrap_draws=10000,seed=seed,simultaneous_radius=radius,
        comparisons=[dict(opponent=o,treatment=a,reference=b,treatment_dose=d1,reference_dose=d0,treatment_budget=b1,reference_budget=b0,
            difference=float(center[i]),simultaneous95=[float(center[i]-radius),float(center[i]+radius)]) for i,(o,a,b,d1,d0,b1,b0) in enumerate(pairs)])


def summarize(rows,source):
    cells=collections.defaultdict(list)
    for x in rows:cells[(dose(x),x['opponent'],family(x),x['budget_ms'])].append(x)
    result=[]
    for (d,o,m,b),rr in sorted(cells.items()):
        valid=[x for x in rr if x.get('valid_episode',True)];bad=len(rr)-len(valid)
        lat=[v for x in valid for v in x['latencies_ms']];cpu=[v for x in valid for v in x['cpu_ms']]
        sims=[v for x in valid for v in x['hidden_simulations']];wins=sum(x['double_capture'] for x in valid)
        result.append(dict(source=source,dose=d,opponent=o,method=m,budget_ms=b,planned_games=len(rr),valid_games=len(valid),invalid_games=bad,
            groups=len({x['group'] for x in rr}),captures=wins,capture_rate_valid=wins/len(valid),capture_rate_bounds=[wins/len(rr),(wins+bad)/len(rr)],
            cost=float(np.mean([x['completion_cost'] for x in valid])),utility=float(np.mean([x['utility'] for x in valid])),
            decisions=len(lat),overruns=sum(x['online_overruns'] for x in valid),overrun_rate=sum(x['online_overruns'] for x in valid)/len(lat),
            p95_ms=float(np.percentile(lat,95)),max_ms=float(max(lat)),cpu_wall_ratio=float(sum(cpu)/sum(lat)),
            hidden_simulations_median=float(np.median(sims)) if sims else None))
    return result


def first_frame(row):
    payload=r.read(row['episode']);frame=payload['trace'][0]
    observation_hash=hashlib.sha256(json.dumps(frame['observation'],sort_keys=True).encode()).hexdigest()
    return dict(dose=dose(row),group=row['group'],world=row['world'],simulations=frame['stats']['simulations'],
        latency_ms=frame['latency_ms'],cpu_ms=frame['cpu_ms'],observation_hash=observation_hash)


def verify_episode(row):
    if row.get('valid_episode',True) is False:
        assert row['utility'] is None and row['completion_cost'] is None and row['double_capture'] is None
        assert r.sha(row['public_evidence'])==row['public_evidence_sha256']
        assert r.sha(row['estimator_arrays'])==row['estimator_arrays_sha256']
        return False
    assert r.sha(row['episode'])==row['episode_sha256'],row['episode']
    assert row['reason'] in ('double_capture','timeout')
    assert abs(row['utility']-(float(row['double_capture'])-.0025*row['decisions']))<1e-11
    return True


def run():
    OUT.mkdir(exist_ok=True);r.verify();acceleration.verify_execution(r.OUT);audit.verify();r.verify_models(r.OUT/'models_frozen.json')
    learning=r.read(r.OUT/'evaluation/results_with_engineering_failures.json');search=r.read(r.OUT/'search/results.json')
    oldstats=r.read(r.OUT/'statistics_with_engineering_failures.json');tasks=r.evaluation_tasks()
    assert len(learning)==9072 and len(search)==528
    assert len({audit.identity(x) for x in learning})==9072
    assert {audit.identity(x) for x in learning}=={audit.identity(x) for x in tasks}
    with futures.ThreadPoolExecutor(8) as pool:verified=list(pool.map(verify_episode,learning+search))
    assert sum(verified)==9071+528
    print('ALL_9599_NORMAL_EPISODE_HASHES_AND_ONE_INVALID_EVIDENCE_VERIFIED',flush=True)
    training=[];fits=0;collection_total=0
    for d in r.DOSES:
        folder=r.level(d)
        for manifest in folder.glob('round_*/models_manifest.json'):r.verify_models(manifest)
        for round_ in range(1,13):
            records=r.read(folder/f'round_{round_:02d}/training.json')['records'];assert len(records)==9
            collection=r.read(folder/f'round_{round_:02d}/collection_results.json');collection_total+=len(collection)
            fits+=len(records)
            for record in records:
                assert record['epochs']==30 and 1<=record['selected_epoch']<=30
                model=folder/f'round_{round_:02d}/models'/f'{record["kind"]}_{record["seed"]}.npz'
                assert r.sha(model)==record['model_sha256']
            for seed in r.SEEDS:
                seed_records=[x for x in records if x['seed']==seed]
                assert len(seed_records)==3
                assert all(x['train_manifest']==seed_records[0]['train_manifest'] and x['validation_manifest']==seed_records[0]['validation_manifest'] for x in seed_records)
            if round_==12:
                for record in records:training.append(dict(dose=d,kind=record['kind'],seed=record['seed'],epochs=record['epochs'],
                    selected_epoch=record['selected_epoch'],train_states=record['train_manifest']['states'],
                    train_episodes=len(record['train_manifest']['files']),validation_states=record['validation_manifest']['states'],
                    parameters=record['parameters'],seconds=record['seconds']))
    assert fits==324 and collection_total==27648
    print('324_FITS_27648_COLLECTION_GAMES_AND_THREE_NETWORK_DATA_EQUALITY_VERIFIED',flush=True)
    valid=[x for x in learning if x.get('valid_episode',True)];invalid=[x for x in learning if not x.get('valid_episode',True)]
    assert len(invalid)==1 and invalid[0]['method']=='DIRECT43' and dose(invalid[0])=='x5'
    assert all(audit.identity(x) in {audit.identity(v) for v in valid} for x in tasks if x['method'].startswith(('JOINT','PRIOR')))
    lp=[(o,m,m,d,'x1',None,None) for o in OPPONENTS for m in ('JOINT','PRIOR') for d in ('x5','x20')]
    primary=grouped_compare(valid,lp,r.BASE_SEED+774)
    sp=[(o,m,m,None,None,b,200) for o in OPPONENTS for m in ('JOINT','PRIOR','U') for b in (2000,10000)]
    budget=grouped_compare(search,sp,r.BASE_SEED+774)
    for current,previous in ((primary,oldstats['learning']),(budget,oldstats['search'])):
        assert current['groups']==previous['groups']
        for a,b in zip(current['comparisons'],previous['comparisons']):
            np.testing.assert_allclose(a['difference'],b['utility_difference'],atol=1e-14,rtol=0)
            np.testing.assert_allclose(a['simultaneous95'],b['simultaneous95'],atol=1e-14,rtol=0)
    print('INDEPENDENT_RECALCULATION_MATCHES_ALL_20_REGISTERED_COMPARISONS',flush=True)
    auxiliary_pairs=[(o,a,b,'x20','x20',None,None) for o in OPPONENTS for a,b in
        [('JOINT','PRIOR'),('JOINT','FACT'),('JOINT','SINGLE'),('JOINT','DIRECT'),('JOINT','AG'),('PRIOR','U'),('PRIOR','AG')]]
    auxiliary=grouped_compare(valid,auxiliary_pairs,r.BASE_SEED+991)
    auxiliary['scope']='POST HOC exploratory family: 14 highest-dose contrasts, 36 grouped histories; not preregistered confirmation'
    seeds=[]
    for d in r.DOSES:
        for o in OPPONENTS:
            for m in FAMILIES:
                for seed in r.SEEDS:
                    rr=[x for x in valid if dose(x)==d and x['opponent']==o and x['method']==m+str(seed)]
                    seeds.append(dict(dose=d,opponent=o,method=m,seed=seed,n=len(rr),captures=sum(x['double_capture'] for x in rr),
                        utility=float(np.mean([x['utility'] for x in rr])),cost=float(np.mean([x['completion_cost'] for x in rr]))))
    u=[x for x in valid if x['method']=='U' and x['opponent']=='rootpuct']
    with futures.ThreadPoolExecutor(8) as pool:roots=list(pool.map(first_frame,u))
    rootcells=collections.defaultdict(dict)
    for x in roots:rootcells[(x['group'],x['world'])][x['dose']]=x
    assert len(rootcells)==72 and all(len(v)==3 for v in rootcells.values())
    assert all(len({x['observation_hash'] for x in cell.values()})==1 for cell in rootcells.values())
    throughput=dict(same_public_roots=72,observations_match_across_doses=True,arms=[])
    for d in r.DOSES:
        rr=[x for x in roots if x['dose']==d]
        throughput['arms'].append(dict(dose=d,median_simulations=float(np.median([x['simulations'] for x in rr])),
            median_paired_ratio_to_x1=float(np.median([cell[d]['simulations']/max(1,cell['x1']['simulations']) for cell in rootcells.values()])),
            mean_cpu_wall_ratio=float(sum(x['cpu_ms'] for x in rr)/sum(x['latency_ms'] for x in rr))))
    print('SAME_ROOT_U_THROUGHPUT',throughput,flush=True)
    probes=r.read(r.OUT/'search/probe_results.json');probe_summary=[]
    with futures.ThreadPoolExecutor(8) as pool:probe_frames=list(pool.map(lambda x:r.read(x['episode'])['trace'][0],probes))
    for m in ('JOINT','PRIOR','U'):
        for b in (200,2000,10000):
            rr=[frame for row,frame in zip(probes,probe_frames) if family(row)==m and row['budget_ms']==b]
            probe_summary.append(dict(method=m,budget_ms=b,queries=len(rr),simulations_median=float(np.median([x['stats']['simulations'] for x in rr])),
                root_candidates_mean=float(np.mean([len(x['stats']['root_actions']) for x in rr]))))
    verification=dict(final_models=27,completed_fits=324,epochs_per_fit=30,collected_games=collection_total,
        attempted_learning_games=9072,valid_learning_games=9071,invalid_learning_games=1,search_games=528,root_probes=len(probes),
        hashes_checked=9599,primary_cells_complete=True,all_registered_statistics_reproduced=True,
        learned_networks_use_same_data_within_dose_and_seed=True)
    paths=[r.OUT/'evaluation/results_with_engineering_failures.json',r.OUT/'search/results.json',r.OUT/'statistics_with_engineering_failures.json',
        r.OUT/'models_frozen.json',r.OUT/'evaluation_continuation_decision.json',Path(__file__)]
    result=dict(verification=verification,primary=primary,search=budget,exploratory_highest_dose=auxiliary,
        arms=summarize(learning,'learning')+summarize(search,'search'),seeds=seeds,training_final_round=training,
        uniform_same_root_throughput=throughput,root_probe_summary=probe_summary,
        failures=invalid,input_hashes={str(p):r.sha(p) for p in paths},analysis_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    r.save(OUT/'analysis.json',result)
    plots(result);write_report(result)
    print('E45C_ANALYSIS_COMPLETE',str(OUT/'analysis.json'),flush=True)


def arm(data,d,o,m,b=200):
    return next(x for x in data['arms'] if x['source']=='learning' and x['dose']==d and x['opponent']==o and x['method']==m and x['budget_ms']==b)


def plots(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'JOINT':'#2368b5','PRIOR':'#21804b','SINGLE':'#d48c16','FACT':'#a263ad','U':'#777777','AG':'#111111'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for method in ('JOINT','PRIOR','FACT','SINGLE','U','AG'):
        rr=[arm(data,d,'rootpuct',method) for d in r.DOSES];style='--' if method in ('U','AG') else '-'
        axes[0].plot([1,5,20],[100*x['capture_rate_valid'] for x in rr],style,marker='o',color=colors[method],label=method)
        axes[1].plot([1,5,20],[x['cost'] for x in rr],style,marker='o',color=colors[method],label=method)
    for ax in axes:ax.set_xscale('log');ax.set_xticks([1,5,20],['1x','5x','20x']);ax.set_xlabel('Training experience dose');ax.grid(alpha=.2)
    axes[0].set_ylabel('Double-capture rate (%)');axes[1].set_ylabel('Failure-penalized completion steps')
    axes[0].set_ylim(45,103);axes[0].legend(ncol=2,fontsize=8)
    fig.suptitle('New 36 histories, adaptive RootPUCT opponent, 200 ms (descriptive means)')
    fig.tight_layout();fig.savefig(OUT/'learning_response.png',dpi=180);fig.savefig(OUT/'learning_response.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for i,x in enumerate(data['primary']['comparisons']):
        lo,hi=x['simultaneous95'];positive=lo>0;color='#21804b' if positive else '#777777'
        ax.errorbar(x['difference'],i,xerr=[[x['difference']-lo],[hi-x['difference']]],fmt='o',color=color,capsize=3)
    ax.set_yticks(range(8),[f"{x['opponent']} / {x['treatment']} / {x['treatment_dose']}-1x" for x in data['primary']['comparisons']])
    ax.axvline(0,color='black',linewidth=.8);ax.invert_yaxis();ax.set_xlabel('Team utility difference; simultaneous 95% intervals')
    ax.set_title('Eight preregistered learning comparisons; 36 grouped public histories');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'primary_intervals.png',dpi=180);fig.savefig(OUT/'primary_intervals.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for m in ('JOINT','PRIOR','U'):
        rr=[next(x for x in data['arms'] if x['source']=='search' and x['opponent']=='rootpuct' and x['method']==m and x['budget_ms']==b) for b in (200,2000,10000)]
        axes[0].plot([.2,2,10],[100*x['capture_rate_valid'] for x in rr],'-o',label=m,color=colors[m]);axes[1].plot([.2,2,10],[x['cost'] for x in rr],'-o',label=m,color=colors[m])
    for ax in axes:ax.set_xscale('log');ax.set_xticks([.2,2,10],['0.2','2','10']);ax.set_xlabel('Per-decision budget (s)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Double-capture rate (%)');axes[1].set_ylabel('Failure-penalized completion steps');axes[0].legend()
    fig.suptitle('Frozen OLD E45-B models, OLD 6 development histories (descriptive, not new confirmation)')
    fig.tight_layout();fig.savefig(OUT/'search_response.png',dpi=180);fig.savefig(OUT/'search_response.svg');plt.close(fig)


def write_report(data):
    # Detailed narrative is emitted by the review step using these verified
    # machine-readable quantities; keep the generated numeric appendix complete.
    lines=['# E45-C 数值分析附录','',
        '原始效用=双捕指示−0.0025×实际决策步数。主比较按36公共历史成组配对；神经臂216局含3种子，不是216张独立地图。','',
        '## 独立复算的预定八项学习比较','',
        '|对手|方法|剂量差|效用差|同时95%区间|','|---|---|---|---:|---|']
    for x in data['primary']['comparisons']:
        lines.append(f"|{x['opponent']}|{x['treatment']}|{x['treatment_dose']}−x1|{x['difference']:+.5f}|[{x['simultaneous95'][0]:+.5f},{x['simultaneous95'][1]:+.5f}]|")
    lines+=['','## 所有方法的描述性汇总','', '|来源|档位|对手|方法|预算ms|正常/计划|双捕|成本|效用|越预算调用|','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for x in data['arms']:
        lines.append(f"|{x['source']}|{x['dose']}|{x['opponent']}|{x['method']}|{x['budget_ms']:g}|{x['valid_games']}/{x['planned_games']}|{x['captures']}/{x['valid_games']}|{x['cost']:.3f}|{x['utility']:.5f}|{x['overruns']}/{x['decisions']}|")
    lines+=['','x5/RootPUCT/DIRECT含一局无效；该行均值仅正常局，双捕率在全部216任务上的界为62/216至63/216，不能将该失败填成普通输局。','',
        '## 20×辅助比较：事后14项同时区间','', '|对手|处理−对照|效用差|同时95%区间|','|---|---|---:|---|']
    for x in data['exploratory_highest_dose']['comparisons']:
        lines.append(f"|{x['opponent']}|{x['treatment']}−{x['reference']}|{x['difference']:+.5f}|[{x['simultaneous95'][0]:+.5f},{x['simultaneous95'][1]:+.5f}]|")
    (OUT/'NUMERICAL_APPENDIX.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':run()
