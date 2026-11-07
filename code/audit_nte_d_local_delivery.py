"""Post-run read-only reconstruction of engineering histories and teacher gate.

This audit adds no experiments, no teacher queries and no test outcomes. Its
descriptive diagnostics do not change the preregistered stop decision.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from run_frozen_d_closed_loop import save, sha, legal_action
from audit_frozen_d_search_interface import observation
from solvers.frozen_d_search_interface import loader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_nte_d_local_baseline_20261027'


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    config, gate = read(OUT/'experiment.json'), read(OUT/'teacher_gate.json')
    integrity = []
    for name in ('source_hashes','protected_hashes','bank_hashes'):
        for p,h in config[name].items():
            assert sha(p)==h, p
        integrity.append(dict(kind=name,files=len(config[name]),passed=True))
    loader.verify()
    frames, transitions, terminal_captures = 0, 0, 0
    engineering_records = []
    for path in sorted((OUT/'engineering/episodes').glob('*.json')):
        data=read(path); packet=read(data['task']['packet_path'])
        boxes=np.asarray(packet['map']);speed=packet['motion']['pursuer_axis_limit']
        e=loader.load_estimator(packet['map'],data['task']['seed'])
        for r in packet['history'][:-1]:
            e.update(observation(r))
        for r in data['trace']:
            obs=observation(r['observation']);e.update(obs)
            np.testing.assert_array_equal(e.particles,np.asarray(r['particles']))
            np.testing.assert_array_equal(e.particle_weights,np.asarray(r['weights']))
            signature=hashlib.sha256(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes()).hexdigest()
            assert signature==r['estimator_sha256']
            assert legal_action(obs.position,np.asarray(r['own_action']),speed,boxes)
            assert not any(r['simulation_errors'])
            assert np.isclose(sum(r['weights']),1.,rtol=0,atol=1e-12)
            frames+=1;transitions+=r['simulated_transitions']
            if r['capture']:
                terminal_captures+=1
                assert r['post_seen']
        engineering_records.append(dict(key=data['task']['key'],frames=len(data['trace']),public_D_reconstruction=True))
    details=[]
    all_queries=[]
    for row in gate['records']:
        values=np.asarray(row['probe_returns'])
        assert values.shape==(96,3)
        diff=values[:,0]-values[:,1]
        mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
        lower,upper=mean-1.96*se,mean+1.96*se
        signal=bool(len(diff)>=24 and lower>0)
        distance=float(np.linalg.norm(np.asarray(row['visit_mean_A'])-row['visit_mean_B']))
        assert signal==row['signal'] and (distance<=.5)==row['stable']
        assert np.isclose(mean,row['heldout_gap']) and np.isclose(se,row['heldout_gap_se'])
        assert row['passed']==(signal and distance<=.5)
        q=row['query_A']['statistics']
        for query in (row['query_A'],row['query_B']):
            stats=query['statistics'];all_queries.append(query)
            assert stats['simulations']==sum(stats['root_visits'])
            assert len(stats['root_actions'])==len(stats['root_visits'])
            assert not any(query['simulation_errors'])
        label='winner_better' if lower>0 else 'winner_worse' if upper<0 else 'unresolved'
        details.append(dict(group=row['group'],family=row['family'],gate_passed=row['passed'],
            winner_minus_runner_mean=mean,nominal95=[lower,upper],signal_class=label,
            winner_model_return=float(values[:,0].mean()),runner_model_return=float(values[:,1].mean()),
            G_model_return=float(values[:,2].mean()),winner_minus_G=float((values[:,0]-values[:,2]).mean()),
            winner_action_norm=float(np.linalg.norm(np.asarray(row['probe_actions'][0]))),
            visit_mean_normalized_norm=float(np.linalg.norm(row['visit_mean_A'])),
            repeat_mean_distance=distance,root_proposals=q['root_candidates'],
            root_simulations=q['simulations'],root_max_visits=max(q['root_visits']),
            mean_action_legal=row['mean_action_legal_A']))
    assert sum(r['gate_passed'] for r in details)==gate['passed_states']==6
    training=read(OUT/'training.json');decision=read(OUT/'decision.json')
    assert not training['started'] and training['epochs_completed']==0
    assert not list((OUT/'models').glob('*'))
    assert not list((OUT/'evaluation').glob('**/episodes/*.json'))
    assert decision['heldout_games']==0 and not decision['formal_training_started']
    interface=read(OUT/'pretraining_interface.json')
    timed=[r['first_total_ms'] for r in interface['records']]
    query_ms=[q['complete_ms'] for q in all_queries]
    diagnostic=dict(teacher_states=12,gate_passes=6,required_passes=7,
        nominal_signal_counts={k:sum(r['signal_class']==k for r in details)
            for k in ('winner_better','winner_worse','unresolved')},
        mean_repeat_distance=float(np.mean([r['repeat_mean_distance'] for r in details])),
        mean_visit_label_norm=float(np.mean([r['visit_mean_normalized_norm'] for r in details])),
        mean_winner_minus_G_model_return=float(np.mean([r['winner_minus_G'] for r in details])),
        G_strictly_higher_model_return_states=sum(r['winner_minus_G'] < -1e-12 for r in details),
        G_tied_model_return_states=sum(abs(r['winner_minus_G'])<=1e-12 for r in details),
        G_lower_model_return_states=sum(r['winner_minus_G'] > 1e-12 for r in details),
        all_visit_mean_actions_legal=all(r['mean_action_legal'] for r in details),
        all_probe_returns_nonzero=all(r['winner_model_return']>0 for r in details),
        actual_teacher_queries=24,
        teacher_simulations=sum(q['statistics']['simulations'] for q in all_queries),
        teacher_simulated_transitions=sum(q['simulated_transitions'] for q in all_queries),
        probe_simulated_transitions=sum(r['probe_simulated_transitions'] for r in gate['records']),
        teacher_root_proposals_range=[min(r['root_proposals'] for r in details),max(r['root_proposals'] for r in details)],
        teacher_query_ms=dict(p50=float(np.median(query_ms)),p95=float(np.percentile(query_ms,95)),max=max(query_ms),over2000=sum(x>2000 for x in query_ms)),
        timed_random_interface_ms=dict(p50=float(np.median(timed)),p95=float(np.percentile(timed,95)),max=max(timed),over200=sum(x>200 for x in timed)),
        cases=details,
        caveats=['descriptive public-model diagnostics, not held-out closed-loop results',
            'nominal paired intervals are development screens, not simultaneous test claims',
            'stable weighted mean can average competing actions; no isolated causal attribution',
            'gate probes visit winner, not direct usefulness of the weighted-mean training label',
            'G comparison is under an eight-step public model, not real-opponent capture performance'])
    save(OUT/'teacher_diagnosis.json',diagnostic)
    result=dict(passed=True,integrity=integrity,engineering_episodes=len(engineering_records),
        engineering_frames=frames,engineering_simulated_transitions=transitions,
        public_D_exact_reconstruction=True,engineering_records=engineering_records,
        hidden_target_not_in_D_reconstruction=True,terminal_captures_checked=terminal_captures,
        gate_recomputed=True,formal_training_not_started=True,test_outcomes_not_consumed=True,
        teacher_trajectories_online_independently_audited=True,
        full_teacher_trajectories_posthoc_replayed=False)
    save(OUT/'public_reconstruction_audit.json',result)
    return diagnostic,result


def augment_report(d,a):
    p=OUT/'REPORT.md'
    text=p.read_text()
    marker='## 教师屏障的逐状态诊断'
    if marker in text:
        text=text[:text.index(marker)].rstrip()+'\n'
    paragraphs=['',marker,'',
        '12个状态的重复搜索均值均通过稳定性检查；独立MC将其中6个判为最高访问动作优于次高访问动作、3个判为更差、3个仍无法区分。这不是“所有标签都没有信号”，而是未达到本轮事前规定的多数状态可靠门槛。所有对照动作均来自公开信息。','',
        '|状态|类型|最高−次高访问动作回报差|名义95%区间|判定|',
        '|---|---|---:|---|---|']
    for i,r in enumerate(d['cases']):
        lo,hi=r['nominal95']
        label={'winner_better':'更好','winner_worse':'更差','unresolved':'证据不足'}[r['signal_class']]
        paragraphs.append(f'|{i+1}|{r["family"]}|{r["winner_minus_runner_mean"]:+.4f}|[{lo:+.4f}, {hi:+.4f}]|{label}|')
    paragraphs += ['',
        '24次2秒教师查询共完成'+str(d['teacher_simulations'])+'次模拟；各A查询根部有'+str(d['teacher_root_proposals_range'])+'个连续动作，有限访问分散到多个候选。最高访问排序在独立回报下并不总可靠。该现象支持检查有限采样的监督质量，但本轮没有通过消融证明唯一原因。','',
        f'12个最高访问动作均有非零模型回报，12个访问加权均值动作均合法。因此不能把本轮停止简单归因为“奖励全为零”或“标签均值非法”。均值动作归一化长度平均为{d["mean_visit_label_norm"]:.4f}；不同方向平均后的动作可能较小，但尚不能认定这是闭环失败原因。','',
        f'同一独立模拟下，G参考动作在{d["G_strictly_higher_model_return_states"]}/12个状态回报严格更高、{d["G_lower_model_return_states"]}个更低、{d["G_tied_model_return_states"]}个持平。最高访问动作减G的平均模型回报差为{d["mean_winner_minus_G_model_return"]:+.4f}。这是8步规划模型内的描述性诊断，不是对真实对手的闭环捕获比较，也不是本轮训练入口必须超过G的附加条件。','',
        '本轮使用标准对数UCT选择与连续渐进扩展，是在既有公共信念搜索上的NTE式适配；没有逐项恢复原作者的Polynomial UCT及原任务。因此即使未来训练完成，也需要记录这一实现差异，不能用来宣布“击败原论文NTE”。本轮尚未建立训练充分的NTE性能基线。','',
        f'事后用公共观测精确重建{a["engineering_episodes"]}条工程前缀、{a["engineering_frames"]}帧D输出，样本与权重逐项一致。工程前缀含{a["engineering_simulated_transitions"]}个模拟转移；24次教师查询含{d["teacher_simulated_transitions"]}个转移、独立探针含{d["probe_simulated_transitions"]}个转移，其几何均已在线独立检查。教师模拟只保存在线审计与轨迹散列，未声称事后重放全部教师模拟。','',
        f'12次随机模型完整首次决策P95={d["timed_random_interface_ms"]["p95"]:.2f}ms、最大={d["timed_random_interface_ms"]["max"]:.2f}ms，超时0；24次教师完整查询最大={d["teacher_query_ms"]["max"]:.2f}ms，超2秒{d["teacher_query_ms"]["over2000"]}次。这些数字不能替代训练后模型的时延验收。','',
        '训练和最终测试阶段未执行，其代码路径仅作为后续工程草稿保留，不能列为已验收训练/评估成果。当前已验收范围是公共信息接口、随机网络推理、有限搜索、工程闭环前缀和教师屏障。','',
        '- [逐状态完整诊断](teacher_diagnosis.json)、[公共历史重建审计](public_reconstruction_audit.json)、[适配差异](ADAPTATION.md)。',
        '- [交接说明](HANDOFF.md)、[教师屏障图](figures/teacher_gate.pdf)。']
    p.write_text(text+'\n'.join(paragraphs)+'\n',encoding='utf-8')


def plots(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=d['cases']; x=np.arange(1,len(rows)+1)
    means=np.array([r['winner_minus_runner_mean'] for r in rows])
    errors=np.array([r['nominal95'][1]-r['winner_minus_runner_mean'] for r in rows])
    fig,ax=plt.subplots(figsize=(8.3,4.1))
    ax.errorbar(x,means,yerr=errors,fmt='o',capsize=3,color='#284b63',lw=1.3)
    for boundary in (4.5,8.5):
        ax.axvline(boundary,color='0.8',lw=.8)
    ax.axhline(0,color='0.4',lw=1,ls='--')
    ax.set_xticks(x);ax.set_xlabel('Development history (1–4 wall; 5–8 corner; 9–12 multi-route)')
    ax.set_ylabel('Held-out model return: visit winner − runner-up')
    ax.set_title('Teacher screen: 6/12 pass; preregistered threshold 7/12')
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout()
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'teacher_gate.png',dpi=180);fig.savefig(folder/'teacher_gate.pdf')
    plt.close(fig)


if __name__=='__main__':
    started=time.perf_counter()
    d,a=audit();plots(d);augment_report(d,a)
    accounting=dict(main_pipeline_seconds=read(OUT/'decision.json')['pipeline_wall_seconds'],
        supplemental_audit_seconds=time.perf_counter()-started,
        registration_to_audit_end_wall_seconds=time.time()-(OUT/'experiment.json').stat().st_mtime,
        conservative_total_bound_includes_preflight=True,registered_cap_seconds=21600)
    assert accounting['registration_to_audit_end_wall_seconds']<21600
    save(OUT/'compute_accounting.json',accounting)
    with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:
        f.write(f'\n从配置登记到事后审计结束墙钟{accounting["registration_to_audit_end_wall_seconds"]:.2f}秒（包含接口验收，是正式6小时计算窗口的保守上界）；仍在预定预算内。详见 [计算账](compute_accounting.json)。\n')
    print(json.dumps(dict(audit_passed=a['passed'],gate_passes=d['gate_passes'],
        engineering_frames=a['engineering_frames'],teacher_queries=d['actual_teacher_queries']),ensure_ascii=False))
