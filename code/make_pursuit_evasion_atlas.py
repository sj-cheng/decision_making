"""Qualitative, outcome-independent trajectory atlas; never official evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/pursuit_evasion_figures_20261107'
LABELS = dict(A1='Uniform pool + search', A2='Geometry ranking + search',
              A3='L5 learned ranking + search', A4='Geometry ranking, direct',
              A5='L5 learned ranking, direct', R1='GEOM search (E46 reference)',
              R2='Single PRIOR search', AG='Assigned geometry (AG)',
              IG='Independent geometry (IG)', APF='RouteAPF',
              RMAPPO='RMAPPO-Clearance', MAPPO_APF='MAPPO + RouteAPF',
              W_B='PPO from scratch (W-B)', W_C='Pretrained + PPO (W-C)',
              GIT='Git d47d188 search + learning')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def init_worker(queue):
    os.sched_setaffinity(0, {queue.get()})


def modern_job(job):
    import numpy as np
    import torch
    import run_l5_followup as run
    target = OUT/'episodes'/f'{job["case"]["family"]}__{job["method"]}.json'
    if target.exists(): return str(target)
    import importlib.util
    if job['method'].startswith('W_') and importlib.util.find_spec('gymnasium') is None:
        python = ROOT/'current/e45w_joint_ppo_and_planning_20261103/venv/bin/python'
        subprocess.run([str(python),'-u',__file__,'one','--case',job['case']['path'],
                        '--method',job['method']],check=True,timeout=300)
        return str(target)
    run.configure(OUT)
    entry, method = job['case'], job['method']
    spec = next((s for s in run.specs() if s['arm']==method and s['train_seed']==17),
                dict(name=method, arm=method, train_seed=17))
    task = run.episode_task(entry, spec, 'qualitative', mode='sim', value=16,
                            world='AB', opponent='rootpuct')
    case = run.read(entry['path'])
    checkpoints = {}
    if method.startswith('W_'):
        from learning.e45w_joint_env import PublicJointPursuitEnv
        from learning.e45w_joint_ppo import StructuredJointPolicy
        path = ROOT/f'current/e45w_joint_ppo_and_planning_20261103/training/{method[-1]}_17/selected_actor.pt'
        env = PublicJointPursuitEnv(dict(case_paths=[entry['path']], record_trace=True,
               belief_evidence_dir=str(OUT/'belief_failures')))
        policy = StructuredJointPolicy(env.observation_space, env.action_space, lambda _:0., initialization_seed=0)
        policy.load_actor_state(torch.load(path,map_location='cpu',weights_only=False))
        policy.set_training_mode(False); torch.manual_seed(task['seed'])
        obs, info = env.reset(options=dict(case_path=entry['path'],world='AB',opponent='rootpuct',
                                           episode_seed=task['seed'],controller_seed=task['seed']))
        with torch.no_grad():
            while True:
                action = policy.predict(obs,deterministic=False)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated: break
        payload = dict(summary=info['task_episode'],trace=env.trace)
        checkpoints[str(path)] = run.sha(path)
    else:
        import run_e45b_role_free as base
        if method in ('AG','IG'):
            from solvers.e45_team_adapter import TeamController
            base.make_controller = lambda c,t: TeamController(c['map'],c['speed'],t['seed'],method)
        elif method in ('APF','RMAPPO','MAPPO_APF'):
            import run_e45j_repaired_benchmark as baseline
            task['apf_coefficients'] = baseline.gains()
            path = baseline.frozen_checkpoint(method,17)
            if path:
                task.update(checkpoint=str(path),checkpoint_sha256=run.sha(path))
                checkpoints[str(path)] = run.sha(path)
            base.make_controller = baseline.factory
        payload, _ = base.episode(dict(task,key=run.digest(task)[:24]))
        if spec.get('checkpoint'): checkpoints[spec['checkpoint']] = run.sha(spec['checkpoint'])
    states = [payload['trace'][0]['state']]+[r['next_state'] for r in payload['trace']]
    from solvers.e45_team_adapter import public_observation
    observations = [public_observation(int(s[8]),np.asarray(s[4:8]).reshape(2,2),
                     np.asarray(s[:4]).reshape(2,2),np.asarray(s[11:13])>.5,
                     np.asarray(s[9:11])>.5,np.asarray(case['map'])) for s in states]
    data = dict(method=method,label=LABELS[method],case_id=entry['id'],family=entry['family'],
        suite=entry['suite'],map=case['map'],states=states,observations=observations,
        summary=payload['summary'],task=task,checkpoint_hashes=checkpoints,
        scope='Qualitative example, fixed16 search / direct policy; no outcome-based selection',
        trace=payload['trace'])
    write(target,data)
    print('DONE',entry['family'],method,len(states)-1,flush=True)
    return str(target)


def legacy_run(case_path):
    import numpy as np
    import torch
    oldcode=OUT/'source/git_d47d188/code'
    sys.path.insert(0,str(oldcode))
    from problems.example8 import Example8
    from learning.oracles import get_oracles
    from solvers.solver import get_solver
    case=json.loads(Path(case_path).read_text());np.random.seed(20261107);torch.manual_seed(17)
    p=Example8();p.obstacles=[np.asarray(b,float) for b in case['map']]
    p.use_minco_dynamics=False
    p.current_evader_speed_lim=1.;p.current_pursuer_speed_lim=case['speed']
    for i in range(4):
        speed=1. if i<2 else case['speed']
        p.action_lims[p.action_idxs[i]]=[[-speed,speed],[-speed,speed]]
        p.control_lims[p.action_idxs[i]]=p.action_lims[p.action_idxs[i]]/p.dt
    p.clear_closed_loop_cache()
    weights=ROOT/'current/memory_observation_v3/models'
    policies=[str(weights/f'model_policy_l45_i{i}.pt') for i in range(4)]
    value=str(weights/'model_value_l45.pt')
    policy,value_model=get_oracles(p,policy_oracle_name='gaussian',policy_oracle_paths=policies,
                                    value_oracle_name='deterministic',value_oracle_path=value)
    common=dict(search_depth=100,number_simulations=1000,C_pw=2.,alpha_pw=.5,
                C_exp=1.,alpha_exp=.25,beta_policy=.9,beta_value=0.,vis_on=False)
    own=get_solver('C_PUCT_V1',policy_oracle=policy,value_oracle=value_model,**common)
    other=get_solver('C_PUCT_V1',policy_oracle=[None]*4,value_oracle=None,
                     **dict(common,number_simulations=3000,beta_policy=0.))
    state=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']=='AB'),float).reshape(-1,1)
    assert p.is_valid(state)
    states=[state[:,0].tolist()]; trace=[]
    for _ in range(80-int(state[8,0])):
        start=time.perf_counter()
        action=own.policy(p,state.copy(),team=[2,3])+other.policy(p,state.copy(),team=[0,1])
        nxt=p.step(state,action,p.dt)
        trace.append(dict(action=action[:,0].tolist(),latency_ms=1000*(time.perf_counter()-start)))
        states.append(nxt[:,0].tolist());state=nxt
        if p.is_terminal(state):break
    import hashlib
    hashes={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in policies+[value]}
    data=dict(method='GIT',label=LABELS['GIT'],case_id=case['id'],family=case['family'],
              map=case['map'],states=states,trace=trace,checkpoint_hashes=hashes,
              summary=dict(double_capture=bool(not np.any(state[p.active_idxs[:2],0]>.5)),
                           captures=int(2-np.sum(state[p.active_idxs[:2],0]>.5)),decisions=len(trace)),
              scope='Historical algorithm on matched geometry/initial state. Full-state search root, original dynamics/reward. Not an equal-information baseline. V3 iteration45 resolved from currently available complete weights; not proof of historical default checkpoint. C++ RNG has no seed API.')
    target=OUT/'episodes'/f'{case["family"]}__GIT.json';write(target,data);print(target,flush=True)


def run_all():
    import run_l5_followup as run
    cfg=run.read(run.OLD/'experiment.json')
    entries=[next(e for e in cfg['entries'] if e['split']=='development' and e['family']==family)
             for family in ('long_wall','multi_route','u_courtyard')]
    methods=[m for m in LABELS if m!='GIT']
    write(OUT/'selection.json',dict(entries=entries,methods=methods,selected_before_outcomes=True,
         note='Map and initial state from development, not ongoing sealed confirmation set'))
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in (40,41,42,43):queue.put(cpu)
    with ProcessPoolExecutor(4,mp_context=ctx,initializer=init_worker,initargs=(queue,)) as pool:
        jobs=[pool.submit(modern_job,dict(case=e,method=m)) for e in entries for m in methods]
        errors=[]
        for f in as_completed(jobs):
            try:f.result()
            except Exception as exc:
                import traceback
                errors.append(dict(error=repr(exc),traceback=traceback.format_exc()))
                print(errors[-1]['traceback'],flush=True)
        write(OUT/'modern_errors.json',errors)
    if errors:
        raise RuntimeError(f'{len(errors)} qualitative jobs require repair; completed trajectories preserved')
    for e in entries:
        target=OUT/'episodes'/f'{e["family"]}__GIT.json'
        if target.exists():continue
        subprocess.run(['taskset','-c','40',sys.executable,'-u',__file__,'legacy',
                        '--case',e['path']],check=True,cwd=ROOT,timeout=900)


def render():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle, Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#e66101','#b2182b','#2166ac','#1b9e77'];names=['E1','E2','P1','P2']
    selection=json.loads((OUT/'selection.json').read_text())
    data=[json.loads(p.read_text()) for p in sorted((OUT/'episodes').glob('*.json'))]
    expected={(e['family'],m) for e in selection['entries'] for m in LABELS}
    assert {(d['family'],d['method']) for d in data} == expected, 'Atlas requires every requested trajectory'
    order={m:i for i,m in enumerate(LABELS)}
    data.sort(key=lambda d:([e['family'] for e in selection['entries']].index(d['family']),order[d['method']]))
    def setup(ax,d):
        for b in d['map']:
            b=np.asarray(b);ax.add_patch(Rectangle((b[0,0],b[1,0]),b[0,1]-b[0,0],b[1,1]-b[1,0],
                facecolor='#d1d5db',edgecolor='#6b7280',lw=.7,zorder=0))
        ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal');ax.grid(alpha=.15,lw=.5)
    def trajectory(ax,d,small=False):
        setup(ax,d);s=np.asarray(d['states']);n=len(s)
        for i in range(4):
            active=s[:,9+i]>.5;ended=np.flatnonzero(~active)
            stop=int(ended[0]) if len(ended) else n-1
            xy=s[:stop+1,2*i:2*i+2];segments=np.stack([xy[:-1],xy[1:]],1)
            styles=['solid']*len(segments)
            if i<2 and d.get('observations'):
                styles=['solid' if any(q is not None for q in o['reports'][i]) else 'dashed'
                        for o in d['observations'][:stop]]
            ax.add_collection(LineCollection(segments,colors=colors[i],linewidths=1.4 if small else 2.,
                                             linestyles=styles,zorder=3,label=names[i]))
            ax.scatter(*xy[0],marker='o',s=18 if small else 40,color=colors[i],edgecolor='white',zorder=5)
            if not small:ax.annotate(names[i],xy[0],xytext=((-18,12),(-18,-16),(5,9),(5,-13))[i],
                                     textcoords='offset points',color=colors[i],weight='bold')
            ax.scatter(*xy[-1],marker='*' if len(ended) else 'X',s=40 if small else 100,
                       color=colors[i],edgecolor='white',lw=.4,zorder=5)
            for t in range(5,len(xy)-1,10 if small else 5):
                ax.annotate('',xy[t],xy[t-1],arrowprops=dict(arrowstyle='->',color=colors[i],lw=.8))
            if i<2 and len(ended) and not small:
                ax.annotate(f'capture t={s[stop,8]:.0f}',xy[-1],xytext=(4,-12),
                            textcoords='offset points',fontsize=8,color=colors[i])
        return s
    (OUT/'pdf').mkdir(exist_ok=True);(OUT/'preview').mkdir(exist_ok=True)
    index=[]
    with PdfPages(OUT/'01_all_games.pdf') as atlas:
        for d in data:
            fig=plt.figure(figsize=(11.7,8.3),layout='constrained');gs=fig.add_gridspec(3,4)
            ax=fig.add_subplot(gs[:,:3]);s=trajectory(ax,d);ax.set(xlabel='x (m)',ylabel='y (m)')
            ax.legend(loc='upper left',ncol=4,fontsize=8,framealpha=.85)
            for row,t in enumerate((0,len(s)//2,len(s)-1)):
                sub=fig.add_subplot(gs[row,3]);setup(sub,d)
                for i in range(4):
                    if s[t,9+i]>.5:
                        sub.scatter(*s[t,2*i:2*i+2],color=colors[i],s=30,marker='s' if i<2 else '^')
                        if i>=2 and d['method']!='GIT':
                            sub.add_patch(Circle(s[t,2*i:2*i+2],6,facecolor=colors[i],alpha=.06,edgecolor=colors[i],lw=.5))
                    else:
                        sub.scatter(*s[t,2*i:2*i+2],color=colors[i],s=55,marker='*',alpha=.6,
                                    edgecolor='white',lw=.3)
                sub.set_title(f't = {s[t,8]:.0f} steps',fontsize=9);sub.tick_params(labelsize=7)
            captured=int(2-np.sum(s[-1,9:11]>.5))
            budget='1000 simulations; privileged root' if d['method']=='GIT' else '16 simulations for search; direct baselines use no search'
            fig.suptitle(f'{LABELS[d["method"]]} | {d["family"]}\nCaptured {captured}/2; {len(s)-1} decisions; {budget}',fontsize=13)
            fig.supxlabel('Circle: start   X: final active position   Star: captured/retired   Arrows: time direction   Disks: sensor range; walls occlude\n'
               'Evader dashed path: hidden from pursuer team (ground truth shown only by plotter). '
               'Historical Git paths: visibility not encoded.  Qualitative examples only.',fontsize=8)
            file=OUT/'pdf'/f'{d["family"]}__{d["method"]}.pdf';fig.savefig(file);atlas.savefig(fig)
            if d['method'] in ('A3','AG','W_C','GIT'):fig.savefig(OUT/'preview'/f'{file.stem}.png',dpi=130)
            index.append(dict(method=d['method'],family=d['family'],captures=captured,decisions=len(s)-1,
                              pdf=str(file.relative_to(OUT))))
            plt.close(fig)
    with PdfPages(OUT/'00_comparison_overview.pdf') as overview:
        for entry in selection['entries']:
            group=[d for d in data if d['family']==entry['family']]
            fig,axs=plt.subplots(4,4,figsize=(14,14),layout='constrained')
            for ax,d in zip(axs.flat,group):
                s=trajectory(ax,d,True);captures=int(2-np.sum(s[-1,9:11]>.5))
                ax.set_title(f'{LABELS[d["method"]]}\n{captures}/2 captures, {len(s)-1} decisions',fontsize=9);ax.tick_params(labelsize=7)
            for ax in list(axs.flat)[len(group):]:ax.axis('off')
            fig.suptitle(f'{entry["family"]} | matched geometry and initial physical state\n'
               'E1/E2: orange/red; P1/P2: blue/green. Git is a privileged historical reference, not a fair baseline.',fontsize=13)
            overview.savefig(fig);fig.savefig(OUT/'preview'/f'overview_{entry["family"]}.png',dpi=110);plt.close(fig)
    write(OUT/'figure_index.json',index)
    lines=['# 追逃博弈轨迹图册','',
        '先看 [三张地图对照总览](00_comparison_overview.pdf)，再看 [全部逐局大图](01_all_games.pdf)。每局单页 PDF 在 `pdf/`，PNG 预览在 `preview/`，完整状态、公开观测、动作和参数在 `episodes/`。','',
        '三张图为开发集 long_wall、multi_route、u_courtyard 的第一个实例，在跑之前固定选取；每种方法各跑一局，共同 AB 初态、共同 RootPUCT3000 对手种子。没有按成功结果换图，也不触碰正在运行的封存确认地图。','',
        '当前搜索方法用16次完整模拟，直接控制器不做搜索；这批是定性演示，不替代200 ms正式结果。网络用训练种子17。W-B/W-C使用冻结的验证集选中 actor，按原评估方式随机采样动作；RMAPPO/MAPPO+APF用其原确定性部署。','',
        'AG 含全队目标匹配，IG 含独立目标选择，RouteAPF/MAPPO+RouteAPF含人工目标选择，属于带先验参考；其余方法的具体约束以原实现为准。轨迹显示仿真真值供查看；当前控制器只接收公开观测，虚线表示目标对追捕团队不可见。','',
        '历史 Git 参考按用户确认的 d47d188 源码隔离编译；V3第45轮权重是当前目录可用的最新完整权重，不能证明是提交当时的默认权重。保留原1000次模拟、深度100、策略概率0.9、价值概率0；对手为历史代码纯搜索3000次。为便于看图使用相同几何、AB物理初态和速度，但保留旧动力学/奖励/全状态搜索根接口，不是同信息、同预算的公平baseline。旧C++未暴露随机种子接口，实际轨迹已保存。','',
        '旧源码在本机Eigen版本下需要一处枚举尺寸显式转int才能编译；仅修改隔离副本，原文件与补丁保存在 `source/`，未改算法。论文正文未自动插图。','',
        '|地图|方法|捕获数|决策数|PDF|','|---|---|---:|---:|---|']
    for r in index:lines.append(f'|{r["family"]}|{LABELS[r["method"]]}|{r["captures"]}/2|{r["decisions"]}|[查看]({r["pdf"]})|')
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'COMPLETE.json',dict(games=len(data),pdfs=len(index)+2,finished=time.strftime('%Y-%m-%d %H:%M:%S'),qualitative_only=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','legacy','render','one'))
    parser.add_argument('--case');parser.add_argument('--method');args=parser.parse_args()
    if args.mode=='run':run_all()
    elif args.mode=='legacy':legacy_run(args.case)
    elif args.mode=='one':
        entry=next(e for e in json.loads((OUT/'selection.json').read_text())['entries'] if e['path']==args.case)
        modern_job(dict(case=entry,method=args.method))
    else:render()
