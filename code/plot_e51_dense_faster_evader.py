"""Actual trajectories for the speed-only dense-scene display."""
import os
os.environ['MPLBACKEND']='Agg'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import plot_e48_dense_obstacles as plot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e51_dense_faster_evader_20261107'
NAMES=dict(A3='L5 + search',R2='Single PRIOR + search',A2='Same-pool geometry + search',
    A4='Geometry direct',A5='L5 direct',AG='Assigned geometry (AG)',GIT='Git d47d188 search + learning')


def read(path):return json.loads(Path(path).read_text())


def run():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    plot.LABELS.update(NAMES)
    cfg=read(OUT/'experiment.json')
    rows=[read(f) for f in (OUT/'results/E51_nominal200ms').glob('*.json')]
    history=[read(f) for f in (OUT/'historical').glob('*.json') if not f.stem.endswith('_case')]
    assert len(rows)==24 and len(history)==2 and all(r['valid'] for r in history)
    (OUT/'figures').mkdir(exist_ok=True)
    idx=[]
    with PdfPages(OUT/'00_method_comparison.pdf') as comparison,PdfPages(OUT/'01_all_games.pdf') as atlas:
        for entry in cfg['entries']:
            selected={r['task']['arm']:r for r in rows+history
                if r['task']['entry']['id']==entry['id'] and r['task']['world']==entry['world']}
            order=['A3','R2','GIT' if 'GIT' in selected else 'A5','A2','A4','AG']
            fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
            for ax,arm in zip(axs.flat,order):
                r=selected[arm];plot.draw(ax,r,True)
                if not r['valid']:
                    ax.set_title(f'{NAMES[arm]}\nENGINEERING INVALID: old belief support exhausted',fontsize=10)
                    continue
                s=r['result']['summary']
                budget='1000sim / full state (historical reference)' if arm=='GIT' else 'direct; no tree' if arm in ('A4','A5','AG') else 'nominal 200ms'
                ax.set_title(f'{NAMES[arm]}\n{s["captures"]}/2 captures, {s["decisions"]} steps\n{budget}',fontsize=10)
            axs.flat[0].legend(handles=[Line2D([0],[0],color=c,lw=2,label=n)
                for c,n in zip(plot.COLORS,['E1','E2','P1','P2'])],loc='upper left',ncol=2,fontsize=8)
            fig.suptitle(f'E51 | Dense15, original starts | map{entry["map_index"]:02d} / {entry["world"]}\n'
                'Original RootPUCT3000 evader; evader axis limit 1.0 -> 1.3; pursuer remains 1.3',fontsize=14)
            fig.supxlabel('Circle=start; star=captured/retired; X=active final; arrows=time direction; dashed E path=hidden.\n'
                'Original 80-step task and frozen pursuers. Git has different information/budget; no smoothing or outcome-based case selection.',fontsize=9)
            comparison.savefig(fig)
            fig.savefig(OUT/'figures'/f'comparison_map{entry["map_index"]:02d}_{entry["world"]}.png',dpi=135)
            plt.close(fig)
            for arm in NAMES:
                if arm not in selected:continue
                r=selected[arm]
                fig,ax=plt.subplots(figsize=(10,10),layout='constrained');plot.draw(ax,r)
                if not r['valid']:
                    ax.set_title(f'{NAMES[arm]} | ENGINEERING INVALID\nOld unit-speed belief support exhausted')
                    fig.suptitle(f'E51 | map{entry["map_index"]:02d} / {entry["world"]} | failed attempt retained')
                    atlas.savefig(fig);plt.close(fig)
                    idx.append(dict(map_index=entry['map_index'],world=entry['world'],method=arm,engineering_invalid=True,error=r['error']))
                    continue
                s=r['result']['summary']
                fig.suptitle(f'E51 | map{entry["map_index"]:02d} / {entry["world"]} | original RootPUCT evader, speed1.3',fontsize=13)
                note=f'Git: 1000sim/full-state historical reference; {s["pursuer_shields"]} pursuer safety stops.' if arm=='GIT' else 'Original pursuer controller; frozen model seed17; nominal200ms interface.'
                fig.supxlabel(note+'\nOnly evader speed changed from E48; ordinary visible initial state. Actual unsmoothed trajectory.',fontsize=10)
                atlas.savefig(fig);plt.close(fig)
                idx.append(dict(map_index=entry['map_index'],world=entry['world'],method=arm,captures=s['captures'],steps=s['decisions']))
    (OUT/'figure_index.json').write_text(json.dumps(idx,indent=2)+'\n')
    lines=['# E51：复杂障碍＋原逃跑者提速','','只将E48原逃跑者每轴速度上限从1.0提高到1.3，追捕者仍为1.3。普通初态、RootPUCT3000/深度100、80步及其余任务条件保留。没有沿用E49困难接管或新增几何逃跑基线。','',
        '|方法|对局数|双捕局数|平均实际步数|','|---|---:|---:|---:|']
    for arm in NAMES:
        rr=[r['result']['summary'] for r in rows+history if r['valid'] and r['task']['arm']==arm]
        lines.append(f'|{NAMES[arm]}|{len(rr)}|{sum(s["double_capture"] for s in rr)}|{np.mean([s["decisions"] for s in rr]):.2f}|')
    lines+=['','当前方法尝试24局：22局正常完成，2局旧信念支持耗尽，工程失效单列；另有原Git历史参考2局。前两张地图每张AB/BA，共4个固定初态；仅作对局展示，不能把这几局比例当稳定总体捕获率。',
        'Git保留原1000模拟/全状态根，使用现存V3第45轮权重，无法证明是提交时默认；公共执行合法性停车单列，不能作为同信息同预算排名。',
        '追捕网络、信念与规划核没有重训或改动，仍保留原目标运动假设。本轮是冻结方法面对更快逃跑者的展示。地图00/AB的A2、A3因旧单位速度信念不能解释提速后的负观测而耗尽支持，未伪装成普通输局或更换初态；表内仅是各方法有效局描述，样本不齐，不能当公平总体排名。没有为消除失效另加重置或修改估计算法。',
        '原逃跑者搜索边界、真实环境上限和执行合法性检查均同步为1.3。原工程合法性屏蔽保留，各局计数在原始数据。',
        '既有E48/E49/E50记录保留；后两轮附加设置不再沿用。']
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'README.md').write_text('''# E51：复杂场景，原逃跑者速度提高到1.3

按最新要求，仅保留复杂15障碍布局，恢复E48普通初态与原RootPUCT逃跑算法；逃跑者每轴速度1.0→1.3，追捕者仍1.3。其余任务与冻结方法不变，没有新增训练、逃跑策略、困难历史或轨迹平滑。

- [四组同初态方法对照（4页）](00_method_comparison.pdf)
- [全部对局大图（26页）](01_all_games.pdf)
- [结果及条件说明](final/REPORT.md)
- [PDF与报告打包](PDF_and_report.zip)

蓝绿为追捕者，橙红为逃跑者；圆起点、星捕获/退休、X仍活跃的终点，箭头为时间方向；逃跑虚线为对追捕团队不可见的区间。所有预选对局均保留，没有按结果换图。

当前6方法各尝试4局，共22局正常、2局工程失效（地图00/AB的A2、A3，旧信念支持耗尽，失效面板保留），冻结种子17、名义200ms接口；Git另有2局历史参考，原1000模拟/全状态信息与当前方法不同。该批只用于看轨迹，不是新独立确认研究。地图与初态及环境随机种子继承E48最前4条登记；不再使用E49人工撤退历史和几何前瞻逃跑者。工程失效不算正常输局，各方法有效样本不齐，不作总体排名。

速度同步进入真实环境、原RootPUCT搜索动作界限与执行检查。追捕网络、信念与规划核保持原样，仍保留原目标运动假设；不声称已经为更快目标重新适配。原始轨迹和原始/实际逃跑动作均保存，工程屏蔽单独计数。

E49/E50已完成的历史数据保留，但本轮不再沿用它们新增的设置。所有本轮输出在此目录。
''')


if __name__=='__main__':run()
