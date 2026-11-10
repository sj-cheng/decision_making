from __future__ import annotations
import gzip, json, pickle, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
from matplotlib import font_manager

_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(_FONT).exists():
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path('/data2/sjc/decision_making')
OUT = ROOT/'current/e49_scatter_equal_20261110'
EP = OUT/'episodes/equal'
METHOD_ORDER = ['IG','AG','RouteAPF','MAPPO_RouteAPF','GEOM_SEARCH','SINGLE_PRIOR','A1','A2','A3','A4','A5']
DISPLAY = {'MAPPO_RouteAPF':'MAPPO+RouteAPF','GEOM_SEARCH':'GEOM_SEARCH','SINGLE_PRIOR':'SINGLE_PRIOR'}

def load_group(group, seed=17):
    found = {}
    for p in (OUT/'results/equal').glob('*.json'):
        d=json.loads(p.read_text())
        t=d.get('task',{})
        if t.get('group')==group and int(t.get('train_seed',-1))==seed and d.get('valid'):
            found[t['arm']] = d
    missing=[m for m in METHOD_ORDER if m not in found]
    if missing: raise RuntimeError(f'{group}: missing {missing}; found {sorted(found)}')
    return found

def read_episode(result):
    with gzip.open(result['episode'],'rb') as f: return pickle.load(f)

def draw_frame(ax, boxes, ep, step, title, final_summary):
    ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for b in boxes:
        b=np.asarray(b,float)
        ax.add_patch(Rectangle((b[0,0],b[1,0]), b[0,1]-b[0,0], b[1,1]-b[1,0],
                               facecolor='#d0d0d0', edgecolor='#777', linewidth=.55, zorder=0))
    trace=ep['trace']
    states=[np.asarray(trace[0]['state'],float)] + [np.asarray(x['next_state'],float) for x in trace[:step]]
    arr=np.asarray(states)
    # state order: evader0, evader1, pursuer0, pursuer1
    ev=arr[:,:4].reshape(-1,2,2); pu=arr[:,4:8].reshape(-1,2,2)
    for j in range(2):
        ax.plot(pu[:,j,0],pu[:,j,1],color='#1769e0',lw=1.45,zorder=2)
        ax.plot(ev[:,j,0],ev[:,j,1],color='#e36f21',lw=1.35,ls='--',zorder=2)
        ax.scatter(pu[-1,j,0],pu[-1,j,1],s=32,color='#1769e0',edgecolor='white',linewidth=.45,zorder=4)
        # captured targets are ringed black at the capture location
        # state layout: time, active pursuer[2], active evader[2], ...
        active=np.asarray(arr[-1,11:13] if arr.shape[1]>=13 else [1,1])
        if active[j] < .5:
            ax.scatter(ev[-1,j,0],ev[-1,j,1],s=55,facecolors='none',edgecolors='black',linewidth=1.2,zorder=5)
        else:
            ax.scatter(ev[-1,j,0],ev[-1,j,1],s=32,color='#e36f21',edgecolor='white',linewidth=.45,zorder=4)
    ax.axvline(0,color='#aaa',lw=.45,ls=':'); ax.axhline(0,color='#aaa',lw=.35,ls=':')
    # elapsed step and outcome
    ax.set_title(f'{title}\n{step:02d}/80 步',fontsize=9,pad=3)

def make(group, seed=17, fps=6):
    cases=json.loads((OUT/'maps'/f'{group}.json').read_text())
    boxes=np.asarray(cases['map'],float)
    results=load_group(group,seed)
    eps={m:read_episode(results[m]) for m in METHOD_ORDER}
    frames=[]
    fig,axes=plt.subplots(3,4,figsize=(13.2,10.2),dpi=130)
    axes=axes.ravel()
    for step in range(81):
        for ax in axes: ax.clear()
        for i,m in enumerate(METHOD_ORDER):
            s=eps[m]['summary']; title=DISPLAY.get(m,m)
            draw_frame(axes[i],boxes,eps[m],step,title,s)
        axes[-1].axis('off')
        fig.suptitle(f'E49 几何逃跑者同速实验｜{group}｜出生点 00｜种子 17\n蓝=追捕者  橙=逃跑者（虚线）  黑圈=已捕获',fontsize=14,y=.985)
        fig.tight_layout(rect=(0,0,1,.95))
        fig.canvas.draw()
        frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
        frames.append(frame)
    path=OUT/f'e49_{group}_seed17_trajectory.gif'
    imageio.mimsave(path,frames,duration=1/fps,loop=0)
    plt.close(fig)
    return path

if __name__=='__main__':
    import sys
    groups=sys.argv[1:] or ['scatter_open_00']
    for g in groups: print(make(g))
