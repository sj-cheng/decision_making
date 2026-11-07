"""E49 state/history and fixed-selection game PDFs, read-only plotting."""
import os
os.environ['MPLBACKEND']='Agg'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
import plot_e48_dense_obstacles as old

OUT=Path(__file__).resolve().parents[1]/'current/e49_hard_initials_20261107'


def read(path):return json.loads(Path(path).read_text())


def initials():
    cfg=read(OUT/'experiment.json');(OUT/'figures').mkdir(exist_ok=True)
    with PdfPages(OUT/'00_hard_initials.pdf') as pdf:
        for entry in cfg['entries'][:2]:
            case=read(entry['path']);states=np.array(case['worlds'][0]['scripted_states']);s=states[-1]
            fig,ax=plt.subplots(figsize=(10,9),layout='constrained');old.setup(ax,case)
            for i in range(4):
                xy=states[:,2*i:2*i+2];color=old.COLORS[i]
                ax.plot(xy[:,0],xy[:,1],color=color,ls='--',lw=1.2,alpha=.65)
                ax.scatter(*xy[0],color=color,s=35,marker='o',alpha=.6)
                ax.scatter(*xy[-1],color=color,s=90,marker='^' if i>=2 else 's',label=('E1 hidden truth','E2 hidden truth','P1','P2')[i],edgecolor='white',zorder=5)
                if i>=2:ax.add_patch(Circle(xy[-1],6,facecolor=color,alpha=.06,edgecolor=color,lw=.7))
            ax.legend(ncol=2,loc='lower left')
            ax.set_title(f'E49 | map00/start{entry["world"]} | hidden handoff at step12\n'
                f'Both targets unseen for >=4 steps; nearest pursuer distance {entry["construction"]["nearest_distance"]:.2f}m',fontsize=13)
            fig.supxlabel('Faded dashed traces: legal pre-handoff observer history. Squares show hidden evader truth for visualization only.\n'
                'Disks indicate range6; walls also occlude. Pursuers receive public histories, never hidden evader coordinates.',fontsize=9)
            pdf.savefig(fig);fig.savefig(OUT/'figures'/f'hard_initial_{entry["world"]}.png',dpi=125);plt.close(fig)


def selected():
    rows={}
    for f in (OUT/'results').glob('E49_*/*.json'):
        r=read(f);t=r['task']
        if 'engineering' in t['stage']:continue
        if t['entry']['map_index']==0 and t['world']=='AB' and t['train_seed']==17:
            mode='fixed16' if t.get('fixed') is not None else 'nominal200ms'
            rows[mode,t['opponent'],t['arm']]=r
    return rows


def games():
    rows=selected()
    if len(rows)!=20:raise RuntimeError(f'Preselected E49 games not all present: {len(rows)}/20')
    with PdfPages(OUT/'01_method_opponent_comparison.pdf') as overview, PdfPages(OUT/'02_representative_games.pdf') as atlas:
        for mode in ('fixed16','nominal200ms'):
            for opponent in ('rootpuct','geometry_escape'):
                fig,axs=plt.subplots(2,3,figsize=(13,9),layout='constrained')
                for ax,arm in zip(axs.flat,old.LABELS):old.draw(ax,rows[mode,opponent,arm],True)
                axs.flat[-1].axis('off');axs.flat[-1].text(0,.9,'Orange/red: evaders\nBlue/green: pursuers\nCircle=start; star=captured/retired\nX=active final position\nDashed evader path=hidden\n\n'
                    'Both targets hidden at step12\n68 physical steps remain\nAG includes target assignment\nFull-state evader, public-belief pursuer',va='top',fontsize=10)
                fig.suptitle(f'E49 | map00/startAB/model17 | {mode} | opponent: {opponent}',fontsize=14)
                overview.savefig(fig);fig.savefig(OUT/'figures'/f'comparison_{mode}_{opponent}.png',dpi=120);plt.close(fig)
                for arm in old.LABELS:
                    fig,ax=plt.subplots(figsize=(10,10),layout='constrained');old.draw(ax,rows[mode,opponent,arm])
                    fig.suptitle(f'E49 | {mode} | {opponent} | hidden handoff, map00/startAB/model17',fontsize=12)
                    fig.supxlabel('Initial targets are hidden. Shown target positions are plotting truth, not controller input.\n'
                        'No outcome-based example selection; failed/invalid examples retained.',fontsize=9)
                    atlas.savefig(fig);plt.close(fig)


def results():
    r=read(OUT/'final/results.json');table={(x['mode'],x['opponent'],x['arm']):x for x in r['table']}
    with PdfPages(OUT/'03_results.pdf') as pdf:
        for opponent in ('rootpuct','geometry_escape'):
            fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            names=list(old.LABELS);short=['Geom+tree','L5+tree','Geom direct','L5 direct','AG']
            for col,mode in enumerate(('fixed16','nominal200ms')):
                for row,key,scale in [(0,'double_capture',100),(1,'completion_cost',1)]:
                    ax=axs[row,col];v=[table[mode,opponent,a][key]*scale if (mode,opponent,a) in table else np.nan for a in names]
                    ax.bar(short,v,color=['#5b8db8','#e8913a','#91b5d1','#f3ba7f','#58a88c']);ax.tick_params(axis='x',rotation=20)
                    ax.set_title(mode+' | '+('double capture (%)' if row==0 else 'completion cost (failure penalized)'));ax.grid(axis='y',alpha=.2)
                    if row==0:ax.set_ylim(0,110)
                    for i,value in enumerate(v):
                        if np.isfinite(value):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom',fontsize=9)
            fig.suptitle(f'E49 | opponent: {opponent} | 6 maps, two hidden histories, frozen models\n'
                f'Invalid tasks: {r["invalid_count"]}; descriptive pilot, not broad optimality evidence',fontsize=13)
            pdf.savefig(fig);fig.savefig(OUT/'figures'/f'results_{opponent}.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('initials','all'));a=p.parse_args();initials()
    if a.mode=='all':games();results()
