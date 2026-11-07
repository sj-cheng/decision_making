"""Actual displacements, own pursuers highlighted and opposing paths separated."""
import os
os.environ['MPLBACKEND']='Agg'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import gzip
import json
import pickle
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
import plot_e48_dense_obstacles as previous

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'current/e50_own_pursuers_20261107'
METHODS=dict(A3='Your L5 + search',R2='Your Single PRIOR + search',A5='Your L5, direct')


def read(p):return json.loads(Path(p).read_text())


def payload(row):
    if row['task']['arm']=='GIT':return dict(summary=row['result']['summary'],trace=row['trace'])
    with gzip.open(row['result']['episode'],'rb') as f:return pickle.load(f)


def draw(ax,row,role):
    task=row['task'];data=payload(row);case=read(task['case']);previous.setup(ax,case)
    tr=data['trace'];s=np.array([tr[0]['state']]+[v['next_state'] for v in tr]);ids=(2,3) if role=='pursuer' else (0,1)
    for i in ids:
        end=np.flatnonzero(s[:,9+i]<.5);stop=int(end[0]) if len(end) else len(s)-1;xy=s[:stop+1,2*i:2*i+2]
        styles=['solid']*stop
        if i<2:styles=['solid' if any(q is not None for q in r['observation']['reports'][i]) else 'dashed' for r in tr[:stop]]
        ax.add_collection(LineCollection(np.stack((xy[:-1],xy[1:]),axis=1),colors=previous.COLORS[i],linewidths=2.,linestyles=styles,label=('E1','E2','P1','P2')[i]))
        ax.scatter(*xy[0],color=previous.COLORS[i],s=35,edgecolor='white',zorder=4)
        ax.scatter(*xy[-1],color=previous.COLORS[i],marker='*' if len(end) else 'X',s=80,edgecolor='white',zorder=4)
        for step in range(5,len(xy),7):ax.annotate('',xy[step],xy[step-1],arrowprops=dict(arrowstyle='->',color=previous.COLORS[i],lw=1.))
    if role=='pursuer':
        for i in (0,1):ax.scatter(*s[0,2*i:2*i+2],color='#999999',s=15,marker='s',alpha=.6)
    ax.legend(loc='lower left',fontsize=8,ncol=2);ax.tick_params(labelsize=8)
    return data['summary']


def all_rows():
    rows=[]
    for item in read(OUT/'episode_index.json')['episodes']:
        row=read(item['path'])
        if row['valid']:rows.append(row)
    return rows


def run():
    (OUT/'figures').mkdir(exist_ok=True);rows=all_rows();selected={};angles={}
    for row in rows:
        t=row['task'];mode='fixed16' if t.get('fixed') is not None else 'nominal200ms'
        if t['entry']['map_index']==0 and t['world']=='AB' and t['train_seed']==17:selected[mode,t['arm']]=row
        if t['arm'] not in METHODS:continue
        data=payload(row)
        for frame in data['trace']:
            s=np.array(frame['state']);d=(np.array(frame['next_state'])[:8]-s[:8]).reshape(4,2)
            for role,ids in [('pursuer',(2,3)),('evader',(0,1))]:
                for i in ids:
                    if s[9+i]>.5 and np.linalg.norm(d[i])>1e-9:angles.setdefault((t['arm'],role),[]).append(np.degrees(np.arctan2(d[i,1],d[i,0])))
    assert all((mode,arm) in selected for mode in ('fixed16','nominal200ms') for arm in METHODS)
    with PdfPages(OUT/'01_your_pursuers_only.pdf') as own, PdfPages(OUT/'02_pursuer_evader_separated.pdf') as split:
        for mode in ('fixed16','nominal200ms'):
            fig,axs=plt.subplots(1,3,figsize=(15,5.8),layout='constrained')
            for ax,arm in zip(axs,METHODS):
                summary=draw(ax,selected[mode,arm],'pursuer');ax.set_title(f'{METHODS[arm]}\n{summary["captures"]}/2 captures, {summary["decisions"]} decisions',fontsize=11)
            fig.suptitle(f'Your pursuers only | {mode} | same E49 hard start and evader | raw positions',fontsize=14)
            fig.supxlabel('Blue/green: pursuers. Gray squares: initial hidden evader truth for plotting only.\n'
                 'No heading filtering or curve smoothing; unchanged motion model. Evader paths are on the separate pages.',fontsize=9)
            own.savefig(fig);fig.savefig(OUT/'figures'/f'pursuers_only_{mode}.png',dpi=130);plt.close(fig)
            fig,axs=plt.subplots(2,3,figsize=(14,10),layout='constrained')
            for col,arm in enumerate(METHODS):
                draw(axs[0,col],selected[mode,arm],'pursuer');draw(axs[1,col],selected[mode,arm],'evader')
                axs[0,col].set_title(METHODS[arm]+' | pursuers',fontsize=11);axs[1,col].set_title('Same game | eight-heading evaders',fontsize=10)
            fig.suptitle(f'{mode} | Pursuers above, evaders below | actual unsmoothed displacements',fontsize=14)
            split.savefig(fig);fig.savefig(OUT/'figures'/f'separated_{mode}.png',dpi=120);plt.close(fig)
    diagnostics=[]
    with PdfPages(OUT/'03_actual_motion_directions.pdf') as pdf:
        fig,axs=plt.subplots(3,2,figsize=(10,12),subplot_kw=dict(projection='polar'),layout='constrained')
        for row,arm in enumerate(METHODS):
            for col,role in enumerate(('pursuer','evader')):
                a=np.array(angles[arm,role]);wrapped=(a+182.5)%360-182.5;counts,bins=np.histogram(wrapped,bins=np.arange(-182.5,182.5,5.))
                ax=axs[row,col];ax.bar(np.deg2rad((bins[:-1]+bins[1:])/2),counts/max(1,counts.sum()),width=np.deg2rad(5),color='#2876a6' if col==0 else '#dc7d29')
                on8=np.abs((a+22.5)%45-22.5)<1e-6
                diagnostics.append(dict(method=arm,role=role,moving_segments=len(a),eight_heading_fraction=float(on8.mean())))
                ax.set_title(f'{METHODS[arm]} | {role}\nExact8-heading fraction: {100*on8.mean():.3f}%',fontsize=10)
        fig.suptitle('Actual motion directions on all saved games | fixed16 and200ms combined\n8-heading constraint belongs to the evader; unchanged continuous pursuer methods',fontsize=13)
        pdf.savefig(fig);fig.savefig(OUT/'figures/motion_directions.png',dpi=120);plt.close(fig)
    with PdfPages(OUT/'04_git_historical_reference.pdf') as pdf:
        for file in sorted((OUT/'historical').glob('*.json')):
            row=read(file);fig,axs=plt.subplots(1,2,figsize=(12,6),layout='constrained');s=draw(axs[0],row,'pursuer');draw(axs[1],row,'evader')
            axs[0].set_title(f'd47d188 pursuers | {s["captures"]}/2 captures, {s["decisions"]} decisions')
            axs[1].set_title('Same fixed E49 evader')
            fig.suptitle(f'Historical reference | {row["task"]["world"]} | full-state root,1000 simulations,original controller',fontsize=12)
            fig.supxlabel(f'Not equal information/budget. Available V3 iteration45; common execution shield count: {s["pursuer_shields"]}.\n'
                 'Original direction preferences are retained, not filtered to make curves look different.',fontsize=9)
            pdf.savefig(fig);fig.savefig(OUT/'figures'/f'git_reference_{row["task"]["world"]}.png',dpi=120);plt.close(fig)
    (OUT/'direction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    shutil.copy2(ROOT/'current/e49_hard_initials_20261107/00_hard_initials.pdf',OUT/'00_same_hard_initials.pdf')


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42});run()
