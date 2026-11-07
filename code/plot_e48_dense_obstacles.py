"""Vector PDFs from saved E48 data; no simulation or selection by outcomes."""
import os
os.environ['MPLBACKEND']='Agg'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from pathlib import Path
import json
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e48_dense_obstacles_20261107'
LABELS=dict(A2='Geometry ranking + search',A3='L5 learned ranking + search',
            A4='Geometry top, direct',A5='L5 learned top, direct',AG='Assigned geometry (AG)')
COLORS=['#e66101','#b2182b','#2166ac','#1b9e77']


def read(p):return json.loads(Path(p).read_text())


def setup(ax,case):
    for b in case['map']:
        b=np.asarray(b);ax.add_patch(Rectangle((b[0,0],b[1,0]),b[0,1]-b[0,0],b[1,1]-b[1,0],
            facecolor='#c7c9cc',edgecolor='#777a7e',lw=.8,zorder=0))
    ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',xlabel='x (m)',ylabel='y (m)')
    ax.grid(alpha=.14)


def layouts():
    cfg=read(OUT/'experiment.json');case=read(cfg['entries'][0]['path'])
    (OUT/'figures').mkdir(exist_ok=True)
    with PdfPages(OUT/'00_layouts.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(9,9),layout='constrained');setup(ax,case)
        for i,b in enumerate(case['map']):ax.text(*np.asarray(b).mean(1),str(i+1),ha='center',va='center',fontsize=9)
        s=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']=='AB'))
        for i in range(4):
            ax.scatter(*s[2*i:2*i+2],color=COLORS[i],marker='s' if i<2 else '^',s=75,
                       edgecolor='white',label=('E1','E2','P1','P2')[i],zorder=4)
        ax.legend(ncol=4);ax.set_title('E48 | Illustration-inspired 15 rectangles | map00, start AB\n'
            '2v2; arena20x20; known map; range6 + occlusion; 80-step survival/capture task')
        pdf.savefig(fig);fig.savefig(OUT/'figures/layout00.png',dpi=130);plt.close(fig)
        fig,axs=plt.subplots(3,4,figsize=(13,10),layout='constrained')
        for ax,entry in zip(axs.flat,[e for e in cfg['entries'] if e['world']=='AB']):
            setup(ax,read(entry['path']));ax.set_title(f'Map {entry["map_index"]:02d}');ax.set(xlabel='',ylabel='')
        fig.suptitle('E48 | All 12 layouts fixed before outcomes | positions/sizes jittered, no rotations')
        pdf.savefig(fig);plt.close(fig)


def selected():
    result={}
    for p in (OUT/'results').glob('*/*.json'):
        r=read(p);t=r['task']
        if t['entry']['map_index']==0 and t['world']=='AB' and t['train_seed']==17:
            result[t['stage'].removeprefix('E48_'),t['arm']]=r
    return result


def draw(ax,row,small=False):
    t=row['task'];case=read(t['case']);setup(ax,case)
    if not row['valid']:
        ax.text(0,0,'ENGINEERING INVALID\n'+row['error'],ha='center',wrap=True);return
    with gzip.open(row['result']['episode'],'rb') as f:payload=pickle.load(f)
    trace=payload['trace'];states=np.array([trace[0]['state']]+[v['next_state'] for v in trace])
    for i in range(4):
        retired=np.flatnonzero(states[:,9+i]<.5);stop=int(retired[0]) if len(retired) else len(states)-1
        xy=states[:stop+1,2*i:2*i+2]
        styles=['solid']*stop
        if i<2:styles=['solid' if any(v is not None for v in r['observation']['reports'][i]) else 'dashed' for r in trace[:stop]]
        ax.add_collection(LineCollection(np.stack([xy[:-1],xy[1:]],axis=1),colors=COLORS[i],
                          linewidths=1.5 if small else 2.2,linestyles=styles,label=('E1','E2','P1','P2')[i]))
        ax.scatter(*xy[0],color=COLORS[i],edgecolor='white',s=25 if small else 60,zorder=4)
        ax.scatter(*xy[-1],color=COLORS[i],marker='*' if len(retired) else 'X',edgecolor='white',
                   s=70 if small else 130,zorder=4)
        for j in range(4,len(xy),7):ax.annotate('',xy[j],xy[j-1],arrowprops=dict(arrowstyle='->',color=COLORS[i],lw=1.))
    s=payload['summary'];ax.set_title(f'{LABELS[t["arm"]]}\n{s["captures"]}/2 captures; {s["decisions"]} decisions',fontsize=10 if small else 14)
    if not small:
        ax.legend(ncol=4,loc='lower left');ax.text(.01,.99,f'Act P95: {np.percentile(s["latencies_ms"],95):.1f} ms\n'
             f'Act >200ms: {s["online_overruns"]}/{s["decisions"]}',transform=ax.transAxes,va='top',fontsize=9,
             bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))


def trajectories():
    rows=selected()
    if len(rows)!=10:raise RuntimeError(f'Preselected representative games not all available: {len(rows)}/10')
    with PdfPages(OUT/'02_representative_games.pdf') as atlas, PdfPages(OUT/'01_method_comparison.pdf') as overview:
        for mode in ('fixed16','nominal200ms'):
            fig,axs=plt.subplots(2,3,figsize=(13,9),layout='constrained')
            for ax,arm in zip(axs.flat,LABELS):draw(ax,rows[mode,arm],True)
            axs.flat[-1].axis('off');axs.flat[-1].text(0,.8,'Circle: start\nStar: captured/retired\nX: active final position\n'
                'Dashed E path: hidden to pursuer team\nAG includes target assignment\n\n'
                'Map00/startAB/seed17 fixed before outcomes\nQualitative examples, not aggregate evidence',va='top')
            fig.suptitle('E48 | '+mode+' | Frozen L5 and geometry | all15 obstacles known',fontsize=14)
            overview.savefig(fig);fig.savefig(OUT/'figures'/f'comparison_{mode}.png',dpi=120);plt.close(fig)
            for arm in LABELS:
                fig,ax=plt.subplots(figsize=(10,10),layout='constrained');draw(ax,rows[mode,arm])
                fig.suptitle('E48 | map00/startAB/seed17 | '+mode,fontsize=13)
                fig.supxlabel('15 axis-aligned rectangles; initial targets visible; later occlusion follows physics.\n'
                    'Shared belief and map. E48 is zero-shot dense-layout diagnosis; goal-line escape is not used.',fontsize=9)
                atlas.savefig(fig);plt.close(fig)


def results():
    r=read(OUT/'final/results.json');table={(v['mode'],v['arm']):v for v in r['table']}
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');names=list(LABELS)
    labels=['Geom+tree','L5+tree','Geom direct','L5 direct','AG']
    for col,mode in enumerate(('fixed16','nominal200ms')):
        for row,key,scale,title in [(0,'double_capture',100,'Double capture (%)'),(1,'completion_cost',1,'Completion cost (steps; failure penalized)')]:
            ax=axs[row,col];values=[table[mode,a][key]*scale if (mode,a) in table else np.nan for a in names]
            ax.bar(labels,values,color=['#5b8db8','#e8913a','#91b5d1','#f3ba7f','#58a88c'])
            ax.set_title(mode+' | '+title);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
            for i,v in enumerate(values):
                if np.isfinite(v):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
            if row==0:ax.set_ylim(0,110)
    fig.suptitle(f'E48 | 12 dense maps x2 starts | frozen models | invalid tasks: {r["invalid_count"]}\n'
        'Map-weighted descriptive results; inference and limitations in final/REPORT.md',fontsize=13)
    fig.savefig(OUT/'03_results.pdf');fig.savefig(OUT/'figures/results.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('layouts','all'));args=parser.parse_args()
    layouts()
    if args.mode=='all':trajectories();results()
