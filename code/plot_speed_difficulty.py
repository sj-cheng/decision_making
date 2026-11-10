"""Plot saved speed-calibration results; never simulates or changes scores."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/speed_difficulty_mpl')
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e48_speed_difficulty_20261107'
COLORS={'A4':'#2e8b57','AG':'#ba7c00','A2':'#477bb6','A3':'#b64b75'}
LABELS={'A4':'Geometric direct','AG':'Assigned geometry','A2':'Geometric search','A3':'Frozen L5 search'}


def main():
    results=json.loads((OUT/'final/results.json').read_text())
    cfg=json.loads((OUT/'experiment.json').read_text())
    table=results['table'];speeds=sorted(set(r['evader_speed'] for r in table))
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    keys=[('double_capture','Both targets captured (%)'),('at_least_one','At least one capture (%)'),
          ('captured_fraction','Fraction of targets captured (%)'),('completion_cost','Failure-penalized completion steps')]
    for ax,(key,title) in zip(axes.flat,keys):
        for arm in COLORS:
            rows=sorted((r for r in table if r['arm']==arm and r['valid']),key=lambda r:r['evader_speed'])
            multiplier=100 if key!='completion_cost' else 1
            ax.plot([r['evader_speed'] for r in rows],[multiplier*r[key] for r in rows],
                    'o-',label=LABELS[arm],color=COLORS[arm])
        ax.axvline(1.3,color='gray',ls=':',lw=1)
        ax.set(title=title,xlabel='Evader per-axis displacement limit',xticks=speeds)
        if key!='completion_cost':ax.set_ylim(-3,103)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Exploratory speed calibration | pursuer limit 1.3 | 7 development maps, AB/BA\nFrozen model 17; nominal 200 ms; engineering failures reported separately',fontsize=11)
    for suffix in ('pdf','png'):
        fig.savefig(folder/f'00_capture_curves.{suffix}',dpi=160)
    plt.close(fig)

    categories=['double_capture','no_first_capture','post_capture_never_seen','post_capture_seen_not_caught']
    fills=['#84b89a','#d98e8e','#e4b268','#8ca3c7']
    fig,axes=plt.subplots(1,len(speeds),figsize=(4*len(speeds),4),squeeze=False,constrained_layout=True)
    for ax,speed in zip(axes.flat,speeds):
        bottom=np.zeros(4)
        for category,color in zip(categories,fills):
            values=[]
            for arm in COLORS:
                row=next(r for r in table if r['evader_speed']==speed and r['arm']==arm)
                values.append(row.get('failure_categories',{}).get(category,0))
            ax.bar(list(COLORS),values,bottom=bottom,color=color,label=category)
            bottom+=values
        ax.set(title=f'Evader limit {speed}',ylabel='Valid episodes',ylim=(0,14.8))
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Outcome/visibility categories from saved public observations')
    for suffix in ('pdf','png'):
        fig.savefig(folder/f'01_failure_types.{suffix}',dpi=160)
    plt.close(fig)

    # This map/world is fixed by family and ID, without looking at outcomes.
    entry=next(e for e in cfg['entries'] if e['family']=='multi_route')
    case=json.loads(Path(entry['path']).read_text())
    fig,axes=plt.subplots(len(speeds),4,figsize=(14,3.7*len(speeds)),squeeze=False,constrained_layout=True)
    for i,speed in enumerate(speeds):
        for j,arm in enumerate(COLORS):
            ax=axes[i,j]
            row=next(r for r in results['rows'] if r['task']['group']==entry['id']
                     and r['task']['world']=='AB' and r['task']['arm']==arm and r['task']['evader_speed']==speed)
            for box in case['map']:
                ax.add_patch(Rectangle((box[0][0],box[1][0]),box[0][1]-box[0][0],box[1][1]-box[1][0],facecolor='#d5d8dc',edgecolor='#687582',lw=.7))
            if row['valid']:
                with gzip.open(row['result']['episode'],'rb') as f:payload=pickle.load(f)
                trace=payload['trace'];states=np.array([trace[0]['state']]+[r['next_state'] for r in trace])
                for robot,color in enumerate(['#ee8a25','#bb3f4c','#3479b4','#339777']):
                    active_index=9+robot
                    last=next((k for k,s in enumerate(states) if s[active_index]<.5),len(states)-1)
                    xy=states[:last+1,2*robot:2*robot+2]
                    if robot<2:
                        for k in range(len(xy)-1):
                            seen=any(q is not None for q in trace[k]['observation']['reports'][robot])
                            ax.plot(xy[k:k+2,0],xy[k:k+2,1],color=color,ls='-' if seen else '--',lw=1.2)
                    else:ax.plot(xy[:,0],xy[:,1],color=color,lw=1.2)
                    ax.plot(*xy[0],marker='o',color=color,ms=3)
                    ax.plot(*xy[-1],marker='*' if states[last,active_index]<.5 else 'x',color=color,ms=6)
                summary=row['result']['summary']
                ax.set_title(f'{arm} | v_e={speed} | {summary["captures"]}/2, {summary["decisions"]} steps',fontsize=9)
            else:
                ax.set_title(f'{arm} | v_e={speed} | ENGINEERING FAILURE',fontsize=9)
            ax.set(xlim=(-10.5,10.5),ylim=(-10.5,10.5),aspect='equal');ax.grid(alpha=.15)
    fig.suptitle('Same predetermined multi-route map and AB initial state\nOrange/red: targets (dashed when unseen); blue/green: pursuers; stars: capture/retirement',fontsize=11)
    for suffix in ('pdf','png'):
        fig.savefig(folder/f'02_paired_trajectories.{suffix}',dpi=150)
    plt.close(fig)
    (folder/'provenance.json').write_text(json.dumps(dict(
        source_results_sha256=hashlib.sha256((OUT/'final/results.json').read_bytes()).hexdigest(),
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        example_selection=dict(group=entry['id'],world='AB',rule='fixed family multi_route, first registered ID, no outcome selection'),
        note='Charts summarize exploratory data; no scientific episodes rerun.'),indent=2)+'\n')


if __name__=='__main__':
    main()
