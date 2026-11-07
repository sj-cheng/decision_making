"""User-facing six-method full-game comparison on fixed E49 input."""
import os
os.environ['MPLBACKEND']='Agg'
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import copy
import gzip
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import plot_e48_dense_obstacles as plot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e50_own_pursuers_20261107';E49=ROOT/'current/e49_hard_initials_20261107'
NAMES=dict(A3='L5 + search',R2='Single PRIOR + search',GIT='Git d47d188 search + learning',
           A2='Same-pool geometry + search',A4='Geometry direct',AG='Assigned geometry (AG)')


def read(p):return json.loads(Path(p).read_text())


def choose(mode):
    result={}
    for folder in [E49/'results'/f'E49_{mode}_geometry_escape',OUT/'results'/f'E50_{mode}']:
        for file in folder.glob('*.json'):
            r=read(file);t=r['task']
            if t['entry']['map_index']==0 and t['world']=='AB' and t['train_seed']==17 and t['arm'] in NAMES:
                assert r['valid'];result[t['arm']]=r
    legacy=read(OUT/'historical/e49_hard_dense_00_AB.json');payload=dict(summary=legacy['result']['summary'],trace=legacy['trace'])
    path=OUT/'plot_cache/git_map00_AB.pkl.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wb') as f:pickle.dump(payload,f)
    row=copy.deepcopy(legacy);row['result']['episode']=str(path);result['GIT']=row
    assert set(result)==set(NAMES)
    initial=None
    for row in result.values():
        with gzip.open(row['result']['episode'],'rb') as f:p=pickle.load(f)
        s=np.array(p['trace'][0]['state'])[:13]
        if initial is None:initial=s
        else:np.testing.assert_array_equal(initial,s)
    return result


def run():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    plot.LABELS.update(NAMES);(OUT/'figures').mkdir(exist_ok=True);index=[]
    with PdfPages(OUT/'00_strong_methods_full_games.pdf') as comparison, PdfPages(OUT/'00a_strong_methods_closeups.pdf') as closeups:
        for mode in ('fixed16','nominal200ms'):
            rows=choose(mode);fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained')
            for ax,arm in zip(axs.flat,NAMES):
                row=rows[arm];plot.draw(ax,row,True);s=row['result']['summary']
                budget=f'1000sim/full state; {s["pursuer_shields"]} execution safety stops' if arm=='GIT' else 'direct; no tree' if arm in ('A4','AG') else mode
                ax.set_title(f'{NAMES[arm]}\n{s["captures"]}/2 captures, {s["decisions"]} decisions\n{budget}',fontsize=10)
                index.append(dict(mode=mode,method=arm,captures=s['captures'],decisions=s['decisions'],budget=budget))
            fig.suptitle('Complex15-obstacle scene | your methods and strong references | same hard initial state\n'
                'Same E49 reactive evader; handoff at step12,68 steps remain. Panel budgets differ as labelled.',fontsize=14)
            handles=[Line2D([0],[0],color=c,lw=2,label=n) for c,n in zip(plot.COLORS,['E1','E2','P1','P2'])]
            axs.flat[0].legend(handles=handles,loc='upper left',ncol=2,fontsize=8,framealpha=.85)
            fig.supxlabel('Circle=start; star=captured/retired; X=active final; arrows=time direction. Dashed evader path=hidden to pursuer team.\n'
                'Raw unsmoothed trajectories. Git is a privileged historical reference, not an equal-budget ranking.',fontsize=9)
            comparison.savefig(fig);fig.savefig(OUT/'figures'/f'strong_methods_full_{mode}.png',dpi=135);plt.close(fig)
            for arm in NAMES:
                fig,ax=plt.subplots(figsize=(10,10),layout='constrained');plot.draw(ax,rows[arm],True)
                budget='Historical1000 simulations/full state' if arm=='GIT' else mode
                ax.set_title(f'{NAMES[arm]} | {budget}\n{rows[arm]["result"]["summary"]["captures"]}/2 captures; '
                             f'{rows[arm]["result"]["summary"]["decisions"]} decisions',fontsize=14)
                ax.legend(handles=handles,loc='lower left',ncol=4)
                fig.supxlabel('Map00/startAB selected before outcomes. Same fixed E49 evader.\n'
                    'Positions are actual simulator endpoints; no angular filtering or curve smoothing.',fontsize=9)
                closeups.savefig(fig);plt.close(fig)
    (OUT/'figure_index.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':run()
