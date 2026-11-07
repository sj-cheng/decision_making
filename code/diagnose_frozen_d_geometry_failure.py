"""Descriptive replay of a stopped G failure; no controller changes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from numba import njit
from run_frozen_d_closed_loop import OUT,save,sha
from solvers.frozen_d_search_interface import PublicMap,geometric_candidates,sensor_visible,route
from l45_corner_shared_20261025.corner_estimator import geometry_labels


@njit(cache=True)
def scores(at,actions,samples,weights,speed,vertices,dist,boxes):
    result=np.zeros((len(actions),3));count=np.zeros(2,np.int64)
    for j in range(len(actions)):
        q=at+actions[j]
        for i in range(64):
            steps,_=route(q,samples[i],speed,vertices,dist,boxes,count)
            result[j,0]+=weights[i]*steps
            result[j,1]+=weights[i]*int(sensor_visible(q,samples[i],boxes))
        result[j,2]=-result[j,0]+result[j,1]
    return result


def main():
    rows=json.loads((OUT/'confirmation/results.json').read_text())
    key='multi_route_H6_h8.468628_v1.3_3506197457_A_G'
    row=next(r for r in rows if r['key']==key)
    episode=json.loads(Path(row['episode']).read_text());entry=episode['task']['entry']
    packet=json.loads((OUT/'bank'/entry['public']).read_text())
    public=PublicMap(packet['map'],packet['motion']['pursuer_axis_limit'])
    boxes=public.boxes;guard=boxes.copy();guard[:,:,0]-=1e-6;guard[:,:,1]+=1e-6
    records=[]
    selected=[i for i,r in enumerate(episode['trace']) if np.max(np.abs(r['own_action']))<1e-10][:3]
    for i in selected:
        frame=episode['trace'][i];samples=np.array(frame['particles']);weights=np.array(frame['weights'])
        labels=geometry_labels(samples,boxes)
        actions,chosen=geometric_candidates(np.array(frame['observation']['position']),samples,weights,
            labels,public.speed,public.vertices,public.dist,boxes,guard)
        np.testing.assert_array_equal(chosen,frame['own_action'])
        values=scores(np.array(frame['observation']['position']),actions,samples,weights,public.speed,public.vertices,public.dist,boxes)
        stop=int(np.argmin(np.linalg.norm(actions,axis=1)))
        records.append(dict(step=frame['step'],selected_action=frame['own_action'],geometry_replay_action=chosen.tolist(),
            actions=actions.tolist(),columns=['weighted_travel_steps','static_visible_mass','score'],scores=values.tolist(),
            stop_rank=int(np.sum(values[:,2]>values[stop,2]+1e-12))+1,
            strict_score_advantage=float(values[stop,2]-max(values[j,2] for j in range(len(values)) if j!=stop)),
            score_ties=int(np.sum(np.abs(values[:,2]-values[stop,2])<1e-12))))
    save(OUT/'geometry_failure_probe.json',dict(role='posthoc descriptive replay of first3 stopped frames in one failure; no policy changes or new confirmation',
        episode=key,episode_sha256=row['episode_sha256'],generator_sha256=sha(__file__),records=records))


if __name__=='__main__':main()
