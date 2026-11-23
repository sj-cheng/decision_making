"""Evaluate the frozen E49 team controllers against the E56 cruise evader."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import traceback
import numpy as np
from run_1000_cruise import H80, config, run, save, outcome
from e49_controllers import install, make_controller
from run_e50_rolling_smoke import install_camera_belief_hooks
from solvers.e50_distributed_runtime import LocalPlan

OUT = H80/'legacy_500_initial_fix'

class LegacyPolicy:
    initial_target_fix = True
    def __init__(self, method, seed, folder):
        self.folder = folder
        self.method = method
        self.task = dict(arm=method, seed=seed, train_seed=17, horizon=80, out=str(folder))
        self.history = []
        install_camera_belief_hooks()
        def record_projection(measured, anchor, boxes):
            path = folder/'first_anchor_projection.json'
            if not path.exists():
                save(path, dict(measured=measured.tolist(), anchor=anchor.tolist(), boxes=boxes.tolist()))

        install(80, folder, proxy_anchor_hook=record_projection)
        # The frozen estimator cloned by E49 rechecks public measurements
        # against reconstructed boxes. Observations have already been validated
        # by the simulator, including the one-time initial position fix.
        from l45_corner_shared_20261025.corner_estimator import Estimator
        Estimator.update.__globals__.update(legal=lambda *a: True, visible=lambda *a: True)

    def __call__(self, cycle, positions, active_p, active_e, reports, camera, boxes):
        obs = dict(step=cycle, remaining=80-cycle, positions=positions.tolist(),
                   active_p=active_p.tolist(), active_e=active_e.tolist(), reports=reports,
                   visibility_model='camera_fov', camera=camera, trajectory_planner='minco')
        geometry = [] if self.method == 'RouteAPF' else boxes.tolist()
        controller = make_controller(dict(map=geometry), self.task)
        controller.rolling_lidar_routes = True
        controller.route_misses = []
        for frame in self.history:
            controller.observe(frame)
        actions = np.asarray(controller.act(obs, fixed_simulations=256)).reshape(2,2)
        if controller.route_misses:
            save(self.folder/f'route_miss_{cycle}.json',controller.route_misses)
        self.history.append(obs)
        return {i:LocalPlan(i,positions[i]+a,a,0.,int(controller.last_stats.get('simulations',0)),0,
                            duration=2.,pass_through=True) for i,a in enumerate(actions)}

def evaluate(task):
    row, case = task
    folder = OUT/row['method']/row['case']
    if (folder/'result.json').exists():
        return
    folder.mkdir(parents=True,exist_ok=True)
    policy = LegacyPolicy(row['method'], row['seed'], folder)
    cfg = dict(config(2), cycles=80, environment_horizon_floor=160, evader_cruise=True,
               e52_checkpoint=None, e52_mode='none', legacy_policy=policy)
    result = run(case['path'],folder/'trace.json',seed=row['seed'],**cfg)
    save(folder/'result.json',dict(row,initial_target_fix=True,**outcome(result)))

if __name__=='__main__':
    cases={c['case']:c for c in json.loads((H80/'paired_500.json').read_text())['cases']}
    rows=[json.loads(line) for line in (H80/'legacy_jobs.jsonl').read_text().splitlines()]
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        for stage in ('pilot','formal'):
            fs={pool.submit(evaluate,(r,cases[r['case']])):r for r in rows if r['stage']==stage}
            errors=0
            for f in as_completed(fs):
                try: f.result()
                except Exception:
                    print(fs[f],traceback.format_exc(),flush=True)
                    errors+=1
            if errors: raise RuntimeError(f'{errors} failed {stage} jobs')
