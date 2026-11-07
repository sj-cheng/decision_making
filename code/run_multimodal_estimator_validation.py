"""Frozen estimator-only development and unseen-history confirmation."""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from audit_l45_rollout_reliability import sha, save, observation
from estimator_audit_geometry import validate_paths, point_legal, blocked
from solvers.local_pursuit import PublicMap
from solvers.multimodal_estimator import Estimator, VERSION
from solvers.estimator_particle_reference import ParticleEstimator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT/"current/l45_multimodal_estimator_20261025"
OLD_BANK = ROOT/"current/l45_occlusion_scenarios_20261025/bank"
OLD_EPISODES = ROOT/"current/l45_occluded_comparison_20261025/episodes"
METHODS = ("particle", "systematic", "diffuse", "persistent", "feasible")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def source_files():
    names = ["run_multimodal_estimator_validation.py", "solvers/multimodal_estimator.py",
        "solvers/estimator_particle_reference.py", "solvers/history_recovery.py",
        "solvers/local_pursuit.py", "estimator_audit_geometry.py",
        "estimator_validation_scenarios.py", "verify_l45_occlusion_scenarios.py",
        "tests/test_multimodal_estimator.py", "L45_MULTIMODAL_ESTIMATION.md"]
    return {str(ROOT/"code"/name): sha(ROOT/"code"/name) for name in names}


def freeze(output):
    path = output/"experiment.json"
    bank = json.loads((output/"bank_frozen.json").read_text())
    for p,h in bank["artifacts"].items():
        if sha(Path(p)) != h:
            raise RuntimeError("Frozen confirmation bank changed")
    protocol = dict(version=VERSION, methods=METHODS, spacing=.5, sigma=.35, persistence=.9,
        particle_budget=64, kernel_sigma=.5, statistic_seed=71120261025, bootstrap=10000,
        sources=source_files(), confirmation_bank_sha256=sha(output/"bank_frozen.json"),
        original_history_files={str(p):sha(p) for p in sorted(OLD_EPISODES.glob("*_G.pt"))},
        no_mcts=True, no_actions=True, no_labels=True, no_neural_training=True,
        no_parameter_sweep=True, deployment_changed=False)
    # tuples normalize to JSON lists for exact resume comparisons.
    protocol = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("Frozen source/protocol changed; use new output")
    else:
        save(path,protocol)
    return protocol


def warmup():
    boxes = np.array([[[-1.,1.],[-5.,5.]]])
    first = observation(dict(step=0, position=[-3.,6.], target=[2.,6.], remaining=80))
    hidden = observation(dict(step=1, position=[-3.,5.], target=None, remaining=79))
    for method in METHODS:
        own = make_estimator(boxes,method,17)
        own.update(first)
        own.update(hidden)
        if isinstance(own,Estimator):
            validate_paths(own.witnesses(),np.array(own.observers),boxes)


def make_estimator(boxes,method,seed):
    return ParticleEstimator(boxes,seed,method=="systematic") if method in ("particle","systematic") else Estimator(boxes,method,seed)


def score(points, weights, samples, truth):
    weights = weights/weights.sum()
    sq = np.sum((points-truth)**2,axis=1)
    density = float(weights @ np.exp(-sq/.5))/(2*np.pi*.25)
    nll = -np.log(max(1e-12,density))
    # Aggregate before ranking: duplicated PF samples represent additional
    # probability mass, not extra equal-density locations.
    cells = np.floor((points+10.)/.5).astype(int)
    unique,inverse = np.unique(cells,axis=0,return_inverse=True)
    mass = np.bincount(inverse,weights=weights)
    order = np.argsort(-mass,kind="stable")
    stop = min(len(order),int(np.searchsorted(np.cumsum(mass[order]),.9))+1)
    selected = unique[order[:stop]]
    area = .25*len(selected)
    truth_cell = np.floor((truth+10.)/.5).astype(int)
    mean = weights @ points
    return dict(sample_coverage=float(np.min(np.linalg.norm(samples-truth,axis=1))<=1.),
        mass_1m=float(weights[sq<=1.].sum()), nll=float(nll), area90=float(area),
        coverage90=float(np.any(np.all(selected==truth_cell,axis=1))),
        mean_error=float(np.linalg.norm(mean-truth)),
        spread=float(np.sqrt(weights @ np.sum((points-mean)**2,axis=1))))


def old_cases():
    for path in sorted(OLD_EPISODES.glob("*_G.pt")):
        saved = torch.load(path,map_location="cpu",weights_only=False)
        task = saved["task"]
        packet = json.loads((OLD_BANK/task["entry"]["public"]).read_text())
        observations = [row["observation"] for row in saved["trajectory"]]
        truths = [np.asarray(row["state"]).reshape(-1)[:2].copy() for row in saved["trajectory"]]
        yield dict(key=task["key"],group=task["entry"]["id"],packet=packet,
            observations=observations,truths=truths,seed=packet["controller_seeds"][task["stream"]],
            historical_particles=[row["particles"] for row in saved["trajectory"]],
            original_failure=saved.get("failure") is not None)


def confirmation_cases(output):
    entries = json.loads((output/"bank/manifest.json").read_text())["entries"]
    for entry in entries:
        packet = json.loads((output/"bank"/entry["public"]).read_text())
        environment = json.loads((output/"bank"/entry["environment"]).read_text())
        sensor = PublicMap(packet["map"],packet["motion"]["pursuer_axis_limit"])
        for world in environment["worlds"]:
            for route,states in world["continuations"].items():
                # Truth is confined to the data provider and later scorer.
                # The estimator receives exactly these sensor observations.
                observations = [sensor.observe(s["step"],np.array(s["pursuer"]),np.array(s["evader"])) for s in states]
                truths = [np.array(s["evader"]) for s in states]
                yield dict(key=f'{entry["id"]}_{world["id"]}_{route}',group=entry["id"],packet=packet,
                    observations=observations,truths=truths,seed=packet["controller_seeds"][0],
                    historical_particles=None,original_failure=False)


def evaluate_case(case):
    # No estimator constructor/update accepts truths, world ID, route, future
    # observations or the full case. Labels stay in the outer evaluation code.
    packet, observations = case["packet"],case["observations"]
    boxes = np.array(packet["map"])
    estimators = {m:make_estimator(boxes,m,case["seed"]) for m in METHODS}
    failed, rows, init_ms = {},[],{}
    for method,own in estimators.items():
        started=time.perf_counter()
        try:
            for item in packet["history"][:-1]:
                own.update(observation(item))
        except (ValueError,RuntimeError) as exc:
            failed[method]=str(exc)
        init_ms[method]=1000*(time.perf_counter()-started)
    for i,obs in enumerate(observations):
        outputs={}
        for method,own in estimators.items():
            if method in failed:
                continue
            try:
                samples,sample_weights=own.update(obs)
                points,weights=own.distribution()
                paths=own.witnesses() if isinstance(own,Estimator) else own.paths
                hidden=obs.target is None
                if hidden and paths is not None:
                    observers=np.array(own.observers if isinstance(own,Estimator) else own.history)
                    violations,transitions=validate_paths(paths,observers,boxes)
                else:
                    violations,transitions=0,0
                if violations:
                    failed[method]=f"Independent witness violation: {violations}"
                    continue
                outputs[method]=(samples,points,weights,own.last_stats.copy(),transitions)
            except (ValueError,RuntimeError) as exc:
                failed[method]=str(exc)
        # Only after all estimator outputs are fixed, read this frame's truth.
        truth=case["truths"][i]
        for method in METHODS:
            own=estimators[method]
            common=dict(method=method,frame=i,hidden=obs.target is None,handoff=i==0)
            if method not in outputs:
                rows.append(common|dict(failed=True,error=failed.get(method),sample_coverage=0.,mass_1m=0.,
                    nll=float(-np.log(1e-12)),area90=400.,coverage90=0.,mean_error=30.,spread=30.,update_ms=0.,build_ms=0.,transitions=0))
                continue
            samples,points,weights,stats,transitions=outputs[method]
            metric=score(points,weights,samples,truth)
            exact=None
            if method=="particle" and case["historical_particles"] is not None:
                exact=bool(np.array_equal(samples,case["historical_particles"][i]))
            rows.append(common|metric|dict(failed=False,update_ms=stats["update_ms"],build_ms=stats["build_ms"],
                transitions=transitions,unique_samples=len(np.unique(samples,axis=0)),
                unique_ancestors=stats.get("unique_ancestors"),survivors=stats.get("survivors"),
                recovered=stats.get("recovered",False),historical_exact=exact,
                support_points=stats.get("support_points"),mode_mass=stats.get("mode_mass")))
    summary={}
    for method in METHODS:
        subset=[r for r in rows if r["method"]==method and r["hidden"]]
        fields=("sample_coverage","mass_1m","nll","area90","coverage90","mean_error","spread")
        summary[method]={k:float(np.mean([r[k] for r in subset])) if subset else 0. for k in fields}
        summary[method].update(handoff_coverage=next(r["sample_coverage"] for r in rows if r["method"]==method),
            failed=method in failed, failure=failed.get(method),init_ms=init_ms[method])
    return dict(key=case["key"],group=case["group"],original_failure=case["original_failure"],summary=summary,frames=rows)


def statistics(results):
    groups=sorted({r["group"] for r in results})
    fields=("sample_coverage","mass_1m","nll","area90","coverage90","mean_error","spread","handoff_coverage")
    values={m:{k:np.array([np.mean([r["summary"][m][k] for r in results if r["group"]==g]) for g in groups]) for k in fields} for m in METHODS}
    rng=np.random.default_rng(71120261025)
    indices=rng.integers(0,len(groups),(10000,len(groups)))
    comparison={}
    for other in ("particle","systematic","diffuse","feasible"):
        comparison["persistent-"+other]={}
        for field in fields:
            delta=values["persistent"][field]-values[other][field]
            boot=delta[indices].mean(axis=1)
            comparison["persistent-"+other][field]=dict(mean=float(delta.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist())
    frames=[f for r in results for f in r["frames"]]
    summary={}
    for m in METHODS:
        fs=[f for f in frames if f["method"]==m]
        times=[f["update_ms"] for f in fs if not f["failed"]]
        summary[m]={k:float(values[m][k].mean()) for k in fields}
        summary[m].update(failed_cases=sum(r["summary"][m]["failed"] for r in results),
            update_mean_ms=float(np.mean(times)),update_p95_ms=float(np.percentile(times,95)),
            build_mean_ms=float(np.mean([f["build_ms"] for f in fs if not f["failed"] and f["build_ms"]>0])) if any(f["build_ms"]>0 for f in fs if not f["failed"]) else 0.,
            independently_verified_transitions=sum(f["transitions"] for f in fs),
            original_particle_exact_frames=sum(f.get("historical_exact") is True for f in fs),
            unique_samples_mean=float(np.mean([f.get("unique_samples",0) for f in fs])),
            unique_ancestors_mean=float(np.mean([f["unique_ancestors"] for f in fs if f.get("unique_ancestors") is not None])) if m in ("particle","systematic") else None)
    checks=dict(no_failures=summary["persistent"]["failed_cases"]==0,
        handoff_coverage=summary["persistent"]["handoff_coverage"]>=.75,
        hidden_coverage=summary["persistent"]["sample_coverage"]>=.70,
        nll_vs_particle=comparison["persistent-particle"]["nll"]["ci95"][1]<0,
        nll_vs_feasible=comparison["persistent-feasible"]["nll"]["ci95"][1]<0,
        concentrated=summary["persistent"]["area90"]<=summary["feasible"]["area90"],
        runtime=summary["persistent"]["update_p95_ms"]<=10.)
    return dict(groups=len(groups),cases=len(results),unit="public_history_group",summary=summary,comparisons=comparison,
        checks=checks,passed=all(checks.values()))


def run(output,stage):
    protocol=freeze(output)
    warmup()
    path=output/(stage+"_results.json")
    if path.exists():
        raise RuntimeError("Completed stage cannot be overwritten")
    if stage=="confirmation" and not (output/"development_results.json").exists():
        raise RuntimeError("Development must finish before confirmation; no tuning after confirmation")
    iterable=old_cases() if stage=="development" else confirmation_cases(output)
    results=[]
    for case in iterable:
        cached=output/stage/(case["key"]+".json")
        record=json.loads(cached.read_text()) if cached.exists() else evaluate_case(case)
        if not cached.exists():
            save(cached,record)
        results.append(record)
        save(output/"status.json",dict(stage=stage,terminal=False,completed=len(results),total=96,pid=os.getpid()))
        if len(results)%8==0:
            print(f'{stage}: {len(results)}/96',flush=True)
    stats=statistics(results)
    stats["sources_unchanged"]=protocol["sources"]==source_files()
    if not stats["sources_unchanged"]:
        raise RuntimeError("Sources changed during evaluation")
    save(path,stats)
    save(output/"status.json",dict(stage=stage+"_completed",terminal=True,completed=len(results),total=96,pid=None,neural_training=False,search_comparison=False))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("development","confirmation"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args=parser.parse_args()
    run(args.output.resolve(),args.stage)
