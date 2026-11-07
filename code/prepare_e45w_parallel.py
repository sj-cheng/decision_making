"""Register immutable shared data and recipe for six E45-W experiments."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from e45f_scenarios import ID_FAMILIES, OOD_FAMILIES, new_id_case, shifted_case, geometry_key

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45w_joint_ppo_and_planning_20261103"
C = ROOT / "current/e45c_budget_scaling_20261028"
BASE = 14530261103


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False))


def main():
    path = OUT / "experiment.json"
    if path.exists():
        cfg=read(path)
        for row in cfg["train_entries"]+cfg["validation_entries"]+cfg["test_entries"]:
            assert sha(row["path"])==row["sha256"],row["path"]
        print("existing frozen data verified"); return
    OUT.mkdir(parents=True,exist_ok=True)
    known=set()
    for directory in ("e45b_role_free_joint_nte_20261027", "e45c_budget_scaling_20261028",
                      "e45f_generalization_benchmark_20261029", "e45g_rl_benchmark_20261030",
                      "e45j_repaired_benchmark_20261031", "e45v_fullstate_hasac_20261103"):
        for p in (ROOT/"current"/directory/"bank").rglob("*.json"):
            item=read(p)
            if isinstance(item,dict) and "map" in item: known.add(geometry_key(item["map"]))
    train=[r|{"sha256":sha(r["path"]),"suite":"training"} for r in read(C/"bank/manifest.json")["entries"] if r["split"]=="train"]
    assert len(train)==480
    entries=[]; attempts=[]
    for split,families,count in (("validation",ID_FAMILIES,4),("test",ID_FAMILIES,6),("test",OOD_FAMILIES,6)):
        for family in families:
            accepted=0; bins=set()
            for index in range(1500):
                if accepted==count: break
                attempt={"split":split,"family":family,"index":index}
                try:
                    case=(new_id_case if family in ID_FAMILIES else shifted_case)(family,index,split,BASE)
                    key=geometry_key(case["map"])
                    if key in known: raise ValueError("Duplicate canonical geometry")
                    category=(case["pattern"],float(case["speed"]))
                    # Test covers exactly each of three entry patterns at both speeds.
                    if split=="test" and category in bins:
                        attempts.append(attempt|{"accepted":False,"reason":"predeclared pattern/speed bin already filled"});continue
                except (AssertionError,ValueError) as exc:
                    attempts.append(attempt|{"accepted":False,"reason":repr(exc)});continue
                known.add(key); bins.add(category)
                target=OUT/"bank"/split/(case["id"]+".json");write(target,case)
                row={"id":case["id"],"path":str(target),"family":family,"pattern":case["pattern"],
                     "speed":case["speed"],"density":case["density"],"split":split,"suite":case["suite"],
                     "source_index":index,"sha256":sha(target),"canonical_geometry_sha256":key}
                entries.append(row);attempts.append(attempt|{"accepted":True,"id":case["id"]});accepted+=1
            assert accepted==count,(split,family,accepted)
            print("bank",split,family,accepted,flush=True)
    models={str(seed):{"pt":str(C/f"learning/x20/round_12/models/single_{seed}.pt"),
                       "npz":str(C/f"learning/x20/round_12/models/single_{seed}.npz")} for seed in (17,29,43)}
    models={k:v|{"pt_sha256":sha(v["pt"]),"npz_sha256":sha(v["npz"])} for k,v in models.items()}
    cfg={"experiment":"E45-W","scope":"six parallel bounded planning and centralized joint-PPO experiments",
         "train_entries":train,"validation_entries":[r for r in entries if r["split"]=="validation"],
         "test_entries":[r for r in entries if r["split"]=="test"],"models":models,
         "seeds":[17,29,43],"opponents":["fixed_D_kernel","rootpuct"],"root_simulations":3000,
         "shared_environment_seed_namespace":BASE,
         "seed_formula":"SeedSequence([14530261103,int(case_sha256[:8],16),int(world=='BA')]).generate_state(1)[0]",
         "worlds":["AB","BA"],"wall_budgets_ms":[50,100,200],"fixed_simulations":[16,64,128],
         "experiments":{"1":"frozen SinglePrior/Geom/U wall-budget curve", "2":"frozen Single 3 seeds, one-intent safe interface",
                        "3":"same frozen planners fixed simulations", "4":"paired end-to-end and nested cost measurement from experiment1",
                        "5":"structured Single/SB3 PPO probability, lifecycle, real-update acceptance",
                        "6":"B random and C pretrained centralized joint PPO, all three seeds"},
         "ppo":{"n_envs":8,"n_steps":256,"rollout_team_steps":2048,"rollouts_per_model":976,
                "team_steps_per_model":1998848,"total_team_steps_cap":11993088,
                "batch_size":256,"n_epochs":10,"learning_rate":.0003,"clip_range":.2,
                "gamma":1.,"gae_lambda":.95,"ent_coef":.001,"vf_coef":.5,"max_grad_norm":.5,
                "target_kl":.03,"normalize_advantage":True,"eval_every_rollouts":100,
                "arms":["B","C"],"provider":"stable-baselines3==2.7.1 PPO.train and collect unchanged",
                "critic":"separate random Network(single) per arm, same initial critic per seed, no old leaf value",
                "selection":"validation stochastic raw utility strictly greater; tie earlier including step0",
                "reward":"raw double-.0025*time + .05 public coverage potential difference; terminal Phi0; no blocked penalty",
                "action":"original Gaussian latent z in guarded finite Box +/-1e6; tanh and target-free safe mapping in env, clipping must be no-op"},
         "validation_recipe":{"groups":12,"stochastic_repeats":2,"deterministic_repeats":1,
                              "primary":"96 stochastic episodes at each checkpoint; deterministic48 descriptive, no mode selection"},
         "test_recipe":{"groups":42,"stochastic_repeats":2,"deterministic_repeats":1,
                        "policy_candidates":"A frozen, B/C selected and terminal, all3 seeds",
                        "results_release":"do not inspect planning/A test efficacy until all six learners/selection freeze"},
         "bank_generation":{"seed_base":BASE,"selection":"geometry, legal causal histories, canonical disjointness, predeclared test pattern/speed bins only",
                            "attempts":attempts},
         "new_test_is_not_for_configuration_selection":True,"automatic_extra_budget":False}
    write(path,cfg);print(json.dumps({"registered":True,"train":len(train),"validation":12,"test":42,"learning_cap":11993088}))


if __name__=="__main__": main()
