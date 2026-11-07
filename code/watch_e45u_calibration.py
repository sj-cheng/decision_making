"""Compact, read-only native calibration progress for tmux."""
import datetime
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "current/e45u_official_hasac_calibration_20261102"
print("E45-U | official HASAC native continuous MPE | tmux:train")
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"))
print("Full recipe: 20 envs; 10M post-warmup team transitions / seed; UTD=0.05")
for seed in (1, 2):
    path = out / f"seed_{seed}/status.json"
    if not path.exists():
        print(f"seed {seed}: starting")
        continue
    row = json.loads(path.read_text())
    counts = row["counts"]
    post = counts["team_transitions"] - counts.get("warmup_team_transitions", 0)
    print(f"\nseed {seed}: {row['status']} | {post:,}/10,000,000 post-warmup team transitions")
    print(f" train calls={counts['train_calls']:,}; vector ticks={counts['vector_ticks']:,}; "
          f"optimizer steps={sum(counts['optimizer_steps'].values()):,}")
    print(f" elapsed={row['elapsed_seconds']/60:.1f} min; update/team={row['post_warmup_train_calls_per_team_transition']}")
    if post > 0 and row["status"] == "training":
        rate = post / row["elapsed_seconds"]
        remaining = max(0, 10000000 - post) / rate
        finish = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))) + datetime.timedelta(seconds=remaining)
        print(f" average={rate:.0f} team transitions/s; linear ETA={remaining/3600:.1f} h "
              f"({finish:%m-%d %H:%M} Beijing)")
    curve = out / f"seed_{seed}/validation.jsonl"
    rows = [json.loads(line) for line in curve.read_text().splitlines()] if curve.exists() else []
    if rows:
        print(f" validation: initial={rows[0]['validation']['mean']:.3f}; "
              f"latest={rows[-1]['validation']['mean']:.3f} at {rows[-1]['step']:,}; "
              f"best={row['best_validation']['validation_mean']:.3f}")
    if row.get("error"): print(" ERROR:", row["error"])
completion = out / "completion_audit.json"
if completion.exists():
    print("\nDecision:", json.loads(completion.read_text())["decision"])
else:
    print("\nLearning outcome pending; engineering smoke passed. No pursuit training launched.")
    print("ETA extrapolates observed average speed; larger replay and shared hardware can slow later training.")
