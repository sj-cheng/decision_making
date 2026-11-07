"""Read-only tmux progress for the privileged pursuit diagnostic."""
import datetime
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "current/e45v_fullstate_hasac_20261103"
print("E45-V | full-state moving fixed-D pursuit diagnostic | tmux:train")
print(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"))
print("20 rollout envs / seed; 1M post-warmup team transitions; UTD .05; no teacher")
for seed in (1, 2):
    path = out / f"seed_{seed}/status.json"
    if not path.exists():
        print(f"\nseed {seed}: initializing/evaluating")
        continue
    row = json.loads(path.read_text()); c = row["counts"]
    warm = c.get("warmup_team_transitions")
    post = max(0, c["team_transitions"] - 10000)
    phase = row["status"] if warm is not None else "warmup/initial evaluation"
    print(f"\nseed {seed}: {phase} | post-warmup teams={post:,}/1,000,000")
    print(f" vector ticks={c['vector_ticks']:,}; train calls={c['train_calls']:,}; "
          f"optimizer steps={sum(c['optimizer_steps'].values()):,}; wall={row['elapsed_seconds']/60:.1f} min")
    if c["train_calls"] >= 1000:
        rate = post / row["elapsed_seconds"]
        remaining = max(0, 1000000 - post) / max(1e-9, rate)
        print(f" average={rate:.0f} team transitions/s; early linear remaining={remaining/3600:.1f} h")
    latest = row.get("latest_validation")
    if latest:
        score = latest["score"]
        print(f" validation step={latest['step']:,}: double={score['double_captures']}/{score['episodes']}; "
              f"first={score['first_capture_rate']:.1%}; utility={score['utility']:.4f}")
        print(f" validation-selected: {row['best_validation']}")
    if row.get("error"): print(" ERROR:", row["error"])
print("\nPrivileged diagnostic only; public-observation training is not launched automatically.")
completion = out / "completion_audit.json"
if completion.exists():
    print("Decision:", json.loads(completion.read_text())["decision"])
