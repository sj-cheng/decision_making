"""Read-only completion verification and standalone native-calibration figure."""
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45u_official_hasac_calibration_20261102"


def main():
    completion = json.loads((OUT / "completion_audit.json").read_text())
    assert completion["complete"]
    assert completion["decision"] == "native_learning_positive_two_seeds"
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    for name, row in manifest["files"].items():
        assert hashlib.sha256((OUT / "official_source" / name).read_bytes()).hexdigest() == row["sha256"]
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for ax, seed in zip(axes, (1, 2)):
        directory = OUT / f"seed_{seed}"
        status = json.loads((directory / "status.json").read_text())
        c = status["counts"]
        assert status["status"] == "complete" and status["unit_audit_passed"]
        assert c["team_transitions"] == 10010000 and c["train_calls"] == 500000
        assert c["sample_calls"] == 500000 and c["sampled_team_rows"] == 500000000
        assert c["agent_transitions"] == 30030000 and c["vector_ticks"] == 500500
        assert len(c["optimizer_steps"]) == 8 and set(c["optimizer_steps"].values()) == {500000}
        assert c["official_eval_calls"] == 50 and c["fixed_eval_episodes"] == 5700
        curve = [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines()]
        assert len(curve) == 51 and len(list((directory / "checkpoints").iterdir())) == 50
        assert [row["step"] for row in curve] == [0] + list(range(210000, 10010001, 200000))
        best = max(curve, key=lambda row: row["validation"]["mean"])
        test = json.loads((directory / "sealed_test.json").read_text())
        assert test["selected_step"] == best["step"]
        for key in ("validation_selected", "terminal"):
            delta = np.array(test[key]["returns"]) - np.array(test["initial"]["returns"])
            mean = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
            recorded = test[key + "_vs_initial"]
            assert len(delta) == 200 and np.isclose(mean, recorded["mean_improvement"])
            assert np.allclose([mean - 1.96 * se, mean + 1.96 * se], recorded["normal_approx_95ci"])
            assert mean - 1.96 * se > 0
        last = [row["validation"]["mean"] for row in curve[-10:]]
        records.append({"seed": seed, "elapsed_hours": status["elapsed_seconds"] / 3600,
                        "completed_beijing": datetime.datetime.fromtimestamp(
                            (directory / "status.json").stat().st_mtime,
                            datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                        "initial_test_mean": test["initial"]["mean"],
                        "selected_test_mean": test["validation_selected"]["mean"],
                        "terminal_test_mean": test["terminal"]["mean"],
                        "selected_step": test["selected_step"],
                        "last_10_validation_range": [min(last), max(last)]})
        x = np.array([row["step"] for row in curve]) / 1e6
        y = np.array([row["validation"]["mean"] for row in curve])
        ax.plot(x, y, label="Fixed validation (100 episodes)", color="#1464a5")
        ax.axhline(y[0], linestyle="--", color="gray", label="Untrained validation")
        ax.scatter([best["step"] / 1e6], [best["validation"]["mean"]],
                   color="#db7c14", marker="*", s=100, label="Validation-selected")
        ax.set(title=f"Seed {seed}", xlabel="Team transitions (millions, includes warmup)",
               ylabel="Mean episode team return (higher is better)", ylim=(-115, -50))
        ax.grid(alpha=.2); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Frozen official HASAC: native continuous MPE calibration")
    fig.savefig(OUT / "validation_curves.png", dpi=180)
    fig.savefig(OUT / "validation_curves.pdf")
    report = {"verified": True, "frozen_author_files_verified": len(manifest["files"]),
              "records": records, "new_training_updates": 0, "new_environment_episodes": 0,
              "decision": "go_to_full_observability_pursuit_interface_validation",
              "scope": "Native calibration passed; pursuit adaptation and retirement semantics remain unvalidated."}
    (OUT / "independent_completion_review_20261103.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
