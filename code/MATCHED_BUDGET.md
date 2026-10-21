# Matched-budget sweep

Run from the repository root. Each matchup is evaluated in episode chunks
by a process pool. Matchups and budgets run sequentially, so `WORKERS` is
the maximum concurrent worker count for the entire sweep.

## Parallel pilot

```bash
cd /home/sjc/decision_making
WORKERS=8 CHUNK_EPISODES=2 EPISODES=20 SEEDS=1 LARGE_SEEDS=1 \
BUDGETS="200 500" LARGE_BUDGET=200 \
OUTPUT_ROOT=current/benchmarks/matched_budget_parallel_pilot \
./code/run_matched_budget_sweep.sh
```

`LARGE_BUDGET=200` makes this a pipeline pilot. It is not the 50,000-simulation
reference used in the paper. The pursuit reference remains at 1,000 simulations.
To test only the machinery cheaply, set `BUDGETS=2 REFERENCE_BUDGET=2 LARGE_BUDGET=2`.

## Parallel performance experiment

```bash
WORKERS=32 CHUNK_EPISODES=5 EPISODES=200 SEEDS=5 LARGE_SEEDS=1 \
OUTPUT_ROOT=current/benchmarks/matched_budget_parallel_full \
./code/run_matched_budget_sweep.sh
```

On a machine with at least 32 available CPU cores, 32 workers is a reasonable
starting point. It will generally be faster than 8 workers, but it should not
be expected to produce a 4x speedup: process startup, model loading, cache and
memory contention, and uneven episode lengths limit scaling. Compare wall time
with the same `EPISODES`, `BUDGETS`, `CHUNK_EPISODES`, and seeds before choosing
the final value. On smaller machines, use `WORKERS` no larger than the available
cores. Budgets default to `200 500 1000 3000 5000`; the pursuit reference is fused
search at 1,000 simulations and the evasion reference is unguided search at
50,000 simulations. `LARGE_SEEDS` controls evaluation seeds for the latter
matchups. Increase it for independent initialization repeats.

For small pilots, use chunks of 1–2 episodes so enough tasks exist to occupy
the workers; for longer runs, chunks of 5–10 episodes reduce model-loading
overhead. Worker count is capped by the number of available chunks and CPUs.
Each process defaults to one PyTorch/BLAS/OpenMP CPU thread (`CPU_THREADS=1`).
Use `NO_PROGRESS=1` when redirecting output to a log.

## Serial timing

```bash
WORKERS=1 EPISODES=20 SEEDS=1 LARGE_SEEDS=1 \
OUTPUT_ROOT=current/benchmarks/matched_budget_serial_timing \
./code/run_matched_budget_sweep.sh
```

Stop competing benchmark/training jobs before this measurement. Parallel
results retain solver-call measurements for diagnostics, but they measure
latency under contention and must not be reported as isolated decision latency.
`config.json` records `parallel`, `workers`, `chunk_episodes`, `cpu_threads`,
`timing_context`, and the role-specific budgets. Use different output directories
for pilots, parallel performance runs, and serial timing runs.

Each matchup writes episode records and a summary CSV. Read
`pursuer_step_mean_latency_ms` for pursuit and `evader_step_mean_latency_ms`
for evasion: each is total measured role-call time divided by call count.
`WARMUP=5` currently excludes the first five steps of every episode, so these
are post-exclusion averages, not whole-episode latency averages. Existing
p50/p95 summary rows average episode-level quantiles, not pooled-call quantiles.

The launcher does not resume partially completed matchups or skip existing
results. Re-running into the same directory overwrites completed CSVs.
