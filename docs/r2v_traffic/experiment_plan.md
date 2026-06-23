# R2V-Traffic Experiment Plan

## Scope

Compare `baseline_uniform` against `R2V-diffusion-not_rare_to_val-full` on Jinan traffic signal control. Smoke uses seed 0. Main uses seeds 0, 1, and 2. Metrics are average travel time, queue length, delay, throughput, and reward.

## Current code status

The command interface now parses and dry-runs for `--r2v off` and `--r2v on`. Gate variants are wired into candidate admission. Full traffic experiments still require CityFlow data, transition buffers, and optionally diffusion score artifacts on the experiment machine.

Experiment plans can be exported directly:

```bash
python3 -m pareto.r2v.traffic_experiment_plan \
  --plan main_pipeline \
  --format shell \
  --output records/r2v_traffic_runs/main_pipeline.sh
```

Use the generated plan as the canonical command order for main runs.

## Smoke

1. Run baseline-off dry-run.
2. Run R2V-on dry-run for `not_rare_to_val + full`.
3. If CityFlow is available, run a small environment smoke.
4. If a diffusion score artifact is available, pass it through `--r2v_artifact_path`; otherwise label the run as proxy, not diffusion.

## Main

Run:

- baseline: `--r2v off --r2v_sampling_mode off`;
- R2V: `--r2v on --generative_backend diffusion --repair_story not_rare_to_val --gate_variant full --r2v_sampling_mode full_r2v`.

Use identical map, seed, records, normalizer, training budget, and evaluation protocol.

## Ablations

Gate: `full`, `no_support`, `no_ood`, `no_dynamics`.

Story: `not_rare_to_val`, `not_val_to_val`.

Sampling: `admitted_only`, `random_same_count`, `shuffled_value`, `inverted_rarity`.

## Claim boundary

Dry-run and artifact validation are integrity evidence. A paper performance claim requires completed traffic metric rows from evaluation.
