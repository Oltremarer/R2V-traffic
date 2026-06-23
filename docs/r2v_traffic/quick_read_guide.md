# R2V-Traffic Quick Read Guide

## Why port R2V to traffic light?

Traffic signal control learns from replayed experience, but ordinary replay often treats all samples too similarly. R2V-Traffic tries to give the learner better replay data without changing the learner itself.

## What is rare in traffic?

Rare can mean unusual queue patterns, phase-pressure mismatch, spillback-like congestion, or coordination patterns across intersections.

## What is valuable?

Valuable means the transition gives useful learning signal for better average travel time, shorter queues, lower delay, higher throughput, or better reward.

## Why rare is not valuable

A rare traffic state may be a crash-like jam, a bad controller action, or an off-support state. It can be important to inspect, but it should not automatically get more training weight.

In code, rarity is a detector/source/sampling signal, not a final admission gate. Final admission is decided by value/OOD, support, and dynamics/safety.

## What detector does

The detector finds candidate transitions. It can use feature-density proxy scores or a diffusion score artifact. It proposes candidates; it does not decide final value.

If no real score artifact is supplied, weighted artifacts are labeled with the proxy candidate model, such as `feature_density_proxy`, not as diffusion evidence. The `diffusion` backend label is reserved for rows joined from an actual score/repair artifact path.

## What repair/proposal does

`not_rare_to_val` starts from ordinary transitions and looks for potentially valuable admitted patterns. `not_val_to_val` starts from rare or low-value candidates and asks whether they can pass admission after proposal.

## What the three gates protect

Support gate protects behavior-support plausibility.

OOD/value gate protects against treating rarity as value.

Dynamics gate protects traffic consistency: legal phase/action, plausible queue evolution, next-state consistency, and reward consistency.

## How admission affects replay

Default admission is `weights_only`: admitted transitions get higher sample weights, while the original transition set stays unchanged. Downstream pair/replay sampling changes, but the base RL Bellman or PPO target is not rewritten.

There is also an opt-in `weights_plus_repaired` interface. It appends admitted repaired transition rows only when an admitted candidate has an explicit repaired transition payload, either from the diffusion score/repair artifact's `repaired_transition` field or from the source transition metadata. If a repaired proposal is explicit but still rejected by the gates, it can be appended as `r2v_row_role=repair_rejected` with `r2v_admitted=false` and `--repair_rejected_weight`; this keeps the generator proposal visible without pretending it passed final admission. The payload needs its own `transition_id` and `sample_id`; if an admitted candidate lacks it, or if any payload is malformed, the checks fail instead of fabricating enhanced data.

## How baseline and R2V compare

Baseline runs with `--r2v off --r2v_sampling_mode off`. R2V full runs with `--r2v on --generative_backend diffusion --repair_story not_rare_to_val --gate_variant full --r2v_sampling_mode full_r2v --r2v_admission_mode weights_only`.

When ER baseline modes are used, compare matched pairs such as `baseline_recent` versus `r2v_recent_full_r2v`. Do not compare a uniform baseline to an R2V overlay built on a different ER mode and call the difference purely R2V.

## How to run first

Check readiness first:

```bash
python3 -m pareto.r2v.experiment_readiness --scenario jinan --traffic_file anon_3_4_jinan_real.json --transition_glob "records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl" --seed 0 --output records/r2v_traffic_runs/readiness_seed0.json
```

Start with:

```bash
python3 -m pareto.r2v.jinan_pair_ablation_runner --dry_run --transition_glob "records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl" --output_root records/r2v_traffic_runs/smoke_baseline --r2v off --r2v_sampling_mode off --force
```

Then run the R2V smoke:

```bash
python3 -m pareto.r2v.build_generative_score_artifact --transitions records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl --output records/r2v_traffic_runs/smoke_r2v/artifacts/diffusion_seed0_proxy_scores.jsonl --summary_output records/r2v_traffic_runs/smoke_r2v/artifacts/diffusion_seed0_proxy_scores_summary.json --backend diffusion --adapter traffic_feature_density_proxy

python3 -m pareto.r2v.jinan_pair_ablation_runner --dry_run --transition_glob "records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl" --output_root records/r2v_traffic_runs/smoke_r2v --r2v on --generative_backend diffusion --r2v_artifact_path records/r2v_traffic_runs/smoke_r2v/artifacts/diffusion_seed0_proxy_scores.jsonl --repair_story not_rare_to_val --repair_metadata_policy metadata_or_proxy --gate_variant full --r2v_sampling_mode full_r2v --r2v_admission_mode weights_only --force
```

The generated proxy score artifact is for smoke/integration only and is marked `paper_claim_eligible=false`. Replace it with a real diffusion-produced artifact, and use `--repair_metadata_policy require_metadata`, before making paper claims.

For a paper run, readiness should include `--require_paper_claim_eligible_diffusion --repair_metadata_policy require_metadata --require_strict_repair_metadata_policy`; this blocks proxy artifacts and proxy repair metadata automatically and requires checkpoint/config/normalization provenance for every diffusion score row.

The generated `main_pipeline` plan also passes each seed's diffusion artifact into the R2V runner with `--r2v_artifact_path`. The final paper manifest checks that diffusion-labeled weighted transitions record score-artifact paths matching the bundled diffusion-score artifacts.

## Next experiments

First run smoke on one seed. Then run baseline vs R2V full on seeds 0, 1, and 2. After that, run gate, story, and sampling ablations. Only completed traffic metrics can support paper claims.

Before making the main paper table, run performance readiness with expected methods `baseline_uniform` and `r2v_diffusion_not_rare_to_val_full`, seeds `0,1,2`, all five metrics, and completed status.
