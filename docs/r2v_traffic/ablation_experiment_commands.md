# Ablation Experiment Commands

Run seed 0 first; expand to seeds 0, 1, and 2 after smoke/main succeed.

```bash
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export DEVICE="${DEVICE:-cuda}"
export ABLATION_ROOT="records/r2v_traffic_runs/ablation_jinan"
export TRANSITIONS="records/paper_final_data_buffers/paper_final_20260602_v1/jinan/*/seed0/transitions_raw.jsonl"
```

## Gate ablations

```bash
for gate in full no_support no_ood no_dynamics; do
  "${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
    --python_bin "${PYTHON_BIN}" \
    --transition_glob "${TRANSITIONS}" \
    --output_root "${ABLATION_ROOT}/gate_${gate}/seed0" \
    --seed 0 \
    --device "${DEVICE}" \
    --r2v on \
    --generative_backend diffusion \
    --repair_story not_rare_to_val \
    --repair_metadata_policy metadata_or_proxy \
    --gate_variant "${gate}" \
    --r2v_sampling_mode full_r2v \
    --force
done
```

## Story ablations

```bash
for story in not_rare_to_val not_val_to_val; do
  "${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
    --python_bin "${PYTHON_BIN}" \
    --transition_glob "${TRANSITIONS}" \
    --output_root "${ABLATION_ROOT}/story_${story}/seed0" \
    --seed 0 \
    --device "${DEVICE}" \
    --r2v on \
    --generative_backend diffusion \
    --repair_story "${story}" \
    --repair_metadata_policy metadata_or_proxy \
    --gate_variant full \
    --r2v_sampling_mode full_r2v \
    --force
done
```

## Sampling ablations

```bash
for mode in admitted_only random_same_count shuffled_value inverted_rarity; do
  "${PYTHON_BIN}" -m pareto.r2v.jinan_pair_ablation_runner \
    --python_bin "${PYTHON_BIN}" \
    --transition_glob "${TRANSITIONS}" \
    --output_root "${ABLATION_ROOT}/sampling_${mode}/seed0" \
    --seed 0 \
    --device "${DEVICE}" \
    --r2v on \
    --generative_backend diffusion \
    --repair_story not_rare_to_val \
    --repair_metadata_policy metadata_or_proxy \
    --gate_variant full \
    --r2v_sampling_mode "${mode}" \
    --force
done
```

Failure conditions: missing transition rows, malformed repair-story gate metadata when strict mode is requested, zero admitted samples for main full gate, pair validation failure, or non-finite metric output.
