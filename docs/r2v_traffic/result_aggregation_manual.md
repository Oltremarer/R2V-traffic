# Result Aggregation Manual

## Rule

Keep performance and integrity separate. Performance rows can include:

- `average_travel_time`;
- `queue_length`;
- `delay`;
- `throughput`;
- `reward`.

The aggregator also accepts legacy aliases:

- `test_avg_travel_time_over` -> `average_travel_time`;
- `test_avg_queue_len_over` -> `queue_length`;
- `test_avg_waiting_time_over` -> `delay`;
- `test_reward_over` -> `reward`.

`test_queuing_vehicle_num_over` is not throughput and is not mapped to throughput.

The output keeps counts explicit:

- `performance.row_count`: number of performance rows with at least one valid traffic metric;
- `performance.metric_value_count`: number of finite metric values consumed across those rows;
- `performance.by_method_row_count`: number of performance rows per method;
- `integrity.artifact_count`: number of integrity/status artifacts summarized separately.

Do not use `metric_value_count` as an experiment-count or seed-count proxy.

## Command

```bash
python3 -m pareto.r2v.result_aggregation \
  --performance_path records/r2v_traffic_runs/aggregation/r2v_performance_rows.jsonl \
  --integrity_path records/r2v_traffic_runs/main_jinan_3seed/seed0/r2v/artifacts/r2v_summary.json \
  --integrity_path records/r2v_traffic_runs/main_jinan_3seed/seed1/r2v/artifacts/r2v_summary.json \
  --integrity_path records/r2v_traffic_runs/main_jinan_3seed/seed2/r2v/artifacts/r2v_summary.json \
  --output records/r2v_traffic_runs/aggregation/r2v_result_aggregation.json
```

`--performance_path` and `--integrity_path` are repeatable. Use shell globs only if your shell expands them to concrete file paths before Python starts.

## Readiness

Paper table readiness requires 6 rows for baseline/R2V x 3 seeds, all five metrics finite, and completed evaluation status. Integrity artifacts should be summarized separately.

Use the readiness checker on performance rows before aggregation:

```bash
python3 -m pareto.r2v.experiment_readiness \
  --no-require_cityflow_data \
  --performance_path records/r2v_traffic_runs/aggregation/r2v_performance_rows.jsonl \
  --require_performance_metrics \
  --expected_performance_method baseline_uniform \
  --expected_performance_method r2v_diffusion_not_rare_to_val_full \
  --expected_performance_seed 0 \
  --expected_performance_seed 1 \
  --expected_performance_seed 2 \
  --require_completed_performance_status \
  --output records/r2v_traffic_runs/aggregation/performance_readiness.json
```

This fails if any row is missing `throughput`; `test_queuing_vehicle_num_over` is not treated as throughput. It also fails if the baseline/R2V x 3-seed grid is incomplete or any row status is not completed. Status remains a readiness field, not a performance metric.
