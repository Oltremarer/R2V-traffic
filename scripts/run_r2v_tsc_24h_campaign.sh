#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/chenyuyang/miniconda3/envs/c2t/bin/python}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/mnt/pan/r2v_tsc_runs/jinan_24h_campaign_$(date +%Y%m%d_%H%M%S)}"
DURATION_HOURS="${DURATION_HOURS:-24}"
EPOCHS="${EPOCHS:-40}"
RUN_BOUNDED_PPO="${RUN_BOUNDED_PPO:-1}"
MODES="${MODES:-full_r2v random_same_count admitted_only rare_only value_only shuffled_value inverted_rarity}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MAX_RUNS="${MAX_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$CAMPAIGN_ROOT/runs" "$CAMPAIGN_ROOT/reports"
DRIVER_LOG="$CAMPAIGN_ROOT/driver.log"
HISTORY_JSONL="$CAMPAIGN_ROOT/command_history.jsonl"
START_EPOCH="$(date +%s)"
DURATION_SECONDS="$((DURATION_HOURS * 3600))"
DEADLINE_EPOCH="$((START_EPOCH + DURATION_SECONDS))"

write_status() {
  local status="$1"
  local phase="$2"
  local current_run="${3:-}"
  local completed="${4:-0}"
  local failed="${5:-0}"
  "$PYTHON_BIN" - "$CAMPAIGN_ROOT/status.json" "$CAMPAIGN_ROOT/heartbeat.json" \
    "$status" "$phase" "$current_run" "$completed" "$failed" \
    "$START_EPOCH" "$DEADLINE_EPOCH" "$DURATION_HOURS" "$EPOCHS" "$RUN_BOUNDED_PPO" "$MODES" "$SEEDS" <<'PY'
import json
import sys
import time

status_path, heartbeat_path = sys.argv[1], sys.argv[2]
payload = {
    "status": sys.argv[3],
    "phase": sys.argv[4],
    "current_run": sys.argv[5],
    "completed_runs": int(sys.argv[6]),
    "failed_runs": int(sys.argv[7]),
    "started_at_epoch": int(sys.argv[8]),
    "deadline_epoch": int(sys.argv[9]),
    "duration_hours": int(sys.argv[10]),
    "epochs_per_run": int(sys.argv[11]),
    "run_bounded_ppo": sys.argv[12] == "1",
    "modes": sys.argv[13].split(),
    "seeds": [int(value) for value in sys.argv[14].split()],
    "updated_at_epoch": time.time(),
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
}
for path in (status_path, heartbeat_path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
PY
}

append_history() {
  local run_id="$1"
  local mode="$2"
  local seed="$3"
  local run_root="$4"
  local command="$5"
  "$PYTHON_BIN" - "$HISTORY_JSONL" "$run_id" "$mode" "$seed" "$run_root" "$command" <<'PY'
import json
import sys
import time

path, run_id, mode, seed, run_root, command = sys.argv[1:]
row = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    "time_epoch": time.time(),
    "run_id": run_id,
    "mode": mode,
    "seed": int(seed),
    "run_root": run_root,
    "command": command,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

run_one() {
  local round="$1"
  local mode="$2"
  local seed="$3"
  local run_index="$4"
  local effective_seed="$((seed + round * 1000))"
  local run_id
  run_id="$(printf 'run%03d_round%02d_seed%d_%s' "$run_index" "$round" "$effective_seed" "$mode")"
  local run_root="$CAMPAIGN_ROOT/runs/$run_id"
  local cmd=(
    "$PYTHON_BIN" -m pareto.r2v.jinan_pair_ablation_runner
    --python_bin "$PYTHON_BIN"
    --output_root "$run_root"
    --seed "$effective_seed"
    --epochs "$EPOCHS"
    --device "$DEVICE"
    --r2v_sampling_mode "$mode"
  )
  if [ "$RUN_BOUNDED_PPO" = "1" ]; then
    cmd+=(--run_bounded_ppo)
  fi
  local command_string
  command_string="$(printf '%q ' "${cmd[@]}")"
  append_history "$run_id" "$mode" "$effective_seed" "$run_root" "$command_string"
  write_status "RUNNING" "$mode" "$run_id" "$((run_index - 1))" "0"
  {
    printf '\n[%s] START %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$run_id"
    printf '%s\n' "$command_string"
  } >> "$DRIVER_LOG"
  local rc=0
  if [ "$DRY_RUN" = "1" ]; then
    local dry_cmd=(
      "$PYTHON_BIN" -m pareto.r2v.jinan_pair_ablation_runner
      --dry_run \
      --python_bin "$PYTHON_BIN" \
      --output_root "$run_root" \
      --seed "$effective_seed" \
      --epochs "$EPOCHS" \
      --device "$DEVICE" \
      --r2v_sampling_mode "$mode"
    )
    if [ "$RUN_BOUNDED_PPO" = "1" ]; then
      dry_cmd+=(--run_bounded_ppo)
    fi
    "${dry_cmd[@]}" >> "$DRIVER_LOG" 2>&1 || rc=$?
  else
    "${cmd[@]}" >> "$DRIVER_LOG" 2>&1 || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    {
      printf '[%s] END %s returncode=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$run_id" "$rc"
    } >> "$DRIVER_LOG"
    return "$rc"
  fi
  {
    printf '[%s] END %s returncode=0\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$run_id"
  } >> "$DRIVER_LOG"
}

completed_runs=0
failed_runs=0
run_index=1
round=0

write_status "RUNNING" "campaign_start" "" "$completed_runs" "$failed_runs"
{
  printf '[%s] CAMPAIGN_START root=%s duration_hours=%s epochs=%s bounded_ppo=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$CAMPAIGN_ROOT" "$DURATION_HOURS" "$EPOCHS" "$RUN_BOUNDED_PPO"
  printf '[%s] modes=%s seeds=%s max_runs=%s dry_run=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$MODES" "$SEEDS" "$MAX_RUNS" "$DRY_RUN"
} >> "$DRIVER_LOG"

while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do
  for seed in $SEEDS; do
    for mode in $MODES; do
      if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then
        break 3
      fi
      if [ "$MAX_RUNS" -gt 0 ] && [ "$run_index" -gt "$MAX_RUNS" ]; then
        write_status "COMPLETED" "max_runs_reached" "" "$completed_runs" "$failed_runs"
        exit 0
      fi
      if run_one "$round" "$mode" "$seed" "$run_index"; then
        completed_runs="$((completed_runs + 1))"
        write_status "RUN_DONE" "completed_run" "" "$completed_runs" "$failed_runs"
      else
        failed_runs="$((failed_runs + 1))"
        write_status "FAILED" "failed_run" "run${run_index}" "$completed_runs" "$failed_runs"
        exit 1
      fi
      run_index="$((run_index + 1))"
    done
  done
  round="$((round + 1))"
done

write_status "COMPLETED" "deadline_reached" "" "$completed_runs" "$failed_runs"
printf '[%s] CAMPAIGN_DONE completed_runs=%s failed_runs=%s\n' \
  "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$completed_runs" "$failed_runs" >> "$DRIVER_LOG"
