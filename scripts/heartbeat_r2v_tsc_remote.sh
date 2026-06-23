#!/usr/bin/env bash
set -u

HB_HOST="${HB_HOST:-Ubuntu-Tailscale}"
HB_ROOT="${HB_ROOT:-/mnt/pan/r2v_tsc_runs/jinan_24h_campaign_20260622_020341}"
HB_INTERVAL="${HB_INTERVAL:-120}"
HB_STALL_SECS="${HB_STALL_SECS:-900}"
HB_MAX_SECONDS="${HB_MAX_SECONDS:-0}"
HB_EXIT_ON_COMPLETED="${HB_EXIT_ON_COMPLETED:-1}"

get_field() {
  printf '%s\n' "$1" | awk -v key="$2" '$1=="HB" && $2==key { $1=$2=""; sub(/^  */, ""); print; exit }'
}

previous_fp=""
stale_ticks=0
first_tick=1
start_epoch="$(date +%s)"

while true; do
  ts="$(date '+%F %T %Z')"
  snapshot="$(
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "$HB_HOST" \
      "ROOT='$HB_ROOT' bash -s" <<'REMOTE'
now="$(date +%s)"
echo "HB root $ROOT"
if [ ! -d "$ROOT" ]; then
  echo "HB root_ok 0"
  exit 0
fi
echo "HB root_ok 1"

status="missing"
phase="missing"
heartbeat_epoch=0
if [ -f "$ROOT/status.json" ]; then
  status="$(grep -m1 '"status"' "$ROOT/status.json" | sed 's/.*"status": *"\([^"]*\)".*/\1/')"
  phase="$(grep -m1 '"phase"' "$ROOT/status.json" | sed 's/.*"phase": *"\([^"]*\)".*/\1/')"
  heartbeat_epoch="$(grep -m1 '"updated_at_epoch"' "$ROOT/status.json" | sed 's/.*"updated_at_epoch": *\([0-9.]*\).*/\1/')"
fi
heartbeat_age="$(awk -v now="$now" -v hb="$heartbeat_epoch" 'BEGIN { if (hb <= 0) print -1; else print int(now - hb) }')"
echo "HB status $status"
echo "HB phase $phase"
echo "HB heartbeat_age $heartbeat_age"

proc_rows="$(
  for d in /proc/[0-9]*; do
    [ -r "$d/cmdline" ] || continue
    pid="${d##*/}"
    cmd="$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)"
    case "$cmd" in
      *"$ROOT"*|*"pareto.r2v.jinan_pair_ablation_runner"*|*"pareto.data.build_pairs"*|*"pareto.train_conditioned_scalar"*|*"formal_pilot_runner"*)
        printf '%s\t%s\n' "$pid" "$cmd"
      ;;
    esac
  done
)"
proc_count="$(printf '%s\n' "$proc_rows" | sed '/^$/d' | wc -l | tr -d ' ')"
pids="$(printf '%s\n' "$proc_rows" | awk -F '\t' '{printf "%s%s", sep, $1; sep=","}')"
echo "HB proc_count ${proc_count:-0}"
echo "HB pids ${pids:-none}"
printf '%s\n' "$proc_rows" | sed '/^$/d' | head -5 | sed 's/^/HB_PROC /'

driver="$ROOT/driver.log"
nohup="$ROOT.nohup.log"
latest_fp="none"
latest_age="-1"
latest_path="none"
if [ -f "$driver" ] || [ -f "$nohup" ]; then
  row="$(find "$driver" "$nohup" -maxdepth 0 -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -1)"
  if [ -n "$row" ]; then
    mt="$(echo "$row" | awk '{print int($1)}')"
    size="$(echo "$row" | awk '{print $2}')"
    path="$(echo "$row" | cut -d' ' -f3-)"
    latest_fp="$mt:$size:$path"
    latest_age="$((now - mt))"
    latest_path="$path"
  fi
fi
echo "HB log_fp $latest_fp"
echo "HB log_age $latest_age"
echo "HB log_path $latest_path"

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, 'BEGIN{u=0;m=0}{gsub(/ /,""); if($1>u)u=$1; if($2>m)m=$2}END{print u "," m}')"
  echo "HB gpu ${gpu:-unknown}"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d; s/^/HB_GPU_PROC /' | head -6
else
  echo "HB gpu unavailable"
fi

for f in "$driver" "$nohup"; do
  [ -f "$f" ] || continue
  tail -n 1200 "$f" 2>/dev/null | grep -nE 'Traceback \(most recent call last\)|CUDA out of memory|ModuleNotFoundError|AssertionError|RuntimeError|Segmentation fault|Killed|returncode=1' | tail -4 | sed "s|^|HB_TRACE $f: |"
done
REMOTE
  )"
  if [ $? -ne 0 ]; then
    echo "[$ts] BAD_SSH host=$HB_HOST root=$HB_ROOT"
    sleep "$HB_INTERVAL"
    continue
  fi

  root_ok="$(get_field "$snapshot" root_ok)"
  status="$(get_field "$snapshot" status)"
  phase="$(get_field "$snapshot" phase)"
  proc_count="$(get_field "$snapshot" proc_count)"
  proc_count="${proc_count:-0}"
  pids="$(get_field "$snapshot" pids)"
  log_fp="$(get_field "$snapshot" log_fp)"
  log_age="$(get_field "$snapshot" log_age)"
  heartbeat_age="$(get_field "$snapshot" heartbeat_age)"
  gpu="$(get_field "$snapshot" gpu)"
  trace_count="$(printf '%s\n' "$snapshot" | grep -c '^HB_TRACE ')"

  advanced=0
  if [ -n "$previous_fp" ] && [ "$log_fp" != "$previous_fp" ]; then
    advanced=1
  fi

  if [ "$root_ok" != "1" ]; then
    state="BAD_ROOT"
  elif [ "$trace_count" -gt 0 ]; then
    state="BAD_TRACEBACK"
  elif [ "$status" = "COMPLETED" ]; then
    state="COMPLETED"
  elif [ "$proc_count" -eq 0 ]; then
    state="NO_PROC"
    stale_ticks=0
  elif [ "$first_tick" -eq 1 ]; then
    state="BASELINE"
    stale_ticks=0
  elif [ "$advanced" -eq 1 ]; then
    state="OK_ADVANCING"
    stale_ticks=0
  else
    stale_ticks=$((stale_ticks + 1))
    if [ $((stale_ticks * HB_INTERVAL)) -ge "$HB_STALL_SECS" ]; then
      state="STALLED"
    else
      state="WARN_NO_DELTA"
    fi
  fi

  echo "[$ts] $state status=${status:-unknown} phase=${phase:-unknown} procs=$proc_count pids=${pids:-none} gpu=${gpu:-unknown} log_age=${log_age:-unknown}s heartbeat_age=${heartbeat_age:-unknown}s trace=$trace_count stale_ticks=$stale_ticks"
  printf '%s\n' "$snapshot" | grep -E '^(HB_PROC|HB_GPU_PROC|HB_TRACE)' | head -12 | sed 's/^/  /'
  previous_fp="$log_fp"
  first_tick=0
  if [ "$state" = "COMPLETED" ] && [ "$HB_EXIT_ON_COMPLETED" = "1" ]; then
    echo "[$(date '+%F %T %Z')] EXIT_COMPLETED root=$HB_ROOT"
    exit 0
  fi
  if [ "$HB_MAX_SECONDS" -gt 0 ]; then
    now_epoch="$(date +%s)"
    if [ $((now_epoch - start_epoch)) -ge "$HB_MAX_SECONDS" ]; then
      echo "[$(date '+%F %T %Z')] EXIT_MAX_SECONDS elapsed=$((now_epoch - start_epoch)) root=$HB_ROOT"
      exit 0
    fi
  fi
  sleep "$HB_INTERVAL"
done
