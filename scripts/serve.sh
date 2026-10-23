#!/bin/bash
# Host wrapper: launch the MiniMax-M3.1-Flash EXL3 serving container on ONE
# DGX Spark. Detached container; this script then (SERVE_WAIT=1, default)
# blocks until /v1/models answers, runs the warm-up (short + ~8K prompt) and
# prints the READY line. This recipe is STANDALONE: there is no external
# supervisor owning readiness.
#
#   PROFILE=q4-32k bash scripts/serve.sh     # default profile (32K q4 KV)
#   PROFILE=fp8-16k bash scripts/serve.sh
#   DRY_RUN=1 bash scripts/serve.sh          # print the commands, run nothing
#   bash scripts/guard.sh stop               # container down, sysctls restored
#
# shellcheck disable=SC2024  # only docker is sudo'd; the caller's own shell writes the files in its own $RUN
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

DRY_RUN="${DRY_RUN:-0}"
TS=$(date +%Y%m%d-%H%M%S)
RUN="${RUN_DIR:-$RUNS_DIR/$TS}"
# JIT caches (Triton, CUDA, vLLM). $CACHE_DIR is mounted at /cache, so the env
# vars below must name the CONTAINER path; a host path there puts every
# compiled kernel inside the container's own filesystem and it vanishes with
# it, and the next boot compiles everything again.
JIT_HOST="$CACHE_DIR/jit"
JIT="/cache/jit"

if [ "$DSPARK" = "1" ]; then
  SPEC="on (num_spec=$DSPARK_NUM_SPEC attn=$DSPARK_ATTN_BACKEND draft_kv=$DSPARK_KV_DTYPE draft=$DSPARK_DRAFT_SAMPLE reject=$DSPARK_REJECTION)"
else
  SPEC="off (no draft; slow sanity path)"
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: would launch container=$CONTAINER image=$IMAGE"
  echo "  profile: $PROFILE (configs/$PROFILE.env)"
  echo "  model  : $MODEL_DIR -> /models/$MODEL_NAME (read-only; verified by scripts/download_and_verify.sh)"
  if [ -n "$EXL3_REV" ]; then
    echo "  exl3   : $EXL3_SRC @ $EXL3_REV (required >= $EXL3_REQUIRED_REV; act-3 batched fix)"
  else
    echo "  exl3   : <unset> (no git checkout at EXL3_SRC=$EXL3_SRC; run build_image.sh)"
  fi
  if [ -n "$PLUGIN_REV" ]; then
    echo "  plugin : $PLUGIN_SRC @ $PLUGIN_REV (required >= $PLUGIN_REQUIRED_REV; mixed-K fix)"
  else
    echo "  plugin : <unset> (no git checkout at PLUGIN_SRC=$PLUGIN_SRC; run build_image.sh)"
  fi
  if [ -n "$PLUGIN_OVERLAY" ]; then
    echo "  overlay: $PLUGIN_OVERLAY -> /plugin:ro (PYTHONPATH=/plugin/src shadows the baked-in plugin)"
  else
    echo "  overlay: none (allowed only when the baked plugin is >= $PLUGIN_REQUIRED_REV)"
  fi
  echo "  uma    : LOAD_TRIM=$LOAD_TRIM EMBED_ALIAS=$EMBED_ALIAS DRAFT_VOCAB_META=$DRAFT_VOCAB_META TRIM_BEFORE_KV=$TRIM_BEFORE_KV MOE_TEMP_ROWS=$MOE_TEMP_ROWS PERIODIC_TRIM_S=$PERIODIC_TRIM_S JIT_WARMUP_THREADS=$JIT_WARMUP_THREADS COMPILE_THREADS=$COMPILE_THREADS (plugin uma_load_memory.py)"
  echo "  mixedk : unified (this recipe never sets VLLM_EXL3_MIXEDK_MODE; grouped = documented fallback only)"
  echo "  run dir: $RUN/ctr -> /mem (guard files stay in $RUN, outside the container)"
  echo "  jit    : $JIT_HOST -> $JIT (via $CACHE_DIR -> /cache)"
  echo "  port   : $API_PORT   maxlen: $MAXLEN   max_num_seqs: $MAX_NUM_SEQS"
  echo "  kv     : cache=$KV_CACHE_DTYPE indexer=$INDEXER_KV_DTYPE (nvfp4 is NOT available on GB10)"
  if [ "$INDEXER_MXFP8" = "1" ]; then
    echo "  indexer_mxfp8: INDEXER_MXFP8=1 -> VLLM_EXL3_INDEXER_MXFP8=1 in the container env (lightning-indexer keeps index_q/index_k MXFP8; preflight-checked in env.sh)"
  else
    echo "  indexer_mxfp8: INDEXER_MXFP8=0 -> VLLM_EXL3_INDEXER_MXFP8 unset"
  fi
  if [ "$KV_Q4" = "1" ]; then
    echo "  q4     : VLLM_EXL3_Q4KV=1 in the container env (main pool 576 B/token/layer via the"
    echo "  plugin q4 overlay pool); indexer stays fp8 e4m3 and the DSpark draft stays"
    echo "  fp8/TRITON_ATTN (fail-closed in env.sh)"
  fi
  if [ "$KVB_SRC" = "formula" ]; then
    echo "  pool   : $MAXLEN tokens x $KV_POOL_PER_TOKEN_B B/token x 1.10 (headroom) = \
$KVB_RAW B -> align up to 256 MiB -> \
$KVB B ($(( KVB / 1048576 )) MiB) via --kv-cache-memory-bytes"
  else
    if [ "$KVB_SRC" = "q4-kvb-py" ] || [ "$KVB_SRC" = "auto-kvb-py" ]; then
      echo "  pool   : KVB=$KVB B ($(( KVB / 1048576 )) MiB) via --kv-cache-memory-bytes (KVB_SRC=$KVB_SRC: scripts/kvb.py --maxlen $MAXLEN --q4 $KV_Q4; per-token cost + 2% margin; re-measure on the first boot)"
    else
      echo "  pool   : KVB=$KVB B ($(( KVB / 1048576 )) MiB) via --kv-cache-memory-bytes (KVB_SRC=$KVB_SRC: environment/profile override, formula NOT applied)"
    fi
  fi
  echo "  mnbt   : $MAX_NUM_BATCHED_TOKENS (chunked prefill; multiple of block size $BLOCK_SIZE)"
  echo "  dspark : $SPEC"
  echo "  autotune: FLASHINFER_AUTOTUNE=$FLASHINFER_AUTOTUNE (1 = engine-default FlashInfer autotune; 0 = --kernel-config enable_flashinfer_autotune:false)"
  echo "  warmup : WARMUP=$WARMUP SERVE_WAIT=$SERVE_WAIT (warm-up runs only when SERVE_WAIT=1: serve.sh then blocks until /v1/models answers)"
  echo "  sample : temperature/top_p defaults EMPTY (pack generation_config.json governs)"
  echo "  guard  : MemAvailable floor $((GUARD_MEM_FLOOR_KIB / 1024)) MiB, swap growth <= ${GUARD_SWAP_MAX_MIB} MiB"
  echo
  echo "vllm serve /models/$MODEL_NAME --served-model-name $SERVED_MODEL_NAME \\"
  echo "  --tensor-parallel-size 1 --language-model-only \\"
  echo "  --block-size $BLOCK_SIZE --disable-hybrid-kv-cache-manager \\"
  echo "  --max-model-len $MAXLEN --max-num-seqs $MAX_NUM_SEQS \\"
  echo "  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \\"
  echo "  --kv-cache-memory-bytes $KVB \\"
  echo "  --kv-cache-dtype $KV_CACHE_DTYPE \\"
  echo "  --attention-config {\"indexer_kv_dtype\":\"$INDEXER_KV_DTYPE\"} \\"
  echo "  --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$DSPARK_NUM_SPEC,\"attention_backend\":\"$DSPARK_ATTN_BACKEND\",\"kv_cache_dtype\":\"$DSPARK_KV_DTYPE\",\"draft_sample_method\":\"$DSPARK_DRAFT_SAMPLE\",\"rejection_sample_method\":\"$DSPARK_REJECTION\"} \\"
  if [ "$FLASHINFER_AUTOTUNE" = "0" ]; then
    echo "  --kernel-config {\"enable_flashinfer_autotune\":false} \\"
  fi
  echo "  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice \\"
  echo "  --override-generation-config {\"max_new_tokens\": $DEFAULT_MAX_TOKENS}"
  echo "  (no --gpu-memory-utilization anywhere: on GB10 the pool is sized by --kv-cache-memory-bytes)"
  echo "  (no draft model path anywhere: the engine resolves dspark/ from the pack)"
  exit 0
fi

# ---- model pack ------------------------------------------------------------
[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR" >&2; echo "run: bash scripts/download_and_verify.sh (downloads the pinned revision and verifies every file)" >&2; exit 1; }
[ -f "$MODEL_DIR/dspark/config.json" ] || { echo "dspark draft missing in $MODEL_DIR: the pack is incomplete." >&2; echo "run: bash scripts/download_and_verify.sh" >&2; exit 1; }
[ -f "$MODEL_DIR/model-00001-of-00016.safetensors" ] || { echo "weight shards missing in $MODEL_DIR: the pack is incomplete." >&2; echo "run: bash scripts/download_and_verify.sh" >&2; exit 1; }
# Byte verification is download_and_verify.sh's job (it hashes all ~119 GB);
# serve.sh only refuses a missing/incomplete directory so a boot never starts
# on a half download.

# ---- pinned checkouts + image labels ---------------------------------------
[ -n "$EXL3_REV" ] || { echo "no fork checkout at EXL3_SRC=$EXL3_SRC (see the README" >&2; echo "quick start); the act-3 fix is required, and serve.sh needs the checkout to" >&2; echo "verify the image label against the pin." >&2; exit 1; }
[ -n "$PLUGIN_REV" ] || { echo "no plugin checkout at PLUGIN_SRC=$PLUGIN_SRC (see the README" >&2; echo "quick start); the mixed-K fix is required, and serve.sh needs the checkout" >&2; echo "to verify the image label against the pin." >&2; exit 1; }
if [ -n "$PLUGIN_OVERLAY" ] && [ ! -d "$PLUGIN_OVERLAY/src" ]; then
  echo "PLUGIN_OVERLAY=$PLUGIN_OVERLAY has no src/ dir: not a plugin checkout." >&2
  exit 1
fi

sudo -n docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not found: $IMAGE (run scripts/build_image.sh)" >&2; exit 1; }
# The image must be built from this recipe's pins, and the runtime plugin
# must carry the mixed-K fix:
#  - the compiled exllamav3 extension must carry the act-3 batched fix (an
#    overlay cannot fix csrc), so the exllamav3 label must be at or after
#    the pinned fork rev;
#  - the plugin .py must carry the mixed-K unified K-array fix, either baked
#    into the image or overlaid at runtime.
img_label() { sudo -n docker image inspect "$IMAGE" --format "{{index .Config.Labels \"$1\"}}" 2>/dev/null; }
RAW_EXL3=$(img_label local.m31.exl3.exllamav3)
IMG_EXL3="${RAW_EXL3#local:}"
IMG_PLUGIN=$(img_label local.m31.exl3.vllm-exl3)
if ! git -C "$EXL3_SRC" merge-base --is-ancestor "$EXL3_REQUIRED_REV" "$EXL3_REV" 2>/dev/null; then
  echo "EXL3_SRC=$EXL3_SRC (HEAD $EXL3_REV) is below the act-3 fix pin" >&2
  echo "$EXL3_REQUIRED_REV: refusing to serve. Point EXL3_SRC at the" >&2
  echo "feat/moe-act3-batched checkout." >&2
  exit 1
fi
if [ -z "$IMG_EXL3" ] || ! git -C "$EXL3_SRC" merge-base --is-ancestor "$EXL3_REQUIRED_REV" "$IMG_EXL3" 2>/dev/null; then
  echo "image $IMAGE was not built from this recipe's fork pin:" >&2
  echo "  exllamav3 ${IMG_EXL3:-<none>} (want >= $EXL3_REQUIRED_REV)" >&2
  exit 1
fi
# plugin_at_least <full sha>: true when <sha> is the pinned rev or a
# descendant; shas unknown to the pinned checkout fail the test (fail-closed).
plugin_at_least() { git -C "$PLUGIN_SRC" merge-base --is-ancestor "$PLUGIN_REQUIRED_REV" "$1" 2>/dev/null; }
if plugin_at_least "$IMG_PLUGIN"; then
  # The baked plugin carries the fix: the overlay is optional and, if set,
  # must itself be at or after the fix.
  if [ -n "$PLUGIN_OVERLAY" ] && ! git -C "$PLUGIN_OVERLAY" merge-base --is-ancestor "$PLUGIN_REQUIRED_REV" HEAD 2>/dev/null; then
    echo "PLUGIN_OVERLAY=$PLUGIN_OVERLAY predates the mixed-K fix" >&2
    echo "$PLUGIN_REQUIRED_REV: unmount it or point it at the fixed branch." >&2
    exit 1
  fi
else
  # Pre-fix plugin baked: the mixed-K fix MUST come from the runtime overlay.
  if [ -z "$PLUGIN_OVERLAY" ] || ! git -C "$PLUGIN_OVERLAY" merge-base --is-ancestor "$PLUGIN_REQUIRED_REV" HEAD 2>/dev/null; then
    echo "image $IMAGE bakes plugin ${IMG_PLUGIN:-<none>}, which predates the" >&2
    echo "mixed-K fix $PLUGIN_REQUIRED_REV, and no fixed overlay is mounted:" >&2
    echo "every MoE layer would decode as garbage. Rebuild the image from the" >&2
    echo "pinned checkouts (run build_image.sh) or point PLUGIN_OVERLAY at a" >&2
    echo "checkout at $PLUGIN_REQUIRED_REV or a descendant." >&2
    exit 1
  fi
fi

mkdir -p "$RUN/ctr" "$JIT_HOST"/{triton,nv-compute,vllm}

# ---- host memory policy (restored by guard.sh stop) ------------------------
SYSCTL_KEYS="vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness"
: >"$RUN/sysctl.before"
for k in $SYSCTL_KEYS; do
  printf '%s=%s\n' "$k" "$(cat "/proc/sys/${k//.//}")" >>"$RUN/sysctl.before"
done

# Only `guard.sh stop` restores the host sysctls. If this script aborts after
# changing them but before the container exists (denied sudo rule, docker run
# failure, ...), the host was left at the serving policy with nothing running.
restore_sysctls_on_abort() {
  [ -s "${RUN:-}/sysctl.before" ] || return 0
  local k v
  while IFS='=' read -r k v; do
    case "$k" in
      vm.compaction_proactiveness|vm.watermark_boost_factor|vm.swappiness) ;;
      *) continue ;;
    esac
    case "$v" in ''|*[!0-9]*) continue ;; esac
    sudo -n sysctl -q "$k=$v" >/dev/null 2>&1 || true
  done <"$RUN/sysctl.before"
  echo "serve.sh aborted before the container was up; restored the host sysctls saved in $RUN/sysctl.before" >&2
}
trap restore_sysctls_on_abort EXIT

sudo -n sysctl -q vm.compaction_proactiveness=0 vm.watermark_boost_factor=0 vm.swappiness=0
# swapoff fails when any page is still swapped in; warn and keep the pre-boot
# base rather than aborting after the sysctls were changed.
if ! sudo -n swapoff -a; then
  echo "WARNING: swapoff -a failed; swap is still in use. Continuing, but the" >&2
  echo "         guard's swap-growth check will measure from a non-zero base." >&2
fi
sudo -n swapon -a 2>/dev/null || true
sync
# `sudo sh -c` needs a sudoers rule for sh itself; `sudo tee` needs only a rule
# for tee.
echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
sleep 3

BASE_SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo)
echo "MemAvailable: $(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB  swap_base: $BASE_SWAP KiB"

sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

UMA_ENV=()
[ "$LOAD_TRIM" = "1" ] && UMA_ENV+=(-e VLLM_EXL3_TRIM_AFTER_LAYER=1)
if [ "$EMBED_ALIAS" = "1" ]; then
  UMA_ENV+=(-e VLLM_EXL3_UMA_POSTLOAD=1 -e VLLM_EXL3_ATS_ALIAS=1 -e VLLM_EXL3_ATS_ALIAS_EMBED=1)
fi
[ "$DRAFT_VOCAB_META" = "1" ] && UMA_ENV+=(-e VLLM_EXL3_DRAFT_VOCAB_META=1)
[ "$TRIM_BEFORE_KV" = "1" ] && UMA_ENV+=(-e VLLM_EXL3_TRIM_BEFORE_KV=1)
[ -n "$MOE_TEMP_ROWS" ] && UMA_ENV+=(-e VLLM_EXL3_MOE_TEMP_ROWS="$MOE_TEMP_ROWS")
[ -n "$PERIODIC_TRIM_S" ] && UMA_ENV+=(-e VLLM_EXL3_PERIODIC_TRIM_S="$PERIODIC_TR_S")
[ -n "${MALLOC_ARENA_MAX:-}" ] && UMA_ENV+=(-e MALLOC_ARENA_MAX="$MALLOC_ARENA_MAX")
[ -n "$JIT_WARMUP_THREADS" ] && UMA_ENV+=(-e VLLM_TRITON_JIT_WARMUP_NUM_THREADS="$JIT_WARMUP_THREADS")
[ -n "$COMPILE_THREADS" ] && UMA_ENV+=(-e TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS")
# EXTRA_ENV: space-separated KEY=VALUE pairs passed to the container as-is
# (diagnostics, e.g. EXTRA_ENV="VLLM_EXL3_MEM_PROBE_TENSORS=1").
for kv in ${EXTRA_ENV:-}; do UMA_ENV+=(-e "$kv"); done
# Runtime plugin overlay (empty = use the plugin baked into the image). The
# overlay carries plugin PYTHON only; the compiled extension comes from the
# image either way.
OVERLAY_MOUNT=()
OVERLAY_ENV=()
if [ -n "$PLUGIN_OVERLAY" ]; then
  OVERLAY_MOUNT=(-v "$PLUGIN_OVERLAY":/plugin:ro)
  OVERLAY_ENV=(-e PYTHONPATH=/plugin/src)
fi
# q4 main KV (KV_Q4, env.sh): the plugin installs the q4 reroute ONLY at
# VLLM_EXL3_Q4KV=1. KV_Q4=0 exports nothing: the container env is identical
# to the fp8 recipe.
Q4KV_ENV=()
if [ "$KV_Q4" = "1" ]; then
  Q4KV_ENV=(-e VLLM_EXL3_Q4KV=1)
fi
# Indexer lightning-indexer MXFP8 (INDEXER_MXFP8, env.sh): the plugin keeps
# index_q/index_k MXFP8 on device ONLY at VLLM_EXL3_INDEXER_MXFP8=1;
# env.sh's preflight has already refused a 1 whose runtime plugin tree lacks
# that code. INDEXER_MXFP8=0 exports nothing.
INDEXER_MXFP8_ENV=()
if [ "$INDEXER_MXFP8" = "1" ]; then
  INDEXER_MXFP8_ENV=(-e VLLM_EXL3_INDEXER_MXFP8=1)
fi

sudo -n docker run -d --name "$CONTAINER" \
  --gpus all \
  --memory=118g --memory-swap=118g --oom-score-adj 1000 --ulimit core=0 \
  --shm-size=2g \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --ulimit=memlock=-1:-1 \
  -v "$MODELS_DIR":/models:ro \
  -v "$CACHE_DIR":/cache:rw \
  -v "$RUN/ctr":/mem:rw \
  -v "$HERE/serve_inner.sh":/root/serve.sh:ro \
  "${OVERLAY_MOUNT[@]}" \
  -p "$API_PORT":8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  -e DEFAULT_MAX_TOKENS="$DEFAULT_MAX_TOKENS" \
  -e DEFAULT_TEMPERATURE="$DEFAULT_TEMPERATURE" -e DEFAULT_TOP_P="$DEFAULT_TOP_P" \
  -e MAXLEN="$MAXLEN" -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e BLOCK_SIZE="$BLOCK_SIZE" \
  -e CG_MODE="${CG_MODE:-}" -e CAPTURE_SIZES="${CAPTURE_SIZES:-}" -e COMPILE_MODE="${COMPILE_MODE:-}" \
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e INDEXER_KV_DTYPE="$INDEXER_KV_DTYPE" \
  "${Q4KV_ENV[@]}" \
  "${INDEXER_MXFP8_ENV[@]}" \
  -e KVB="$KVB" -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e FLASHINFER_AUTOTUNE="$FLASHINFER_AUTOTUNE" \
  "${UMA_ENV[@]}" \
  "${OVERLAY_ENV[@]}" \
  -e DSPARK="$DSPARK" -e DSPARK_NUM_SPEC="$DSPARK_NUM_SPEC" \
  -e DSPARK_ATTN_BACKEND="$DSPARK_ATTN_BACKEND" -e DSPARK_KV_DTYPE="$DSPARK_KV_DTYPE" \
  -e DSPARK_DRAFT_SAMPLE="$DSPARK_DRAFT_SAMPLE" -e DSPARK_REJECTION="$DSPARK_REJECTION" \
  -e VLLM_EXL3_MODEL_DIR="/models/$MODEL_NAME" \
  -e TRITON_CACHE_DIR="$JIT/triton" \
  -e CUDA_CACHE_MAXSIZE=4294967296 -e CUDA_CACHE_PATH="$JIT/nv-compute" \
  -e XDG_CACHE_HOME="$JIT" -e VLLM_CACHE_ROOT="$JIT/vllm" \
  --entrypoint /bin/bash \
  "$IMAGE" /root/serve.sh \
  >"$RUN/container_id.txt" 2>&1 || {
    echo "docker run failed:" >&2
    cat "$RUN/container_id.txt" >&2
    exit 1
  }

echo "LAUNCHED $CONTAINER ($(cat "$RUN/container_id.txt"))"
echo "run dir: $RUN"
echo "autotune: FLASHINFER_AUTOTUNE=$FLASHINFER_AUTOTUNE   warmup: WARMUP=$WARMUP"
# The container exists: from here only `guard.sh stop` may restore the host.
trap - EXIT

# ---- stage breadcrumbs -----------------------------------------------------
echo "preflight" >"$RUN/stage"
# Detached (setsid nohup) so it outlives this shell and a terminal logout; it
# ends when the container's log stream ends. The watcher must not inherit the
# caller's open files.
no_inherited_fds() {
  local fd
  # shellcheck disable=SC2045  # fd names are always numeric and we WANT word splitting here
  for fd in $(ls /proc/$BASHPID/fd); do
    [ "$fd" -gt 2 ] && eval "exec $fd>&-" 2>/dev/null
  done
  exec "$@"
}
READY_MEM_AWK='/^Cached:/{c=$2} /^MemFree:/{f=$2} END{printf "page cache %.2f GiB, free %.2f GiB\n", c/1048576, f/1048576}'
(no_inherited_fds setsid nohup bash -c '
  sudo -n docker logs -f "$1" 2>&1 | while IFS= read -r line; do
    case "$line" in
      *"Starting to load model"*)       echo "load-weights" >"$2/stage";;
      *"Graph capturing"*)              echo "graph-capture" >"$2/stage";;
      *"init engine"*)                  echo "engine-init" >"$2/stage";;
      *"Application startup complete"*) echo "ready" >"$2/stage"
        awk "$3" /proc/meminfo >"$2/ready-mem"
        cat "$2/ready-mem" >&2 ;;
    esac
  done' _ "$CONTAINER" "$RUN" "$READY_MEM_AWK") >/dev/null 2>&1 &
echo "stage watcher: pid $!"

# ---- memory guard ---------------------------------------------------------
GUARD_FLOOR_KIB="$GUARD_MEM_FLOOR_KIB"
GUARD_SWAP_MAX_KIB=$((GUARD_SWAP_MAX_MIB * 1024))
(no_inherited_fds sudo -n setsid nohup bash "$HERE/guard.sh" watch "$CONTAINER" "$RUN/stage" "$RUN/guard.log" "$GUARD_FLOOR_KIB" "$BASE_SWAP" "$GUARD_SWAP_MAX_KIB") \
  >/dev/null 2>&1 &
# The guard itself waits up to 24 s for the container to appear before it
# arms; cover its full window before declaring UNGUARDED.
GUARD_ARMED=0
for _ in $(seq 1 300); do
  grep -q "guard: armed" "$RUN/guard.log" 2>/dev/null && { GUARD_ARMED=1; break; }
  sleep 0.1
done
if [ "$GUARD_ARMED" = 1 ]; then
  echo "guard armed: MemAvailable floor $((GUARD_FLOOR_KIB / 1024)) MiB, swap growth <= ${GUARD_SWAP_MAX_MIB} MiB, 0.05 s"
else
  echo "WARNING: the memory guard did not arm within 30 s (see $RUN/guard.log). The serve is UNGUARDED;" >&2
  echo "         stop it with: bash scripts/guard.sh stop" >&2
fi
echo "watch: bash scripts/guard.sh status   stop: bash scripts/guard.sh stop"

# ---- endpoint wait, warm-up, ready line -------------------------------------
# The serve is usable only when /v1/models answers, and (WARMUP=1, default)
# the first requests after every boot JIT-compile kernels when FlashInfer
# autotune is off — a cost that must land in a logged warm-up request, not in
# the first real one. Each warm-up request's wall time and MemAvailable
# before/after go to $RUN/warmup.log; a failure fails the start.
if [ "$SERVE_WAIT" != "1" ]; then
  echo "SERVE_WAIT=0: not waiting for /v1/models here (caller owns readiness)."
  echo "  warm-up later with: python3 $HERE/warmup.py --run $RUN --port $API_PORT --model $SERVED_MODEL_NAME --scripts-dir $HERE"
  exit 0
fi
WARMUP_LOG="$RUN/warmup.log"
{
  echo "warmup log: $(date -Is) container=$CONTAINER"
  echo "  FLASHINFER_AUTOTUNE=$FLASHINFER_AUTOTUNE WARMUP=$WARMUP"
} >"$WARMUP_LOG"

endpoint_code() {
  curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null || echo 000
}
echo "endpoint: waiting for /v1/models on port $API_PORT (up to ${WARMUP_WAIT_S} s)"
API_WAITED=0
API_UP=0
while [ "$API_WAITED" -lt "$WARMUP_WAIT_S" ]; do
  if [ "$(endpoint_code)" = "200" ]; then
    API_UP=1
    break
  fi
  # Give up early if the container died (crash, OOM, guard kill): the wait
  # would otherwise run to the full timeout against a dead endpoint.
  if [ "$(sudo -n docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo missing)" != "true" ]; then
    echo "endpoint: $CONTAINER is no longer running; aborting the wait" >&2
    break
  fi
  sleep 5
  API_WAITED=$(( API_WAITED + 5 ))
done
if [ "$API_UP" != 1 ]; then
  echo "endpoint: /v1/models never answered within ${WARMUP_WAIT_S} s (code=$(endpoint_code))" >&2
  echo "  the serve is NOT ready; see $RUN/guard.log and: sudo -n docker logs $CONTAINER" >&2
  echo "  stop with: bash scripts/guard.sh stop" >&2
  exit 1
fi
echo "endpoint: /v1/models answered after ${API_WAITED} s"

if [ "$WARMUP" = "1" ]; then
  echo "warmup: sending the short and ~8K-token requests (max_tokens 32; log: $WARMUP_LOG)"
  if ! python3 "$HERE/warmup.py" \
      --run "$RUN" --port "$API_PORT" --model "$SERVED_MODEL_NAME" \
      --scripts-dir "$HERE"; then
    echo "WARMUP FAILED: the start fails; the HTTP error is in $WARMUP_LOG" >&2
    echo "  stop with: bash scripts/guard.sh stop" >&2
    exit 1
  fi
else
  echo "warmup: skipped (WARMUP=0); the first real request pays the JIT-compile cost" | tee -a "$WARMUP_LOG"
fi

echo "READY: $SERVED_MODEL_NAME at http://127.0.0.1:$API_PORT (run dir $RUN; warmup log $WARMUP_LOG)"
echo "next   : bash scripts/smoke.sh"
