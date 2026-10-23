#!/bin/bash
# In-container serve script for MiniMax-M3.1-Flash EXL3 2.15bpw, TP1 on GB10.
#
# Every flag below is the GB10-measured configuration (see the README's
# "Measured results"); the container env (set by scripts/serve.sh from
# scripts/env.sh + configs/<profile>.env) is authoritative, and the defaults
# here only make a hand-run container fail loudly instead of silently when an
# env var is missing.
set -euo pipefail

MAXLEN="${MAXLEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-fp8}"
# q4 main KV: the host gate (env.sh) exports VLLM_EXL3_Q4KV=1 into the
# container ONLY in q4 mode (KV_Q4=1); the variable is absent in fp8 mode,
# so this default is 0 there. Defense in depth: re-check every q4 invariant
# in-container.
KV_Q4="${VLLM_EXL3_Q4KV:-0}"
# Lightning-indexer MXFP8 (INDEXER_MXFP8, env.sh): the host preflight has
# already refused a 1 whose runtime plugin tree lacks the indexer-MXFP8
# code; the plugin reads the variable itself, and the banner below records
# whether the kept-MXFP8 path is armed so the boot log can be correlated
# with the measured free-memory numbers.
INDEXER_MXFP8="${VLLM_EXL3_INDEXER_MXFP8:-0}"
# Explicit KV pool size in bytes, computed in env.sh (kvb.py budget or the
# legacy conservative formula). On GB10 unified memory
# --gpu-memory-utilization is the wrong knob: with
# --kv-cache-memory-bytes set, the engine sizes the pool from it and skips
# memory profiling.
KVB="${KVB:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
# Mixed-K MoE dispatch: VLLM_EXL3_MIXEDK_MODE is deliberately NOT set here (or
# anywhere in this recipe): unset = the plugin's default unified
# exl3_moe_mixedk dispatch, the GB10-validated mode (parity run 6: cosine 1.0
# on the pack's real mixed-K experts). The plugin logs the selected mode once
# at first use and scripts/smoke.sh fails the boot unless the serve log says
# 'EXL3 mixed-K MoE dispatch mode: unified'; grouped is a documented fallback
# env (VLLM_EXL3_MIXEDK_MODE=grouped) at a measured 3x8-CTA-launch decode
# cost, never a serve default.
DSPARK="${DSPARK:-1}"
DSPARK_NUM_SPEC="${DSPARK_NUM_SPEC:-3}"
# Draft backend — GB10-MEASURED: "auto" FAILS (it picks FLASHINFER, which has
# no block size 128); FA fp8 draft KV is FA3/Hopper-only, NOT supported on
# sm_121. The only pair that boots and passes on GB10 is TRITON_ATTN + fp8
# draft KV, which every profile pins.
DSPARK_ATTN_BACKEND="${DSPARK_ATTN_BACKEND:-TRITON_ATTN}"
DSPARK_KV_DTYPE="${DSPARK_KV_DTYPE:-fp8}"
DSPARK_DRAFT_SAMPLE="${DSPARK_DRAFT_SAMPLE:-probabilistic}"
DSPARK_REJECTION="${DSPARK_REJECTION:-block}"
# FlashInfer autotune (FLASHINFER_AUTOTUNE, env.sh): 0 (measured default)
# passes enable_flashinfer_autotune:false via --kernel-config, skipping the
# boot tuning and its extra memory (the autotuned boot measured LESS free
# memory at ready and the same ms/step); the first requests after every boot
# then JIT-compile kernels instead — exactly the cost serve.sh's warm-up step
# pays and logs.
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-0}"
# Default output cap for requests without max_tokens (0 = none); see
# DEFAULT_MAX_TOKENS in env.sh. A client value still wins per request.
# Sampling defaults are intentionally NOT overridden: the pack's
# generation_config.json (temperature 1.0 / top_p 0.95) governs unmodified.
gen_json=""
GENCFG=()
if [ "${DEFAULT_MAX_TOKENS:-32768}" != "0" ]; then
  gen_json="\"max_new_tokens\": ${DEFAULT_MAX_TOKENS:-32768}"
  GENCFG=(--override-generation-config "{$gen_json}")
fi

if [ "$DSPARK" = "1" ]; then
  # No draft model path: the engine resolves the pack's dspark/ subfolder by
  # itself (upstream vllm/config/speculative.py, _MINIMAX_M3_DSPARK_SUBFOLDER).
  # attention_backend/kv_cache_dtype are the GB10-verified pair (see the
  # comment above): TRITON_ATTN + fp8. num_spec=3: one verify step covers 4
  # tokens, which is what the CUDA-graph capture sizes 1-4 pin.
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":'"$DSPARK_NUM_SPEC"',"attention_backend":"'"$DSPARK_ATTN_BACKEND"'","kv_cache_dtype":"'"$DSPARK_KV_DTYPE"'","draft_sample_method":"'"$DSPARK_DRAFT_SAMPLE"'","rejection_sample_method":"'"$DSPARK_REJECTION"'"}'
  SPEC_ARGS=(--speculative-config "$SPEC_CONFIG")
else
  SPEC_ARGS=()
  echo "DSPARK=0: serving WITHOUT the draft (slow sanity path)"
fi

# In-container q4 gate (defense in depth; the host gate is env.sh's). The
# indexer and the DSpark draft must stay fp8 and the main pool must be the
# overlay's nvfp4 container; anything else refuses before exec vllm.
if [ "$KV_Q4" = "1" ]; then
  if [ "$KV_CACHE_DTYPE" != "nvfp4" ] || [ "$INDEXER_KV_DTYPE" != "fp8" ] \
     || [ "$DSPARK" != "1" ] || [ "$DSPARK_KV_DTYPE" != "fp8" ] \
     || [ "$DSPARK_ATTN_BACKEND" != "TRITON_ATTN" ]; then
    echo "q4 gate: VLLM_EXL3_Q4KV=1 needs --kv-cache-dtype nvfp4," >&2
    echo "indexer fp8, DSPARK=1 with fp8/TRITON_ATTN draft; refusing." >&2
    exit 1
  fi
fi

echo "m31 serve: model=/models/${VLLM_EXL3_MODEL_DIR##*/} maxlen=$MAXLEN seqs=$MAX_NUM_SEQS"
echo "  block_size=$BLOCK_SIZE kv=$KV_CACHE_DTYPE indexer=$INDEXER_KV_DTYPE"
echo "  kv_pool_bytes=$KVB mnbt=$MAX_NUM_BATCHED_TOKENS dspark=$DSPARK draft_attn=$DSPARK_ATTN_BACKEND draft_kv=$DSPARK_KV_DTYPE"
echo "  flashinfer_autotune=$FLASHINFER_AUTOTUNE"
echo "  indexer_mxfp8=$INDEXER_MXFP8"
[ -n "$KVB" ] || { echo "KVB is empty: env.sh did not run (source scripts/env.sh first)" >&2; exit 1; }

# CUDA graphs: CG_MODE (FULL_DECODE_ONLY) + CAPTURE_SIZES (token counts,
# comma list) pin the graph set; unset keeps vLLM's defaults. One sequence
# with DSPARK_NUM_SPEC=3 decodes 4 tokens per step, so 1..4 covers decode.
CGCFG=()
CC=()
if [ -n "${CG_MODE:-}" ]; then
  CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4}"
  CC+=("\"cudagraph_mode\":\"$CG_MODE\"" "\"cudagraph_capture_sizes\":[$CAPTURE_SIZES]")
  CGCFG+=(--max-cudagraph-capture-size "${CAPTURE_SIZES##*,}")
  echo "  cudagraph_mode=$CG_MODE capture_sizes=[$CAPTURE_SIZES]"
fi
# COMPILE_MODE: torch.compile mode (0/NONE = eager modules; full CUDA graphs
# still captured by the runner — the measured set: the larger compile modes
# cut neither rss nor step time and cost GiB of host memory).
if [ -n "${COMPILE_MODE:-}" ]; then
  case "$COMPILE_MODE" in
    [0-9]) CC+=("\"mode\":$COMPILE_MODE") ;;
    *) CC+=("\"mode\":\"$COMPILE_MODE\"") ;;
  esac
  echo "  compile_mode=$COMPILE_MODE"
fi
if [ "${#CC[@]}" -gt 0 ]; then
  CCJSON="{$(IFS=,; echo "${CC[*]}")}"
  CGCFG=(--compilation-config "$CCJSON" "${CGCFG[@]}")
fi

# FlashInfer autotune: see the FLASHINFER_AUTOTUNE comment above. Any future
# --kernel-config key MUST be appended to KC below, never passed as a second
# --kernel-config flag: vLLM takes one --kernel-config and a second flag
# would silently drop the first.
KC=()
if [ "$FLASHINFER_AUTOTUNE" = "0" ]; then
  KC+=('"enable_flashinfer_autotune":false')
  echo "  flashinfer_autotune=off (--kernel-config; first requests will JIT-compile instead)"
fi
if [ "${#KC[@]}" -gt 0 ]; then
  KCJSON="{$(IFS=,; echo "${KC[*]}")}"
  KCFG=(--kernel-config "$KCJSON")
else
  KCFG=()
fi

exec vllm serve "${VLLM_EXL3_MODEL_DIR:-/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw}" \
    --served-model-name "${SERVED_MODEL_NAME:-MiniMax-M3.1-Flash-2.15bpw}" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --language-model-only \
    --block-size "$BLOCK_SIZE" \
    --disable-hybrid-kv-cache-manager \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --kv-cache-memory-bytes "$KVB" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --attention-config '{"indexer_kv_dtype":"'"$INDEXER_KV_DTYPE"'"}' \
    "${SPEC_ARGS[@]}" \
    "${CGCFG[@]}" \
    "${KCFG[@]}" \
    --reasoning-parser minimax_m3 \
    --tool-call-parser minimax_m3 \
    --enable-auto-tool-choice \
    "${GENCFG[@]}"
