#!/usr/bin/env bash
# Shared pins and measured defaults for serving MiniMax-M3.1-Flash EXL3
# 2.15bpw on ONE DGX Spark. Source this; do not execute it.
#
# Every default here is the configuration the measured boots ran (see README
# "Measured results"). The profile under configs/ selected by PROFILE= can
# tighten it; nothing widens it.
#
# ---- gate arming -----------------------------------------------------------
# RECIPE_ENV_NO_GATES=1 (set ONLY by scripts/guard.sh before sourcing this
# file) skips every fail-closed gate and preflight below while still defining
# every variable, default and export. Reason: a gate calls `exit 1`, and an
# `exit` in a sourced file terminates the SOURCING shell -- a `|| true` on the
# source line cannot catch it -- so a gate firing at guard time (a knob or the
# plugin tree changed under a running serve) would silently kill `guard.sh
# stop` before it could end its own watcher. Under RECIPE_ENV_NO_GATES the
# guarded derivations that can exit are skipped entirely (kvb.py is not run;
# KVB stays empty -- guard.sh never reads it). Every launch/build path
# (serve.sh, serve_inner.sh, build_image.sh, download_and_verify.sh, smoke.sh)
# sources this file WITHOUT the variable, so every gate stays fail-closed for
# every launch.
if [ -n "${RECIPE_ENV_NO_GATES:-}" ]; then
  GATES_ON=0
else
  GATES_ON=1
fi

# ---- model ----------------------------------------------------------------
# Directory name under $MODELS_DIR (must match the Hub repo name: the engine
# resolves the pack's dspark/ draft through the model path).
MODEL_NAME="MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw"
MODEL_REPO="vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw"
# Hub revision, pinned. The measured boots served revision
# e2682d076e220e71e2a7ea7b9885e70390c27698; the Hub repo's history was
# later replaced by a commit that only removed a stray temp file, and the
# current head 99bf3a57af5d07245ea9abb5426e7dce6af6356b carries
# byte-identical weights -- the same git blob oid and size for every one of
# the 40 files -- so this pin tracks it. The bytes, not the commit id, are
# the authority: scripts/SHA256SUMS + scripts/PACK-BLOB-SUMS --
# download_and_verify.sh never falls back to a different tree on its own;
# it refuses unless every file matches the recorded hashes (see the
# README's "Pack revision" section).
MODEL_REV="${MODEL_REV:-99bf3a57af5d07245ea9abb5426e7dce6af6356b}"

# ---- paths ----------------------------------------------------------------
RECIPE_HOME="${RECIPE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/$MODEL_NAME}"
RUNS_DIR="${RUNS_DIR:-$RECIPE_HOME/runs}"
# Runtime-fork checkouts (cloned from the public repos at the pinned refs; see
# the README quick start). The image is built from exactly these trees and
# serve.sh re-checks their revisions against the image labels.
SRC_DIR="${SRC_DIR:-$HOME/minimax-m31-src}"
EXL3_SRC="${EXL3_SRC:-$SRC_DIR/exllamav3}"
PLUGIN_SRC="${PLUGIN_SRC:-$SRC_DIR/vllm-exl3}"
# JIT caches (Triton, CUDA, vLLM) mounted at /cache by serve.sh.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/m31-exl3}"

# ---- images ---------------------------------------------------------------
# Upstream engine image vllm/vllm-openai:minimax-m31-flash (linux/arm64),
# pinned by digest (the digest the measured boots ran on).
BASE_IMAGE_DIGEST="${BASE_IMAGE_DIGEST:-sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c}"
# exllamav3 fork: the GPU-validated tree is branch feat/moe-act3-batched at
# EXL3_REQUIRED_REV. It adds the act-3 (clamped SwiGLU-OAI) case to the
# batched MoE kernels' shared fused-activation helper; without it every
# MiniMax MoE layer outputs garbage (GB10 parity: batched prefill cosine
# ~0.10 without the fix, 1.0 with it). build_image.sh refuses a checkout
# below this rev and labels the image with the full sha; serve.sh re-checks
# the label.
EXL3_REPO="https://github.com/vcruz305/exllamav3.git"
EXL3_REQUIRED_REV="fdd77361971144e8d175e09e7b50b7100739fd95"
EXL3_REV=""
EXL3_SHORT=""
if git -C "$EXL3_SRC" rev-parse --git-dir >/dev/null 2>&1; then
  EXL3_REV="$(git -C "$EXL3_SRC" rev-parse HEAD)"
  EXL3_SHORT="$(git -C "$EXL3_SRC" rev-parse --short HEAD)"
fi
# vllm-exl3 plugin: branch feat/minimax-m3-exl3 at PLUGIN_REQUIRED_REV. That
# tree carries, all fail-closed: the true-bit-width per-expert K arrays of the
# unified mixed-K dispatch (upstream main builds them in half-bit units -> the
# output is wrong for EVERY mixed-K pack served in unified mode), the
# load-time cross-validation and the one-time dispatch-mode log line; the
# aarch64 exllamav3 guard patcher; the indexer lightning-indexer MXFP8
# retention; the q4 (nvfp4-layout) main-KV pool; the UMA load hygiene
# (per-layer trim, embed alias, draft vocab meta, trim-before-KV); and the
# C++20 build fix the plugin setup needs on torch 2.14.
PLUGIN_REPO="https://github.com/vcruz305/vllm-exl3.git"
PLUGIN_REQUIRED_REV="688220ee76d3fa49627d904082550af9458c3cd2"
PLUGIN_REV=""
PLUGIN_SHORT=""
if git -C "$PLUGIN_SRC" rev-parse --git-dir >/dev/null 2>&1; then
  PLUGIN_REV="$(git -C "$PLUGIN_SRC" rev-parse HEAD)"
  PLUGIN_SHORT="$(git -C "$PLUGIN_SRC" rev-parse --short HEAD)"
fi
IMAGE=""
if [ -n "$EXL3_SHORT" ] && [ -n "$PLUGIN_SHORT" ]; then
  IMAGE="${IMAGE:-m31-flash-exl3-spark:dev-$EXL3_SHORT-$PLUGIN_SHORT}"
fi
# Runtime plugin overlay (plugin PYTHON only; the compiled extension always
# comes from the image): PLUGIN_OVERLAY=<host dir of a plugin checkout> is
# mounted read-only at /plugin with PYTHONPATH=/plugin/src, shadowing the
# plugin baked into the image. Empty (default) = use the baked-in plugin,
# which serve.sh allows ONLY when it is at or after PLUGIN_REQUIRED_REV; an
# image that bakes an older plugin (the GB10-measured image dev-fdd7736-
# a5e91ed) is accepted ONLY with an overlay at or after PLUGIN_REQUIRED_REV.
PLUGIN_OVERLAY="${PLUGIN_OVERLAY:-}"

# ---- profile --------------------------------------------------------------
# PROFILE selects configs/<name>.env. Every profile pins exactly the flag set
# of a measured boot (see README "Profiles"); the default is the largest
# context that passed a needle test within the guard floor.
PROFILE="${PROFILE:-q4-32k}"
PROFILE_FILE="$RECIPE_HOME/configs/$PROFILE.env"
if [ "$GATES_ON" = "1" ] && [ ! -f "$PROFILE_FILE" ]; then
  echo "PROFILE=$PROFILE: no such profile at $PROFILE_FILE" >&2
  echo "available: $(cd "$RECIPE_HOME/configs" && for f in *.env; do printf '%s ' "${f%.env}"; done)" >&2
  exit 1
fi
# The profile may only tighten: it is sourced before the serve defaults below
# so every default it leaves untouched stays the measured value, and the
# fail-closed gates at the end of this file re-validate the combination.
if [ -f "$PROFILE_FILE" ]; then
  # shellcheck source=/dev/null
  . "$PROFILE_FILE"
fi

# ---- serve ----------------------------------------------------------------
# MAXLEN's authority is the profile (every profile pins a measured context
# length; GATES_ON=1 refuses a missing profile). This fallback only keeps
# the arithmetic below alive in the RECIPE_ENV_NO_GATES=1 path (guard.sh,
# whose sandbox has no configs/): it equals the default profile's value, so
# it can never widen anything.
MAXLEN="${MAXLEN:-32768}"
CONTAINER="${CONTAINER:-m31-flash-tp1}"
API_PORT="${API_PORT:-8000}"
# OpenAI model id the server answers to.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMax-M3.1-Flash-2.15bpw}"
# One decode stream only: the MSA indexer tops out at one running sequence and
# the memory budget has no room for more.
MAX_NUM_SEQS=1
# MSA sparse-attention block size. Mandatory upstream for this architecture.
BLOCK_SIZE=128
# KV precision knobs. On GB10 (sm_121) nvfp4 is NOT available upstream for
# either pool (the engine raises on non-SM100); the options are fp8 or bf16.
# The indexer key cache stays fp8 in every profile.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-fp8}"
# ---- q4 (nvfp4-layout) main KV (plugin VLLM_EXL3_Q4KV) ---------------------
# 0 (default) = fp8 main KV. 1 = q4 main KV: the container gets
# VLLM_EXL3_Q4KV=1 (the plugin reroutes the main-KV backend selector so the
# main pool takes the upstream nvfp4 spec arithmetic at 576 B/token/layer),
# the main --kv-cache-dtype becomes nvfp4, and the KVB default switches from
# the legacy fp8 formula to scripts/kvb.py's q4 budget (46.25 KiB/token +2%
# margin). FAIL-CLOSED: q4 requires the indexer AND the DSpark draft to stay
# fp8 -- the plugin refuses an nvfp4 indexer, a bf16 draft would blow the
# budget, and fp8 draft KV is Triton-attention-only on sm_121 -- so KV_Q4=1
# is refused unless INDEXER_KV_DTYPE=fp8, DSPARK=1, DSPARK_KV_DTYPE=fp8 and
# DSPARK_ATTN_BACKEND=TRITON_ATTN. KV_Q4=0 is refused with
# KV_CACHE_DTYPE=nvfp4 (nvfp4 without the overlay is a guaranteed upstream
# raise).
KV_Q4="${KV_Q4:-0}"

# ---- KV pool sizing (explicit bytes; NOT --gpu-memory-utilization) ---------
# The profile sets MAXLEN (every profile pins a measured context length);
# an empty one is a broken profile, not a fallback case.
if [ "$GATES_ON" = "1" ] && [ -z "${MAXLEN:-}" ]; then
  echo "MAXLEN is empty: the profile $PROFILE did not set a context length." >&2
  exit 1
fi
# On GB10 unified memory a fraction-of-device knob is the wrong control: with
# --kv-cache-memory-bytes set, the engine sizes the pool from the byte budget
# and skips memory profiling. Measured combined pool cost (main + indexer +
# draft): 72.65 KiB/token fp8 (scripts/kvb.py; the naive component sum is
# 72.50, the measured cost is +0.21% per-layer rounding), 46.25 KiB/token q4.
# KVB resolution: a KVB set by the profile or the environment is an explicit
# override that always wins (recorded in KVB_SRC so DRY_RUN prints can never
# mislabel it); KVB unset on the q4 path is derived by scripts/kvb.py; KVB
# unset on the fp8 path falls back to the legacy formula below (90 KiB/token x
# 1.10, aligned up to 256 MiB) -- deliberately conservative, never a
# recommendation.
KV_POOL_PER_TOKEN_B=92160     # 90 KiB legacy formula constant
KV_POOL_HEADROOM_PCT=110     # 1.10 headroom (percent)
KV_POOL_ALIGN=268435456      # 256 MiB alignment
KVB_RAW=$(( (MAXLEN * KV_POOL_PER_TOKEN_B * KV_POOL_HEADROOM_PCT + 99) / 100 ))
KVB_CALC=$(( (KVB_RAW + KV_POOL_ALIGN - 1) / KV_POOL_ALIGN * KV_POOL_ALIGN ))
KVB_WANT_AUTO=0
if [ "${KVB:-}" = "auto" ]; then KVB_WANT_AUTO=1; fi
if [ -n "${KVB:-}" ] && [ "$KVB_WANT_AUTO" = "0" ]; then
  KVB_SRC=override
elif [ "$KVB_WANT_AUTO" = "1" ] || [ "$KV_Q4" = "1" ]; then
  if [ "$GATES_ON" = "0" ]; then
    # RECIPE_ENV_NO_GATES (guard.sh): never run the kvb.py derivation at
    # guard time -- its failure path calls `exit`, which would kill the
    # sourcing shell. guard.sh does not read KVB, so it stays empty.
    KVB=""
    KVB_SRC="skipped-no-gates"
  else
    KVB="$(python3 "$RECIPE_HOME/scripts/kvb.py" --maxlen "$MAXLEN" --q4 "$KV_Q4" --kvb-only)" || {
      echo "scripts/kvb.py failed: cannot derive the KV pool budget" >&2
      exit 1
    }
    if [ "$KVB_WANT_AUTO" = "1" ]; then
      KVB_SRC=auto-kvb-py
    else
      KVB_SRC=q4-kvb-py
    fi
  fi
else
  KVB="$KVB_CALC"
  KVB_SRC=formula
fi

# ---- all-resident load (no expert offload) ---------------------------------
# Every routed expert stays device-resident; the only aliasing is the pack's
# own BF16 embedding, read back from its shard through a read-only mapping
# (row gathers only) instead of a CUDA copy. Unified-memory load hygiene
# (plugin src/vllm_exl3/uma_load_memory.py), all measured:
#   LOAD_TRIM=1        VLLM_EXL3_TRIM_AFTER_LAYER=1: gc + malloc_trim +
#                      empty_cache before every MoE layer's arena allocation
#                      (loader temporaries are real residency on GB10)
#   EMBED_ALIAS=1      VLLM_EXL3_UMA_POSTLOAD=1 + the ATS embedding alias
#                      (the 2.29 GiB BF16 embedding never gets a CUDA copy)
#   DRAFT_VOCAB_META=1 VLLM_EXL3_DRAFT_VOCAB_META=1: the DSpark draft ships
#                      no embed/lm_head (shared from the target), so its own
#                      placeholders go on the meta device
#   TRIM_BEFORE_KV=1   VLLM_EXL3_TRIM_BEFORE_KV=1: hand the profile run's
#                      cached activations and compile scratch back before the
#                      KV pool is allocated and before warmup
#   MOE_TEMP_ROWS      VLLM_EXL3_MOE_TEMP_ROWS: fused-MoE temp capacity
#                      (rows per expert); 512 >= the fat-expert threshold (256)
#                      is exact and saves 324 MiB vs the plugin default 2048
#   PERIODIC_TRIM_S    VLLM_EXL3_PERIODIC_TRIM_S: glibc malloc_trim period
#                      while serving
#   JIT_WARMUP_THREADS VLLM_TRITON_JIT_WARMUP_NUM_THREADS (vLLM default 4)
#   COMPILE_THREADS    TORCHINDUCTOR_COMPILE_THREADS: one in-process compile
#                      thread; no subprocess worker pool (each worker is a
#                      full torch import in host memory)
LOAD_TRIM="${LOAD_TRIM:-1}"
EMBED_ALIAS="${EMBED_ALIAS:-1}"
DRAFT_VOCAB_META="${DRAFT_VOCAB_META:-1}"
TRIM_BEFORE_KV="${TRIM_BEFORE_KV:-1}"
MOE_TEMP_ROWS="${MOE_TEMP_ROWS:-512}"
PERIODIC_TRIM_S="${PERIODIC_TRIM_S:-30}"
JIT_WARMUP_THREADS="${JIT_WARMUP_THREADS:-1}"
COMPILE_THREADS="${COMPILE_THREADS:-1}"
# MALLOC_ARENA_MAX: glibc arena cap inside the container (measured 2; the
# default arena fan-out grows host anonymous memory during torch.compile).
MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# CUDA graphs / compile mode (measured set: decode runs as FULL graphs at
# sizes 1-4 -- one sequence with num_spec=3 decodes 4 tokens per step -- and
# the modules stay eager; the larger compile modes cut neither rss nor step
# time and cost GiB of host memory).
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4}"
COMPILE_MODE="${COMPILE_MODE:-0}"
# Chunked-prefill chunk size. Measured 1024; must stay a multiple of
# BLOCK_SIZE, and larger values do not fit the memory budget.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
# FLASHINFER_AUTOTUNE: 0 (measured default) passes
# --kernel-config '{"enable_flashinfer_autotune":false}' and skips the boot
# FlashInfer autotune; with autotune off the first requests after every boot
# JIT-compile kernels instead, which is exactly what the WARMUP step pays
# and logs. (The autotuned boot measured LESS free memory at ready and the
# same ms/step; see README.)
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-0}"
# WARMUP: 1 (default) = after /v1/models answers, serve.sh sends one short
# chat request and one ~8K-token prompt (the recipe's own scripts
# concatenated plus a random nonce, max_tokens 32) and only then prints the
# READY line; each request's wall time and MemAvailable before/after go to
# <run dir>/warmup.log, and a warm-up failure fails the start.
WARMUP="${WARMUP:-1}"
WARMUP_WAIT_S="${WARMUP_WAIT_S:-3600}"
# SERVE_WAIT: 1 (default) = serve.sh blocks until /v1/models answers, runs
# the WARMUP step and prints READY. This recipe is standalone: there is no
# external supervisor owning readiness.
SERVE_WAIT="${SERVE_WAIT:-1}"

# ---- DSpark speculative decoding -----------------------------------------
# The draft ships inside the pack (dspark/ subfolder): the engine resolves it
# from the model path; NO draft model path is passed anywhere. GB10-measured
# draft pair: TRITON_ATTN + fp8 draft KV (draft attention_backend "auto"
# FAILS on GB10 -- it picks FLASHINFER, which has no block size 128; FA fp8
# draft KV is FA3/Hopper-only). k=3 (num_speculative_tokens): one verify step
# covers 4 tokens, which is what the CUDA-graph capture sizes 1-4 above pin.
DSPARK="${DSPARK:-1}"
DSPARK_NUM_SPEC="${DSPARK_NUM_SPEC:-3}"
DSPARK_ATTN_BACKEND="${DSPARK_ATTN_BACKEND:-TRITON_ATTN}"
DSPARK_KV_DTYPE="${DSPARK_KV_DTYPE:-fp8}"
DSPARK_DRAFT_SAMPLE="${DSPARK_DRAFT_SAMPLE:-probabilistic}"
DSPARK_REJECTION="${DSPARK_REJECTION:-block}"

# ---- q4 fail-closed gate (runs after every knob it checks is defined) -------
q4_refuse() {
  echo "KV_Q4 gate: refusing to serve. $*" >&2
  echo "KV_Q4=1 needs INDEXER_KV_DTYPE=fp8, DSPARK=1, DSPARK_KV_DTYPE=fp8 and" >&2
  echo "DSPARK_ATTN_BACKEND=TRITON_ATTN; the indexer and the draft stay fp8." >&2
  exit 1
}
if [ "$GATES_ON" = "1" ]; then
if [ "$KV_Q4" = "1" ]; then
  case "$INDEXER_KV_DTYPE" in
    fp8) ;;
    *) q4_refuse "INDEXER_KV_DTYPE=$INDEXER_KV_DTYPE: an nvfp4 indexer is refused by the plugin and any other value breaks the 46.25 KiB/token budget." ;;
  esac
  [ "$DSPARK" = "1" ] || q4_refuse "DSPARK=$DSPARK: the q4 budget is costed with the draft's fp8 pool."
  [ "$DSPARK_KV_DTYPE" = "fp8" ] || q4_refuse "DSPARK_KV_DTYPE=$DSPARK_KV_DTYPE: 'auto' resolves to a bf16 draft pool and None would inherit the main nvfp4 dtype; the draft must be fp8."
  [ "$DSPARK_ATTN_BACKEND" = "TRITON_ATTN" ] || q4_refuse "DSPARK_ATTN_BACKEND=$DSPARK_ATTN_BACKEND: fp8 draft KV is served by TRITON_ATTN only on sm_121."
  case "$KV_CACHE_DTYPE" in
    fp8|nvfp4) ;;
    *) q4_refuse "KV_CACHE_DTYPE=$KV_CACHE_DTYPE conflicts with KV_Q4=1 (the main pool is nvfp4 in q4 mode)." ;;
  esac
  KV_CACHE_DTYPE=nvfp4   # the q4 main pool: --kv-cache-dtype nvfp4 + VLLM_EXL3_Q4KV=1
  echo "KV_Q4=1: main KV -> nvfp4 (q4 overlay pool, 576 B/token/layer); indexer stays $INDEXER_KV_DTYPE; draft stays $DSPARK_KV_DTYPE." >&2
else
  if [ "$KV_Q4" != "0" ]; then
    echo "KV_Q4 must be 0 or 1, got '$KV_Q4'" >&2
    exit 1
  fi
  if [ "$KV_CACHE_DTYPE" = "nvfp4" ]; then
    echo "KV_Q4 gate: refusing to serve. KV_CACHE_DTYPE=nvfp4 without" >&2
    echo "KV_Q4=1 is a guaranteed upstream raise (no overlay installed). Use" >&2
    echo "KV_Q4=1 or KV_CACHE_DTYPE=fp8." >&2
    exit 1
  fi
fi
fi

# ---- indexer lightning-indexer MXFP8 (plugin VLLM_EXL3_INDEXER_MXFP8) -----
# 1 (default) = the container gets VLLM_EXL3_INDEXER_MXFP8=1: the plugin's
# lightning-indexer keeps its index_q/index_k tensors MXFP8 on device and
# runs the indexer GEMM through vLLM's MarlinMxfp8LinearKernel (W8A16)
# instead of dequantizing to bf16 first. GB10-measured (same 16K fp8 boot
# with and without): identical quality and speed, and +0.4-0.5 GiB free
# memory at every stage (steady 1.07-1.17 vs 0.57-0.67 GiB) -- see README.
# 0 = the variable is not set at all: the plugin dequantizes index_q/index_k.
INDEXER_MXFP8="${INDEXER_MXFP8:-1}"
if [ "$GATES_ON" = "1" ]; then
if [ "$INDEXER_MXFP8" != "0" ] && [ "$INDEXER_MXFP8" != "1" ]; then
  echo "INDEXER_MXFP8 must be 0 or 1, got '$INDEXER_MXFP8'" >&2
  exit 1
fi
fi
# Fail-closed preflight (runs only at INDEXER_MXFP8=1): the runtime plugin
# code -- the PLUGIN_OVERLAY when one is mounted, else the plugin source of
# record PLUGIN_SRC -- must actually contain the indexer MXFP8 code at or
# after the pinned plugin rev. Fail closed when the absence is determinable;
# warn and continue only when neither the git nor the grep check can decide.
if [ "$GATES_ON" = "1" ]; then
if [ "$INDEXER_MXFP8" = "1" ]; then
  IM_DIR="$PLUGIN_OVERLAY"
  [ -n "$IM_DIR" ] || IM_DIR="$PLUGIN_SRC"
  IM_HAS_CODE=0
  IM_UNDECIDABLE=0
  if git -C "$IM_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    if git -C "$IM_DIR" merge-base --is-ancestor "$PLUGIN_REQUIRED_REV" HEAD 2>/dev/null; then
      IM_HAS_CODE=1
    elif grep -q "indexer_mxfp8_move_to_device" "$IM_DIR/src/vllm_exl3/exl3.py" 2>/dev/null; then
      IM_HAS_CODE=1
    fi
  elif [ -e "$IM_DIR/src/vllm_exl3/exl3.py" ]; then
    if grep -q "indexer_mxfp8_move_to_device" "$IM_DIR/src/vllm_exl3/exl3.py" 2>/dev/null; then
      IM_HAS_CODE=1
    fi
  else
    IM_UNDECIDABLE=1
  fi
  if [ "$IM_UNDECIDABLE" = "1" ]; then
    echo "WARNING: INDEXER_MXFP8=1 but $IM_DIR is neither a git checkout nor a" >&2
    echo "readable plugin tree: the indexer MXFP8 code presence cannot be" >&2
    echo "determined. Continuing (best-effort); if the plugin lacks the code," >&2
    echo "VLLM_EXL3_INDEXER_MXFP8 is a silent no-op." >&2
  elif [ "$IM_HAS_CODE" != "1" ]; then
    echo "INDEXER_MXFP8 gate: refusing to serve. $IM_DIR carries neither rev" >&2
    echo "$PLUGIN_REQUIRED_REV nor the function indexer_mxfp8_move_to_device in" >&2
    echo "src/vllm_exl3/exl3.py: the plugin would silently ignore" >&2
    echo "VLLM_EXL3_INDEXER_MXFP8=1 and the boot would lose the measured" >&2
    echo "0.4-0.5 GiB under load. Point PLUGIN_OVERLAY at a checkout at" >&2
    echo "$PLUGIN_REQUIRED_REV or a descendant, or set INDEXER_MXFP8=0." >&2
    exit 1
  fi
fi
fi

# ---- request defaults -----------------------------------------------------
# Output cap for requests that send no max_tokens (agent clients send none).
# Without it a reply that loops runs until the context window is full.
# A client value still wins. 0 = no server default.
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-32768}"
# Default sampling for requests that send no temperature/top_p. Both stay
# EMPTY: the pack's generation_config.json (temperature 1.0 / top_p 0.95)
# governs unmodified -- the measured bench ran exactly at those values and
# they are the model's own recommendation. Empty = no override.
DEFAULT_TEMPERATURE="${DEFAULT_TEMPERATURE:-}"
DEFAULT_TOP_P="${DEFAULT_TOP_P:-}"

# ---- memory guard ---------------------------------------------------------
# Floors the memory guard kills the container at: MemAvailable below
# GUARD_MEM_FLOOR_KIB or swap growth above GUARD_SWAP_MAX_MIB, sampled every
# 0.05 s. Default 0.5 GiB. A lower floor is an explicit opt-in by the
# operator (the measured 24K runs used 209716 KiB = 0.2 GiB, see README);
# the guard exists to keep the HOST alive, not to maximize context.
GUARD_MEM_FLOOR_KIB="${GUARD_MEM_FLOOR_KIB:-524288}"
GUARD_SWAP_MAX_MIB="${GUARD_SWAP_MAX_MIB:-64}"

export MODEL_NAME MODEL_REPO MODEL_REV RECIPE_HOME MODELS_DIR MODEL_DIR RUNS_DIR SRC_DIR CACHE_DIR
export BASE_IMAGE_DIGEST IMAGE EXL3_REPO EXL3_REQUIRED_REV EXL3_SRC EXL3_REV EXL3_SHORT
export PLUGIN_REPO PLUGIN_REQUIRED_REV PLUGIN_SRC PLUGIN_REV PLUGIN_SHORT PLUGIN_OVERLAY
export CONTAINER API_PORT SERVED_MODEL_NAME MAXLEN MAX_NUM_SEQS BLOCK_SIZE
export KV_CACHE_DTYPE INDEXER_KV_DTYPE KV_Q4 INDEXER_MXFP8 MAX_NUM_BATCHED_TOKENS
export KV_POOL_PER_TOKEN_B KV_POOL_HEADROOM_PCT KV_POOL_ALIGN KVB KVB_SRC KVB_RAW
export LOAD_TRIM EMBED_ALIAS DRAFT_VOCAB_META TRIM_BEFORE_KV MOE_TEMP_ROWS PERIODIC_TRIM_S
export MALLOC_ARENA_MAX CG_MODE CAPTURE_SIZES COMPILE_MODE JIT_WARMUP_THREADS COMPILE_THREADS
export FLASHINFER_AUTOTUNE WARMUP WARMUP_WAIT_S SERVE_WAIT
export DSPARK DSPARK_NUM_SPEC DSPARK_DRAFT_SAMPLE DSPARK_REJECTION
export DSPARK_ATTN_BACKEND DSPARK_KV_DTYPE
export DEFAULT_MAX_TOKENS DEFAULT_TEMPERATURE DEFAULT_TOP_P
export GUARD_MEM_FLOOR_KIB GUARD_SWAP_MAX_MIB PROFILE
