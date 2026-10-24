#!/usr/bin/env python3
"""KVB calculator for the MiniMax-M3.1-Flash EXL3 Spark recipe.

Given MAXLEN and the KV_Q4 mode, prints the per-token pool cost, the raw
pool requirement, the recommended --kv-cache-memory-bytes budget, and the
token capacity a given budget buys. Pure arithmetic, no vllm import, no
GPU: the constants are the engine lane's measurements/expectations, and
every number this prints must be re-measured on the first real boot.

Per-token pool costs (all three pools combined, per token):

  KV_Q4=0 (fp8, MEASURED on GB10, boot 13): 72.65 KiB/token
      = main MSA fp8 KV 60 KiB + indexer fp8 e4m3 7.5 KiB
        + DSpark draft fp8 5 KiB. The naive component sum is 72.50; the
        measured cost is +0.21% (per-layer/per-page rounding), so the
        margin below covers the observed rounding with ~10x headroom.
  KV_Q4=1 (q4 main, EXPECTED, not yet measured): 46.25 KiB/token
      = main MSA q4 (nvfp4-layout) 33.75 KiB (576 B/token/layer) +
        indexer fp8 e4m3 7.5 KiB + DSpark draft fp8 5 KiB. The indexer
        and the draft MUST stay fp8 in this mode: the plugin overlay
        (VLLM_EXL3_Q4KV=1) refuses an nvfp4 indexer fail-closed, and a
        bf16 draft (DSPARK_KV_DTYPE=auto) would quadruple the draft leg
        of this sum.

Safety margin: +2% over the raw requirement, then ceil (no 256 MiB
alignment: that alignment is the legacy fp8 formula's policy; a measured
budget uses an explicit minimal byte count). Anchors: the 16K fp8
production boot ran with KVB=1238000000 against a measured raw cost of
1,218,864,742 B, i.e. an effective +1.57% margin; the 48K q4 production
boot (configs/q4-48k.env) ran with KVB=2374395495 against a raw cost of
2,327,838,720 B and the engine allocated 50,048 tokens of KV (1.02x the
context limit), i.e. the +2% margin held. The +2% here covers the +0.21%
per-layer rounding overhead measured on fp8 with headroom; re-measure
before moving either constant.

The legacy fp8 sizing formula (90 KiB/token x 1.10, aligned up to
256 MiB) predates the 72.65 KiB/token measurement and is kept ONLY as
env.sh's KV_Q4=0 default (KVB_LEGACY_FORMULA) so the fp8 profile's
bytes never move; it is not a recommendation, and KVB=auto never uses
it.

Usage:
  python3 scripts/kvb.py --maxlen 49152 --q4 1               # q4-48k (default)
  python3 scripts/kvb.py --maxlen 32768 --q4 1               # q4-32k
  python3 scripts/kvb.py --maxlen 49152 --q4 1 --kvb 2374395495
  python3 scripts/kvb.py --maxlen 49152 --q4 1 --kvb-only    # env.sh hook
"""
from __future__ import annotations

import argparse
import math
import sys

KIB = 1024

# MEASURED combined fp8 pool cost (main + indexer + draft), KiB/token.
FP8_COST_KIB_PER_TOKEN = 72.65
# EXPECTED combined q4 pool cost (q4 main + fp8 indexer + fp8 draft),
# KiB/token. 33.75 = 576 B/token/layer for the main MSA pool.
Q4_COST_KIB_PER_TOKEN = 46.25

# Documented safety margin over the raw requirement (see module docstring).
SAFETY_MARGIN_PCT = 2

# Legacy fp8 formula constants (env.sh's KV_Q4=0 default; see module
# docstring). Kept here so tests can prove env.sh and this file agree.
LEGACY_FP8_COST_B_PER_TOKEN = 92160          # 90 KiB
LEGACY_HEADROOM_PCT = 110                   # x 1.10
LEGACY_ALIGN_BYTES = 268435456              # 256 MiB


def _cost_b_per_token(kv_q4: int) -> float:
    """Combined per-token pool cost in bytes for the given mode."""
    if isinstance(kv_q4, bool) or not isinstance(kv_q4, int):
        raise ValueError(f"kv_q4 must be 0 or 1, got {kv_q4!r}")
    if kv_q4 == 1:
        return Q4_COST_KIB_PER_TOKEN * KIB
    if kv_q4 == 0:
        return FP8_COST_KIB_PER_TOKEN * KIB
    raise ValueError(f"kv_q4 must be 0 or 1, got {kv_q4!r}")


def _check_maxlen(maxlen: int) -> int:
    if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen <= 0:
        raise ValueError(f"maxlen must be a positive integer, got {maxlen!r}")
    return maxlen


def raw_pool_bytes(maxlen: int, kv_q4: int) -> float:
    """Raw pool requirement for maxlen tokens (no margin), in bytes."""
    return _check_maxlen(maxlen) * _cost_b_per_token(kv_q4)


def recommended_kvb(maxlen: int, kv_q4: int) -> int:
    """--kv-cache-memory-bytes budget: raw requirement x (1 + margin), ceil."""
    return math.ceil(raw_pool_bytes(maxlen, kv_q4) * (100 + SAFETY_MARGIN_PCT) / 100)


def expected_tokens(kvb: int, kv_q4: int) -> int:
    """Token capacity a given byte budget buys at the raw per-token cost."""
    if not isinstance(kvb, int) or isinstance(kvb, bool) or kvb < 0:
        raise ValueError(f"kvb must be a non-negative integer, got {kvb!r}")
    return math.floor(kvb / _cost_b_per_token(kv_q4))


def legacy_formula_kvb(maxlen: int) -> int:
    """env.sh's legacy fp8 formula value (90 KiB/token x 1.10, 256 MiB
    aligned). Byte-compat reference only; never a recommendation."""
    _check_maxlen(maxlen)
    raw = (maxlen * LEGACY_FP8_COST_B_PER_TOKEN * LEGACY_HEADROOM_PCT + 99) // 100
    return (raw + LEGACY_ALIGN_BYTES - 1) // LEGACY_ALIGN_BYTES * LEGACY_ALIGN_BYTES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="MiniMax-M3.1-Flash Spark recipe KVB calculator")
    p.add_argument("--maxlen", type=int, required=True,
                   help="max-model-len the pool must cover (tokens)")
    p.add_argument("--q4", type=int, choices=(0, 1), required=True,
                   help="0 = fp8 pool (measured cost), 1 = q4 main KV (expected cost)")
    p.add_argument("--kvb", type=int, default=None,
                   help="byte budget to invert into a token capacity")
    p.add_argument("--kvb-only", action="store_true",
                   help="print only the recommended KVB integer (env.sh hook)")
    args = p.parse_args(argv)

    cost = _cost_b_per_token(args.q4)
    raw = raw_pool_bytes(args.maxlen, args.q4)
    kvb = recommended_kvb(args.maxlen, args.q4)
    if args.kvb_only:
        print(kvb)
        return 0
    mode = "q4 main KV (EXPECTED) + fp8 indexer + fp8 draft" if args.q4 \
        else "fp8 (MEASURED)"
    print(f"mode               : {mode}")
    print(f"per_token_cost     : {cost / KIB:.2f} KiB/token")
    print(f"raw_pool_bytes    : {raw!r} B ({raw / KIB / KIB / KIB:.2f} GiB) for {args.maxlen} tokens")
    print(f"kvb                : {kvb} B ({kvb / KIB / KIB / KIB:.2f} GiB) "
          f"(raw x 1.0{SAFETY_MARGIN_PCT}, ceil; --kv-cache-memory-bytes)")
    print(f"expected_tokens    : {expected_tokens(kvb, args.q4)} tokens at kvb "
          f"(raw cost; must cover maxlen={args.maxlen})")
    if args.kvb is not None:
        toks = expected_tokens(args.kvb, args.q4)
        verdict = "OK covers maxlen" if toks >= args.maxlen else "INSUFFICIENT for maxlen"
        print(f"expected_tokens    : {toks} tokens at --kvb {args.kvb} B -> {verdict}")
    print("re-measure on the first real boot; constants are measurements, not guarantees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
