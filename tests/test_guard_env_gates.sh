#!/bin/bash
# Behavioural test of guard.sh's immunity to env.sh's fail-closed gates and of
# the stale-pid noise fix. env.sh's gates (the KV_Q4 gate, the INDEXER_MXFP8
# validity check and overlay preflight, the kvb.py derivation) call `exit 1`,
# and an `exit` in a sourced file terminates the SOURCING shell, so before the
# RECIPE_ENV_NO_GATES fix a gate that fired at guard time (a knob changed under
# a running serve) silently killed `guard.sh stop` before it could end its own
# watcher. This test runs the REAL env.sh -- not a stub -- with knobs that make
# it refuse a launch (KV_Q4=1 with DSPARK_KV_DTYPE=auto, INDEXER_MXFP8=7) and
# asserts:
#   - a watcher still arms and records its pid under those knobs,
#   - `guard.sh stop` under those knobs reaches its own stop logic: rc 0, no
#     unbound-variable error, the OWN watcher killed, a FOREIGN watcher left
#     running,
#   - a stale guard.pid no longer prints
#     "/proc/<pid>/cmdline: No such file or directory",
#   - serve.sh STILL refuses the same knobs: the gates stay fail-closed for
#     every launch (DRY_RUN=1 so a broken gate can never launch anything).
# The sandbox is hermetic like test_guard_stop.sh's: every child runs with a
# PATH whose `docker` stub says the container is up and whose `sudo` is a
# pass-through, and each sandbox recipe gets its own RUNS_DIR (preset in the
# environment; env.sh keeps a preset RUNS_DIR).
#
#   bash tests/test_guard_env_gates.sh  # prints PASS/FAIL, rc 0/1
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$HERE/.." && pwd)/scripts"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/guard-env-gates-test.XXXXXX")"
FAILS=0
PIDS=()

# shellcheck disable=SC2317  # invoked by the EXIT trap below
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  rm -rf "$WORK"
}
trap cleanup EXIT

say() { printf '%s\n' "$*"; }
check() {  # check <desc> <rc-ish bool: 0 = good>
  if [ "$2" = 0 ]; then say "ok    : $1"; else say "FAIL  : $1"; FAILS=$((FAILS + 1)); fi
}

# Hermetic stubs, always first on the children's PATH: a `docker` whose
# inspect says the (nonexistent) container is running, and a `sudo`
# pass-through for boxes without passwordless sudo.
mkdir -p "$WORK/bin"
printf '#!/bin/sh\ncase "$1" in inspect) echo true; exit 0;; *) exit 1;; esac\n' >"$WORK/bin/docker"
printf '#!/bin/sh\n[ "$1" = "-n" ] && shift\nexec "$@"\n' >"$WORK/bin/sudo"
chmod 700 "$WORK/bin/docker" "$WORK/bin/sudo"
SANDBOX_PATH="$WORK/bin:$PATH"

# The knobs that make env.sh refuse a launch. Both watchers and the stop run
# with them, proving no env.sh gate can take the guard down at guard time.
BAD_KNOBS=(KV_Q4=1 DSPARK_KV_DTYPE=auto INDEXER_MXFP8=7)

# Two sandbox recipes: A = this recipe, B = a foreign recipe standing in for
# another production guard. Unlike test_guard_stop.sh this test copies the REAL
# guard.sh AND the REAL env.sh -- the gates under test are env.sh's own -- and
# pins RUNS_DIR into each sandbox through the environment (env.sh keeps a
# preset RUNS_DIR).
for r in a b; do
  mkdir -p "$WORK/$r/scripts" "$WORK/$r/runs/r1"
  cp "$SCRIPTS/guard.sh" "$WORK/$r/scripts/guard.sh"
  cp "$SCRIPTS/env.sh" "$WORK/$r/scripts/env.sh"
done

# The host's CURRENT sysctl values as A's saved baseline: the stop branch's
# restore step then rewrites the same values (a no-op), so the test never
# changes host policy even when it reaches that code.
for k in vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness; do
  printf '%s=%s\n' "$k" "$(cat "/proc/sys/${k//.//}")" >>"$WORK/a/runs/r1/sysctl.before"
done

# A stale pid file whose pid is long gone: before the redirect-order fix the
# stop printed ".../guard.sh: line NN: /proc/<pid>/cmdline: No such file or
# directory" for it on every stop.
STALE_PID=4000000
while [ -d "/proc/$STALE_PID" ]; do STALE_PID=$((STALE_PID + 1)); done
mkdir -p "$WORK/a/runs/r1/stale"
printf '%s\n' "$STALE_PID" >"$WORK/a/runs/r1/stale/guard.pid"

# The watchers get floor 0 and their swap base pinned to the host's CURRENT
# swap usage: the guard's own memory/swap kill checks are not under test here
# and must not fire mid-test (with base 0 any real swap usage would).
CUR_SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo)

start_watch() {  # start_watch <recipe> -> echoes the watcher pid
  local r="$1"
  env "${BAD_KNOBS[@]}" "RUNS_DIR=$WORK/$r/runs" PATH="$SANDBOX_PATH" \
    bash "$WORK/$r/scripts/guard.sh" watch "test-ctr-$r" \
    "$WORK/$r/runs/r1/stage" "$WORK/$r/runs/r1/guard.log" 0 "$CUR_SWAP" 65536 \
    >/dev/null 2>&1 &
  local p=$!
  PIDS+=("$p")
  say "$p"
}

PID_A="$(start_watch a)"
PID_B="$(start_watch b)"

# The watchers must have SURVIVED sourcing the real env.sh with knobs that
# make it refuse a launch, and recorded exactly their own pid.
sleep 0.5
check "watcher A survived the gated env.sh and recorded its own pid" \
  "$([ "$(head -n1 -- "$WORK/a/runs/r1/guard.pid" 2>/dev/null)" = "$PID_A" ]; echo $?)"
check "watcher B survived the gated env.sh and recorded its own pid" \
  "$([ "$(head -n1 -- "$WORK/b/runs/r1/guard.pid" 2>/dev/null)" = "$PID_B" ]; echo $?)"

# The stop under test: A's stop, with both watchers alive, the same refusing
# knobs in the environment, and the stale pid file planted above.
STOP_OUT="$WORK/stop.out"
env "${BAD_KNOBS[@]}" "RUNS_DIR=$WORK/a/runs" PATH="$SANDBOX_PATH" \
  bash "$WORK/a/scripts/guard.sh" stop "test-ctr-a" >"$STOP_OUT" 2>&1
STOP_RC=$?
sleep 0.5
check "stop with refusing knobs exits rc 0 (reached its own stop logic)" "$([ "$STOP_RC" = 0 ]; echo $?)"
check "stop output has no unbound-variable error" \
  "$(! grep -q 'unbound variable' "$STOP_OUT"; echo $?)"
check "stop output has no stale-pid cmdline noise" \
  "$(! grep -q 'cmdline: No such file or directory' "$STOP_OUT"; echo $?)"
check "stop ended the OWN guard" "$(! kill -0 "$PID_A" 2>/dev/null; echo $?)"
check "stop left the FOREIGN guard running" "$(kill -0 "$PID_B" 2>/dev/null; echo $?)"

# serve.sh must STILL refuse the same knobs: the gates stay fail-closed for
# every launch. DRY_RUN=1 so a removed gate could never launch anything real,
# and the sandbox PATH first so no real docker could run either.
env "${BAD_KNOBS[@]}" DRY_RUN=1 PATH="$SANDBOX_PATH" bash "$SCRIPTS/serve.sh" \
  >"$WORK/serve_q4.out" 2>&1
SRV_Q4_RC=$?
check "serve.sh refuses KV_Q4=1 + DSPARK_KV_DTYPE=auto (rc $SRV_Q4_RC != 0)" \
  "$([ "$SRV_Q4_RC" != 0 ]; echo $?)"
check "serve.sh names the KV_Q4 gate in the refusal" \
  "$(grep -q 'KV_Q4 gate: refusing to serve' "$WORK/serve_q4.out"; echo $?)"

env INDEXER_MXFP8=7 DRY_RUN=1 PATH="$SANDBOX_PATH" bash "$SCRIPTS/serve.sh" \
  >"$WORK/serve_imx.out" 2>&1
SRV_IMX_RC=$?
check "serve.sh refuses INDEXER_MXFP8=7 (rc $SRV_IMX_RC != 0)" \
  "$([ "$SRV_IMX_RC" != 0 ]; echo $?)"
check "serve.sh names the INDEXER_MXFP8 check in the refusal" \
  "$(grep -q 'INDEXER_MXFP8 must be 0 or 1' "$WORK/serve_imx.out"; echo $?)"

if [ "$FAILS" = 0 ]; then say "PASS: guard.sh survives env.sh gates; launch gates intact; no stale-pid noise"; exit 0
else say "FAIL: $FAILS check(s) failed"; exit 1; fi
