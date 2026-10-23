#!/bin/bash
# Behavioural test of guard.sh's narrow stop. `guard.sh stop` must end ONLY
# this recipe's own recorded guard -- never another recipe's guard, not even
# one whose pid was planted into this recipe's runs dir (another serving recipe on the
# same box can run the same script name). The sandbox is
# hermetic: every child runs with a PATH whose `docker` is a stub that says
# the container is up (so the watchers arm and stay alive for the check
# window) and whose `sudo` is a pass-through for boxes without passwordless
# sudo (e.g. a root CI box); each sandbox recipe gets its own RUNS_DIR under
# a temp dir; and the planted sysctl.before carries the host's CURRENT
# values so the stop's restore step rewrites them unchanged. The property
# under test is WHICH pids a stop may kill, and that does not depend on
# sudo or docker.
#
#   bash tests/test_guard_stop.sh     # prints PASS/FAIL, rc 0/1
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$HERE/.." && pwd)/scripts"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/guard-stop-test.XXXXXX")"
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

# Two sandbox recipes: A = this recipe, B = a foreign recipe standing in for
# the DSV4.1 production one. Each scripts dir gets the real guard.sh and a
# minimal env.sh pinning RUNS_DIR inside the sandbox (the real env.sh is
# deliberately NOT copied: its RUNS_DIR and floors belong to the real host).
for r in a b; do
  mkdir -p "$WORK/$r/scripts" "$WORK/$r/runs/r1"
  cp "$SCRIPTS/guard.sh" "$WORK/$r/scripts/guard.sh"
  printf 'RUNS_DIR="%s/%s/runs"\n' "$WORK" "$r" >"$WORK/$r/scripts/env.sh"
done

# The host's CURRENT sysctl values as A's saved baseline: the stop branch's
# restore step then rewrites the same values (a no-op), so the test never
# changes host policy even when it reaches that code.
for k in vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness; do
  printf '%s=%s\n' "$k" "$(cat "/proc/sys/${k//.//}")" >>"$WORK/a/runs/r1/sysctl.before"
done

# The watchers get floor 0 and their swap base pinned to the host's CURRENT
# swap usage: the guard's own memory/swap kill checks are not under test here
# and must not fire mid-test (with base 0 any real swap usage would).
CUR_SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo)

start_watch() {  # start_watch <recipe> -> echoes the watcher pid
  local r="$1"
  PATH="$SANDBOX_PATH" bash "$WORK/$r/scripts/guard.sh" watch "test-ctr-$r" \
    "$WORK/$r/runs/r1/stage" "$WORK/$r/runs/r1/guard.log" 0 "$CUR_SWAP" 65536 \
    >/dev/null 2>&1 &
  local p=$!
  PIDS+=("$p")
  say "$p"
}

PID_A="$(start_watch a)"
PID_B="$(start_watch b)"

# The watchers must have recorded exactly their own pid in their own run dir.
sleep 0.5
check "watcher A recorded its own pid" "$([ "$(head -n1 -- "$WORK/a/runs/r1/guard.pid" 2>/dev/null)" = "$PID_A" ]; echo $?)"
check "watcher B recorded its own pid" "$([ "$(head -n1 -- "$WORK/b/runs/r1/guard.pid" 2>/dev/null)" = "$PID_B" ]; echo $?)"

# Plant B's pid in A's runs dir: even a foreign pid file inside this recipe's
# runs dir must not make the stop kill B (the /proc cmdline check).
mkdir -p "$WORK/a/runs/planted"
printf '%s\n' "$PID_B" >"$WORK/a/runs/planted/guard.pid"

# A's status must count only A's own guard, never B's.
STATUS_A="$(PATH="$SANDBOX_PATH" bash "$WORK/a/scripts/guard.sh" status "test-ctr-a" 2>/dev/null)"
check "status counts only the own guard" "$(printf '%s\n' "$STATUS_A" | grep -q '^own guard procs: 1$'; echo $?)"

# The stop under test: A's stop, with both watchers alive.
PATH="$SANDBOX_PATH" bash "$WORK/a/scripts/guard.sh" stop "test-ctr-a" >/dev/null 2>&1
sleep 0.5
check "stop ended the OWN guard" "$(! kill -0 "$PID_A" 2>/dev/null; echo $?)"
check "stop left the FOREIGN guard running" "$(kill -0 "$PID_B" 2>/dev/null; echo $?)"

if [ "$FAILS" = 0 ]; then say "PASS: guard.sh stop is narrow (own pid file + cmdline verified)"; exit 0
else say "FAIL: $FAILS check(s) failed"; exit 1; fi
