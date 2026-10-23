"""Unit tests for scripts/kvb.py and the KV_Q4 boot-profile gate.

Lives in tests/ on purpose: unittest discovery from the repo root does not
recurse into scripts/ (not a package). scripts/kvb.py is loaded directly by
file path and nothing here touches sys.path.

Layers under test:
  1. Pure arithmetic of scripts/kvb.py (per-token costs, the +2% budget,
     the inverse token capacity, the legacy fp8 formula cross-check) and the
     printed GiB labels (a wrong divisor once made the 1.47 GiB budget
     print as 1509.60 "GiB").
  2. The recipe's fail-closed q4 gate, exercised end-to-end through
     DRY_RUN=1 scripts/serve.sh (argv/env prints, no docker, no server)
     and through scripts/serve_inner.sh with a stubbed `vllm` binary on
     PATH (the argv the engine would actually receive).

The DEFAULT profile's argv is PINNED byte-for-byte: the no-env DRY_RUN block
(the default profile, configs/q4-32k.env — the measured 32K q4 boot) must
equal the measured boot's block exactly, so the default profile can never
drift. Run from the recipe root:

  python3 -m unittest
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE = os.path.join(REPO, "scripts", "serve.sh")
SERVE_INNER = os.path.join(REPO, "scripts", "serve_inner.sh")

_spec = importlib.util.spec_from_file_location(
    "kvb", os.path.join(REPO, "scripts", "kvb.py"))
kvb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kvb)  # type: ignore[union-attr]

# Knobs env.sh owns; tests must control them explicitly.
KNOBS = (
    "KVB", "KV_Q4", "MAXLEN", "KV_CACHE_DTYPE", "INDEXER_KV_DTYPE",
    "DSPARK", "DSPARK_KV_DTYPE", "DSPARK_ATTN_BACKEND",
    "DEFAULT_MAX_TOKENS", "FLASHINFER_AUTOTUNE",
    "INDEXER_MXFP8", "PLUGIN_OVERLAY", "PLUGIN_SRC", "PROFILE",
    "GUARD_MEM_FLOOR_KIB",
    "VLLM_EXL3_INDEXER_MXFP8",
)


def _env(extra):
    """A near-clean environment: os.environ minus every recipe knob."""
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update(extra)
    return env


def _dry_run(extra):
    """DRY_RUN=1 scripts/serve.sh; returns (rc, stdout+stderr)."""
    env = _env(extra)
    env["DRY_RUN"] = "1"
    p = subprocess.run(
        ["bash", SERVE], env=env, capture_output=True, text=True, timeout=120
    )
    return p.returncode, p.stdout + p.stderr


def _argv_block(text):
    """The `vllm serve ...` echo block of the DRY_RUN print (no timestamps).

    Ends at the block's own terminator line: stderr is captured after
    stdout, so late stderr (env.sh's INDEXER_MXFP8 best-effort preflight
    warning) would otherwise land inside the returned block."""
    lines = text.splitlines()
    out = []
    for ln in lines:
        if out or ln.startswith("vllm serve "):
            out.append(ln)
            if ln.startswith("  (no draft model path"):
                break
    return out


# The DEFAULT profile's DRY_RUN argv block at MAXLEN=32768, pinned from the
# measured 32K q4 boot. Every byte is asserted, so the default profile can
# never drift. @@ is the two-character line-continuation marker the print
# carries (backslash backslash); it is substituted below so this source
# file needs no backslash literals.
_PINNED_Q4_TEMPLATE = """vllm serve /models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw --served-model-name MiniMax-M3.1-Flash-2.15bpw @@
  --tensor-parallel-size 1 --language-model-only @@
  --block-size 128 --disable-hybrid-kv-cache-manager @@
  --max-model-len 32768 --max-num-seqs 1 @@
  --max-num-batched-tokens 1024 @@
  --kv-cache-memory-bytes 1582930330 @@
  --kv-cache-dtype nvfp4 @@
  --attention-config {"indexer_kv_dtype":"fp8"} @@
  --speculative-config {"method":"dspark","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"} @@
  --kernel-config {"enable_flashinfer_autotune":false} @@
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice @@
  --override-generation-config {"max_new_tokens": 32768}
  (no --gpu-memory-utilization anywhere: on GB10 the pool is sized by --kv-cache-memory-bytes)
  (no draft model path anywhere: the engine resolves dspark/ from the pack)"""
PINNED_Q4_ARGV = [
    ln.replace("@@", chr(92)) for ln in _PINNED_Q4_TEMPLATE.splitlines()
]


class KvbArithmeticTest(unittest.TestCase):
    """kvb.py constants and arithmetic, pinned to the measured numbers."""

    def test_costs(self):
        self.assertEqual(kvb.FP8_COST_KIB_PER_TOKEN, 72.65)   # measured
        self.assertEqual(kvb.Q4_COST_KIB_PER_TOKEN, 46.25)    # expected
        self.assertEqual(kvb.Q4_COST_KIB_PER_TOKEN, 33.75 + 7.5 + 5.0)
        self.assertEqual(kvb.FP8_COST_KIB_PER_TOKEN, 60 + 7.5 + 5 + 0.15)

    def test_q4_raw_pool_is_the_documented_number(self):
        # 46.25 KiB/token x 32768 tokens: the goal-context expectation.
        self.assertEqual(kvb.raw_pool_bytes(32768, 1), 1551892480.0)

    def test_budget_is_raw_plus_two_percent(self):
        self.assertEqual(kvb.SAFETY_MARGIN_PCT, 2)
        self.assertEqual(kvb.recommended_kvb(32768, 1), 1582930330)
        self.assertEqual(kvb.recommended_kvb(16384, 1), 791465165)
        self.assertEqual(kvb.recommended_kvb(32768, 0), 2486484075)

    def test_expected_tokens_inverse(self):
        # The q4 budget must buy back at least MAXLEN tokens at the raw cost.
        self.assertEqual(kvb.expected_tokens(1582930330, 1), 33423)
        self.assertGreaterEqual(kvb.expected_tokens(1582930330, 1), 32768)
        self.assertEqual(kvb.expected_tokens(1551892480, 1), 32768)
        # A main pool still at fp8 (72.65 KiB/token) would fall short:
        self.assertLess(kvb.expected_tokens(1582930330, 0), 32768)

    def test_legacy_formula_matches_env_sh(self):
        # env.sh's KVB_CALC (90 KiB/token x 1.10, 256 MiB aligned) must keep
        # producing the same numbers it always did.
        self.assertEqual(kvb.legacy_formula_kvb(32768), 3489660928)
        self.assertEqual(kvb.legacy_formula_kvb(16384), 1879048192)
        self.assertEqual(kvb.legacy_formula_kvb(262144), 26575110144)

    def test_fail_closed(self):
        for bad in (2, -1, "1", None, True):
            with self.assertRaises(ValueError):
                kvb.raw_pool_bytes(32768, bad)  # type: ignore[arg-type]
        for bad in (0, -1, 1.5, "8", None):
            with self.assertRaises(ValueError):
                kvb.raw_pool_bytes(bad, 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            kvb.expected_tokens(-1, 1)


class KvbCLITest(unittest.TestCase):
    """The CLI contract env.sh relies on."""

    def _kvb(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "kvb.py"), *args],
            capture_output=True, text=True, timeout=60,
        )

    def test_kvb_only_hook(self):
        p = self._kvb("--maxlen", "32768", "--q4", "1", "--kvb-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "1582930330")

    def test_full_print_pins_bytes_and_the_gib_label(self):
        # The byte values are pinned AND the GiB label must be a true
        # binary-GiB figure: 1582930330 B is 1.47 GiB, never 1509.60 "GiB"
        # (a 1024**2 divisor once produced that).
        p = self._kvb("--maxlen", "32768", "--q4", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("46.25 KiB/token", p.stdout)
        self.assertIn("1551892480.0 B", p.stdout)
        self.assertIn("1582930330 B (1.47 GiB)", p.stdout)
        self.assertIn("1551892480.0 B (1.45 GiB)", p.stdout)
        self.assertIn("33423 tokens at kvb", p.stdout)
        # A budget below the raw requirement is reported INSUFFICIENT.
        p = self._kvb("--maxlen", "32768", "--q4", "1",
                      "--kvb", "1000000000")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("INSUFFICIENT for maxlen", p.stdout)

    def test_rejects_bad_mode(self):
        p = self._kvb("--maxlen", "32768", "--q4", "7", "--kvb-only")
        self.assertNotEqual(p.returncode, 0)


class ServeDryRunTest(unittest.TestCase):
    """The q4 gate end-to-end through DRY_RUN=1 scripts/serve.sh."""

    def test_default_argv_is_the_measured_q4_32k_boot(self):
        # No env: the default profile (configs/q4-32k.env) must print the
        # measured 32K q4 boot's argv byte-for-byte.
        rc, out = _dry_run({})
        self.assertEqual(rc, 0, out)
        self.assertEqual(_argv_block(out), PINNED_Q4_ARGV)
        self.assertIn("VLLM_EXL3_Q4KV=1 in the container env", out)
        self.assertIn("--kv-cache-memory-bytes 1582930330", out)
        # The measured guard floor opt-in is part of the profile.
        self.assertIn("MemAvailable floor 204 MiB", out)

    def test_fp8_profiles_pin_their_own_measured_pools(self):
        # fp8-16k: the measured 1.238e9 fp8 pool, fp8 main dtype, no q4
        # overlay, and the PUBLIC default 0.5 GiB guard floor.
        rc, out = _dry_run({"PROFILE": "fp8-16k"})
        self.assertEqual(rc, 0, out)
        self.assertIn("--kv-cache-dtype fp8", out)
        self.assertIn("--kv-cache-memory-bytes 1238000000", out)
        self.assertNotIn("Q4KV", out)
        self.assertIn("MemAvailable floor 512 MiB", out)
        # fp8-24k: the measured 1.857e9 fp8 pool and its 0.2 GiB floor.
        rc, out = _dry_run({"PROFILE": "fp8-24k"})
        self.assertEqual(rc, 0, out)
        self.assertIn("--kv-cache-memory-bytes 1857000000", out)
        self.assertNotIn("Q4KV", out)
        self.assertIn("MemAvailable floor 204 MiB", out)

    def test_unknown_profile_is_refused(self):
        rc, out = _dry_run({"PROFILE": "48k"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("no such profile", out)

    def test_wider_context_via_env_keeps_the_pinned_pool(self):
        # A MAXLEN the operator passes by env cannot widen the pool: the
        # profile's pinned KVB wins, so the engine simply allocates fewer
        # tokens than the (now larger) context limit.
        rc, out = _dry_run({"MAXLEN": "49152"})
        self.assertEqual(rc, 0, out)
        self.assertIn("--max-model-len 49152", out)
        self.assertIn("--kv-cache-memory-bytes 1582930330", out)

    def test_q4_gates_refuse_fail_closed(self):
        cases = [
            ("bf16 draft under q4", {
                "KV_Q4": "1", "DSPARK_KV_DTYPE": "auto"}),
            ("FLASH_ATTN with fp8 draft", {
                "KV_Q4": "1", "DSPARK_KV_DTYPE": "fp8",
                "DSPARK_ATTN_BACKEND": "FLASH_ATTN"}),
            ("nvfp4 indexer", {
                "KV_Q4": "1", "INDEXER_KV_DTYPE": "nvfp4",
                "DSPARK_ATTN_BACKEND": "TRITON_ATTN",
                "DSPARK_KV_DTYPE": "fp8"}),
            ("no draft", {
                "KV_Q4": "1", "DSPARK": "0",
                "DSPARK_ATTN_BACKEND": "TRITON_ATTN",
                "DSPARK_KV_DTYPE": "fp8"}),
            ("conflicting main dtype knob", {
                "KV_Q4": "1", "KV_CACHE_DTYPE": "bfloat16",
                "DSPARK_ATTN_BACKEND": "TRITON_ATTN",
                "DSPARK_KV_DTYPE": "fp8"}),
        ]
        for desc, extra in cases:
            with self.subTest(desc):
                rc, out = _dry_run(extra)
                self.assertNotEqual(rc, 0, out)
                self.assertIn("KV_Q4 gate: refusing to serve", out)

    def test_nvfp4_without_the_overlay_is_refused(self):
        rc, out = _dry_run({"KV_Q4": "0", "KV_CACHE_DTYPE": "nvfp4"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("guaranteed upstream raise", out)

    def test_garbage_kv_q4_is_refused(self):
        rc, out = _dry_run({"KV_Q4": "yes"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("KV_Q4 must be 0 or 1", out)

    def test_kvb_auto_and_overrides(self):
        # auto: kvb.py budget for the current mode; explicit numbers still win.
        rc, out = _dry_run({
            "KV_Q4": "1", "KVB": "auto",
            "DSPARK_ATTN_BACKEND": "TRITON_ATTN",
            "DSPARK_KV_DTYPE": "fp8",
        })
        self.assertEqual(rc, 0, out)
        self.assertIn("KVB_SRC=auto-kvb-py", out)
        self.assertIn("--kv-cache-memory-bytes 1582930330", out)
        rc, out = _dry_run({
            "KV_Q4": "1", "KVB": "1600000000",
            "DSPARK_ATTN_BACKEND": "TRITON_ATTN",
            "DSPARK_KV_DTYPE": "fp8",
        })
        self.assertEqual(rc, 0, out)
        self.assertIn("KVB_SRC=override", out)
        self.assertIn("--kv-cache-memory-bytes 1600000000", out)


class ServeInnerStubTest(unittest.TestCase):
    """scripts/serve_inner.sh with a stubbed `vllm` on PATH: the argv the
    engine would receive, and the in-container q4 gate."""

    Q4_ENV = {
        "VLLM_EXL3_Q4KV": "1",
        "KV_CACHE_DTYPE": "nvfp4",
        "INDEXER_KV_DTYPE": "fp8",
        "DSPARK": "1", "DSPARK_NUM_SPEC": "3",
        "DSPARK_ATTN_BACKEND": "TRITON_ATTN", "DSPARK_KV_DTYPE": "fp8",
        "DSPARK_DRAFT_SAMPLE": "probabilistic", "DSPARK_REJECTION": "block",
        "MAXLEN": "32768", "MAX_NUM_SEQS": "1", "BLOCK_SIZE": "128",
        "KVB": "1582930330", "MAX_NUM_BATCHED_TOKENS": "1024",
        "FLASHINFER_AUTOTUNE": "0", "DEFAULT_MAX_TOKENS": "32768",
        "SERVED_MODEL_NAME": "MiniMax-M3.1-Flash-2.15bpw",
        "VLLM_EXL3_MODEL_DIR":
            "/models/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw",
    }

    def _run_inner(self, tmp, extra):
        binp = os.path.join(tmp, "bin")
        os.makedirs(binp, exist_ok=True)
        argv_file = os.path.join(tmp, "argv")
        stub = os.path.join(binp, "vllm")
        with open(stub, "w") as f:
            f.write('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$STUB_ARGV"\nexit 0\n')
        os.chmod(stub, 0o755)
        env = _env(extra)
        env["PATH"] = binp + os.pathsep + env.get("PATH", "")
        env["STUB_ARGV"] = argv_file
        p = subprocess.run(
            ["bash", SERVE_INNER], env=env, capture_output=True,
            text=True, timeout=60,
        )
        argv = None
        if os.path.exists(argv_file):
            with open(argv_file) as f:
                # The stub dumps one arg per line; normalize to a single
                # line so flag/value pairs can be matched as substrings.
                argv = " ".join(f.read().split())
        return p.returncode, p.stdout + p.stderr, argv

    def test_q4_argv_reaches_the_engine(self):
        with tempfile.TemporaryDirectory(prefix="q4inner.") as tmp:
            rc, out, argv = self._run_inner(tmp, dict(self.Q4_ENV))
        self.assertEqual(rc, 0, out)
        self.assertIsNotNone(argv)
        assert argv is not None
        self.assertIn("--kv-cache-dtype nvfp4", argv)
        self.assertIn('--attention-config {"indexer_kv_dtype":"fp8"}', argv)
        self.assertIn('"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8"', argv)
        self.assertIn("--kv-cache-memory-bytes 1582930330", argv)

    def test_fp8_argv_unchanged_and_no_q4(self):
        env = dict(self.Q4_ENV)
        env.pop("VLLM_EXL3_Q4KV")
        env["KV_CACHE_DTYPE"] = "fp8"
        env["KVB"] = "1238000000"
        with tempfile.TemporaryDirectory(prefix="q4inner.") as tmp:
            rc, out, argv = self._run_inner(tmp, env)
        self.assertEqual(rc, 0, out)
        self.assertIsNotNone(argv)
        assert argv is not None
        self.assertIn("--kv-cache-dtype fp8", argv)
        self.assertNotIn("nvfp4", argv)

    def test_q4_gate_refuses_in_container(self):
        # The overlay flag without the nvfp4 main pool: refuse before exec.
        env = dict(self.Q4_ENV)
        env["KV_CACHE_DTYPE"] = "fp8"
        with tempfile.TemporaryDirectory(prefix="q4inner.") as tmp:
            rc, out, argv = self._run_inner(tmp, env)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("q4 gate:", out)
        self.assertIsNone(argv)
        # bf16 draft under q4: refuse too.
        env = dict(self.Q4_ENV)
        env["DSPARK_KV_DTYPE"] = "auto"
        with tempfile.TemporaryDirectory(prefix="q4inner.") as tmp:
            rc, out, argv = self._run_inner(tmp, env)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("q4 gate:", out)
        self.assertIsNone(argv)


if __name__ == "__main__":
    unittest.main()
