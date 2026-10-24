"""Unit tests for the INDEXER_MXFP8 knob and its fail-closed preflight.

Lives in tests/ for the same reason test_q4boot.py does (unittest
discovery from the root does not recurse into scripts/). Nothing here
touches sys.path; the knob is exercised end-to-end through DRY_RUN=1
scripts/serve.sh (no docker, no server) and through scripts/serve_inner.sh
with a stubbed `vllm` binary on PATH.

Layers under test:
  1. Knob resolution: default 1 (armed), explicit 0 (variable unset),
     anything else refused.
  2. The fail-closed preflight in env.sh: a git checkout below the
     pinned plugin rev whose tree lacks the function refuses; a readable
     plugin tree lacking the function refuses; a tree carrying the
     function is allowed; an undecidable dir (no checkout, no readable
     tree) warns and continues; INDEXER_MXFP8=0 skips the preflight
     entirely.
  3. Propagation wiring: serve.sh passes -e VLLM_EXL3_INDEXER_MXFP8=1 to
     the container ONLY at INDEXER_MXFP8=1, and serve_inner.sh reports the
     in-container value in its banner (the plugin reads the variable; it
     never reaches the engine argv).

Run from the recipe root:  python3 -m unittest
"""
import os
import subprocess
import tempfile
import unittest
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE = os.path.join(REPO, "scripts", "serve.sh")
SERVE_INNER = os.path.join(REPO, "scripts", "serve_inner.sh")

# Knobs env.sh owns; tests must control them explicitly (superset of
# test_q4boot.py's list: the indexer knob and the plugin source knobs).
KNOBS = (
    "KVB", "KV_Q4", "MAXLEN", "KV_CACHE_DTYPE", "INDEXER_KV_DTYPE",
    "DSPARK", "DSPARK_KV_DTYPE", "DSPARK_ATTN_BACKEND",
    "DEFAULT_MAX_TOKENS", "FLASHINFER_AUTOTUNE",
    "INDEXER_MXFP8", "PLUGIN_OVERLAY", "PLUGIN_SRC", "PROFILE",
    "GUARD_MEM_FLOOR_KIB",
    "VLLM_EXL3_INDEXER_MXFP8",
)

ARMED_LINE = (
    "  indexer_mxfp8: INDEXER_MXFP8=1 -> VLLM_EXL3_INDEXER_MXFP8=1 in the"
    " container env"
)
UNSET_LINE = (
    "  indexer_mxfp8: INDEXER_MXFP8=0 -> VLLM_EXL3_INDEXER_MXFP8 unset"
)


def _env(extra):
    """A near-clean environment: os.environ minus every recipe knob."""
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env.update(extra)
    return env


def _dry_run(script, extra):
    """DRY_RUN=1 <script>; returns (rc, stdout+stderr)."""
    env = _env(extra)
    env["DRY_RUN"] = "1"
    p = subprocess.run(
        ["bash", script], env=env, capture_output=True, text=True, timeout=120
    )
    return p.returncode, p.stdout + p.stderr


def _normalize(text, *roots):
    """Blank the timestamped run dir and the two checkout locations so two
    runs from different directories can be diffed (the recipe home is the
    only path that differs; every other byte must be identical)."""
    for r in roots:
        text = text.replace(r, "REPO_HOME")
    return re.sub(r"runs/\d{8}-\d{6}", "runs/TIMESTAMP", text)


def _plugin_dir(tmp, name, function_text):
    """A plugin-shaped dir; exl3.py contains function_text (or lacks it)."""
    d = os.path.join(tmp, name)
    os.makedirs(os.path.join(d, "src", "vllm_exl3"), exist_ok=True)
    with open(os.path.join(d, "src", "vllm_exl3", "exl3.py"), "w") as f:
        f.write(function_text)
    return d


def _git_dir(tmp, name):
    """A real git checkout with one commit and NO indexer code."""
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "README"), "w") as f:
        f.write("not the indexer branch\n")
    subprocess.run(["git", "-C", d, "init", "-q"], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", d, "add", "README"], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "-m", "pre-indexer"], check=True, timeout=60)
    return d


class KnobTest(unittest.TestCase):
    """Default/0/invalid resolution, end-to-end through DRY_RUN."""

    def test_default_is_1_and_arms_the_container_env(self):
        # The default (INDEXER_MXFP8 unset) must resolve to 1 when the
        # runtime plugin tree carries the indexer code.
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            allow = _plugin_dir(
                tmp, "allow",
                "def indexer_mxfp8_move_to_device(t):\n    return t\n")
            rc, out = _dry_run(SERVE, {"PLUGIN_OVERLAY": allow})
        self.assertEqual(rc, 0, out)
        self.assertIn(ARMED_LINE, out)
        self.assertNotIn(UNSET_LINE, out)

    def test_zero_output_differs_from_armed_by_exactly_one_line(self):
        # INDEXER_MXFP8=0 must reproduce the armed (default) DRY_RUN output
        # byte-for-byte except the one summary line: same argv, same pool,
        # same guard line — only the indexer knob's own line changes.
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            allow = _plugin_dir(
                tmp, "allow",
                "def indexer_mxfp8_move_to_device(t):\n    return t\n")
            rc1, out1 = _dry_run(SERVE, {"PLUGIN_OVERLAY": allow})
            rc0, out0 = _dry_run(
                SERVE, {"INDEXER_MXFP8": "0", "PLUGIN_OVERLAY": allow})
        self.assertEqual(rc1, 0, out1)
        self.assertEqual(rc0, 0, out0)
        lines1 = _normalize(out1, REPO).splitlines()
        lines0 = _normalize(out0, REPO).splitlines()
        # Each run's own indexer_mxfp8 summary line (and nothing else) is
        # dropped, then the remaining streams must be byte-identical.
        rest1 = [l for l in lines1 if "indexer_mxfp8:" not in l]
        rest0 = [l for l in lines0 if "indexer_mxfp8:" not in l]
        self.assertTrue(any(ARMED_LINE in l for l in lines1))
        self.assertTrue(any(UNSET_LINE in l for l in lines0))
        self.assertEqual(rest1, rest0)

    def test_invalid_value_is_refused(self):
        for bad in ("yes", "2", "on"):
            with self.subTest(bad):
                rc, out = _dry_run(SERVE, {"INDEXER_MXFP8": bad})
                self.assertNotEqual(rc, 0, out)
                self.assertIn("INDEXER_MXFP8 must be 0 or 1", out)


class PreflightTest(unittest.TestCase):
    """The fail-closed overlay preflight in env.sh."""

    def test_git_checkout_without_the_code_refuses(self):
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            bad = _git_dir(tmp, "gitbad")
            rc, out = _dry_run(SERVE, {"PLUGIN_OVERLAY": bad})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("INDEXER_MXFP8 gate: refusing to serve", out)
        self.assertIn("41fdb7d", out)
        self.assertIn("indexer_mxfp8_move_to_device", out)

    def test_readable_tree_without_the_function_refuses(self):
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            bad = _plugin_dir(
                tmp, "plainbad", "def other():\n    pass\n")
            rc, out = _dry_run(SERVE, {"PLUGIN_OVERLAY": bad})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("INDEXER_MXFP8 gate: refusing to serve", out)

    def test_tree_with_the_function_is_allowed(self):
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            allow = _plugin_dir(
                tmp, "allow",
                "def indexer_mxfp8_move_to_device(t):\n    return t\n")
            rc, out = _dry_run(SERVE, {"PLUGIN_OVERLAY": allow})
        self.assertEqual(rc, 0, out)
        self.assertIn(ARMED_LINE, out)
        self.assertNotIn("INDEXER_MXFP8 gate", out)

    def test_undecidable_dirs_warn_and_continue(self):
        cases = [
            ("nonexistent overlay", {
                "PLUGIN_OVERLAY": "/nonexistent/imx-overlay"}),
            ("no overlay, nonexistent plugin source", {
                "PLUGIN_SRC": "/nonexistent/imx-plugin-src"}),
        ]
        for desc, extra in cases:
            with self.subTest(desc):
                rc, out = _dry_run(SERVE, extra)
                self.assertEqual(rc, 0, out)
                self.assertIn("WARNING: INDEXER_MXFP8=1", out)
                self.assertIn("cannot be", out)
                # Warn and CONTINUE: the knob stays armed.
                self.assertIn(ARMED_LINE, out)

    def test_zero_skips_the_preflight_entirely(self):
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            bad = _plugin_dir(tmp, "plainbad", "def other():\n    pass\n")
            rc, out = _dry_run(
                SERVE, {"INDEXER_MXFP8": "0", "PLUGIN_OVERLAY": bad})
        self.assertEqual(rc, 0, out)
        self.assertNotIn("INDEXER_MXFP8 gate", out)
        self.assertNotIn("WARNING: INDEXER_MXFP8", out)
        self.assertIn(UNSET_LINE, out)


class PropagationTest(unittest.TestCase):
    """serve.sh wiring and the in-container banner via stubbed vllm."""

    def test_serve_sh_passes_the_env_var_only_at_1(self):
        with open(SERVE) as f:
            src = f.read()
        self.assertEqual(
            src.count("INDEXER_MXFP8_ENV=(-e VLLM_EXL3_INDEXER_MXFP8=1)"), 1)
        self.assertIn(
            'if [ "$INDEXER_MXFP8" = "1" ]; then\n'
            "  INDEXER_MXFP8_ENV=(-e VLLM_EXL3_INDEXER_MXFP8=1)",
            src)
        # The array is expanded exactly once, inside the docker run.
        self.assertEqual(src.count('"${INDEXER_MXFP8_ENV[@]}"'), 1)

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
                argv = " ".join(f.read().split())
        return p.returncode, p.stdout + p.stderr, argv

    INNER_ENV = {
        "KV_CACHE_DTYPE": "fp8", "INDEXER_KV_DTYPE": "fp8",
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

    def test_env_var_reaches_the_container_but_never_the_argv(self):
        # With the variable in the container env, serve_inner's banner
        # reports it armed; the engine argv is IDENTICAL to the argv
        # without it (the plugin reads the variable, not the argv).
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            rc1, out1, argv1 = self._run_inner(
                tmp, dict(self.INNER_ENV, VLLM_EXL3_INDEXER_MXFP8="1"))
        with tempfile.TemporaryDirectory(prefix="imx.") as tmp:
            rc0, out0, argv0 = self._run_inner(tmp, dict(self.INNER_ENV))
        self.assertEqual(rc1, 0, out1)
        self.assertEqual(rc0, 0, out0)
        self.assertIn("indexer_mxfp8=1", out1)
        self.assertIn("indexer_mxfp8=0", out0)
        self.assertNotIn("VLLM_EXL3_INDEXER_MXFP8", out0)
        self.assertEqual(argv1, argv0)
        self.assertIsNotNone(argv1)


if __name__ == "__main__":
    unittest.main()
