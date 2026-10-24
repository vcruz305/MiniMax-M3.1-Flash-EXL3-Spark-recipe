"""Tests for the mimalloc purge/reset knobs (MIMALLOC_PURGE_DELAY/RESET_DELAY).

torch's libc10.so links mimalloc statically on aarch64, so the EngineCore is a
mimalloc client and keeps freed load-time CPU memory inside its arena until
the purge/reset delays expire (measured: ~879 MiB held in a 1 GiB arena until
the end of a ~30K-token prefill). The recipe therefore puts both delays into
the serving container env with a default of 0, an empty value meaning "pass
nothing" (upstream mimalloc defaults), and a fail-closed check for anything
that is not a non-negative integer.

Everything here runs through the real scripts/env.sh and the real
DRY_RUN=1 scripts/serve.sh: no docker, no server, no GPU.

  python3 -m unittest discover -s tests          # from the recipe root
"""
import os
import subprocess
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_SH = os.path.join(REPO, "scripts", "env.sh")
SERVE = os.path.join(REPO, "scripts", "serve.sh")

KNOBS = ("MIMALLOC_PURGE_DELAY", "MIMALLOC_RESET_DELAY")


def _env(extra):
    """A near-clean environment: os.environ minus the two knobs under test."""
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


def _plan_line(text, prefix):
    """The single DRY_RUN plan line starting with <prefix> (no timestamps)."""
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


class DefaultTest(unittest.TestCase):
    """Both knobs default to 0, i.e. purge/reset immediately."""

    def test_default_is_zero_and_shown_in_the_plan(self):
        rc, out = _dry_run({})
        self.assertEqual(rc, 0, out)
        plan = _plan_line(out, "mimalloc:")
        self.assertIn("-e MIMALLOC_PURGE_DELAY=0", plan)
        self.assertIn("-e MIMALLOC_RESET_DELAY=0", plan)

    def test_env_sh_defaults_are_zero(self):
        p = subprocess.run(
            ["bash", "-c",
             '. scripts/env.sh >/dev/null && '
             'printf "%s %s\\n" "$MIMALLOC_PURGE_DELAY" "$MIMALLOC_RESET_DELAY"'],
            cwd=REPO, env=_env({}), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "0 0")


class OverrideTest(unittest.TestCase):
    """Explicit values reach the plan verbatim; empty means 'pass nothing'."""

    def test_values_are_passed_through(self):
        rc, out = _dry_run({"MIMALLOC_PURGE_DELAY": "250",
                            "MIMALLOC_RESET_DELAY": "1000"})
        self.assertEqual(rc, 0, out)
        plan = _plan_line(out, "mimalloc:")
        self.assertIn("-e MIMALLOC_PURGE_DELAY=250", plan)
        self.assertIn("-e MIMALLOC_RESET_DELAY=1000", plan)

    def test_empty_means_upstream_defaults(self):
        # An explicit empty value must NOT collapse back to 0: nothing is
        # passed, so the container keeps the upstream mimalloc defaults.
        rc, out = _dry_run({"MIMALLOC_PURGE_DELAY": "", "MIMALLOC_RESET_DELAY": ""})
        self.assertEqual(rc, 0, out)
        self.assertEqual(
            _plan_line(out, "mimalloc:"),
            "mimalloc: none (both knobs empty: upstream mimalloc defaults, "
            "freed memory stays in the arena until purge)")
        self.assertNotIn("-e MIMALLOC_PURGE_DELAY", out)
        self.assertNotIn("-e MIMALLOC_RESET_DELAY", out)

    def test_one_empty_one_set(self):
        rc, out = _dry_run({"MIMALLOC_PURGE_DELAY": "", "MIMALLOC_RESET_DELAY": "0"})
        self.assertEqual(rc, 0, out)
        plan = _plan_line(out, "mimalloc:")
        self.assertIn("-e MIMALLOC_RESET_DELAY=0", plan)
        self.assertNotIn("MIMALLOC_PURGE_DELAY", plan)


class FailClosedTest(unittest.TestCase):
    """Anything but empty or a non-negative integer refuses the launch."""

    def test_rejects_non_integer_values(self):
        for knob in KNOBS:
            for bad in ("abc", "-5", "1.5", " 0", "0x10", "10s"):
                with self.subTest(knob=knob, value=bad):
                    rc, out = _dry_run({knob: bad})
                    self.assertNotEqual(rc, 0, out)
                    self.assertIn(f"{knob} must be a non-negative integer", out)

    def test_guard_path_stays_immune(self):
        # scripts/guard.sh sources env.sh with RECIPE_ENV_NO_GATES=1: a bad
        # knob must not abort the guard's own stop path (an `exit` in a
        # sourced file would otherwise kill it silently).
        p = subprocess.run(
            ["bash", "-c",
             'RECIPE_ENV_NO_GATES=1 . scripts/env.sh >/dev/null && echo survived'],
            cwd=REPO, env=_env({"MIMALLOC_PURGE_DELAY": "abc"}),
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("survived", p.stdout)


class DockerRunTest(unittest.TestCase):
    """The plan line and the real docker run must not drift apart."""

    def test_docker_run_receives_the_env_array(self):
        with open(SERVE, encoding="utf-8") as handle:
            text = handle.read()
        run = text.split('sudo -n docker run -d --name "$CONTAINER"', 1)
        self.assertEqual(len(run), 2, "serve.sh's docker run block moved")
        self.assertIn('"${MIMALLOC_ENV[@]}"', run[1].split('--entrypoint', 1)[0])


if __name__ == "__main__":
    unittest.main()