#!/usr/bin/env python3
"""Post-start warm-up for the MiniMax-M3.1-Flash EXL3 serve (Spark host side).

Called by scripts/serve.sh after /v1/models answers (WARMUP=1, the default).
The first requests after every boot JIT-compile kernels
(_prepare_dflash_inputs_kernel, CuTeDSL MXFP8QuantizeSwizzledKernel), so
this script pays that cost where it is logged instead of in the first real
request:

  request 1  one short chat request (max_tokens 32)
  request 2  one ~8K-token prompt built from the recipe's own scripts
             concatenated, plus a random nonce (max_tokens 32)

Each request's wall time and MemAvailable (host /proc/meminfo) before/after
are appended to <run dir>/warmup.log. Any failure — connection error or a
non-200 reply — writes the HTTP error to that log and exits non-zero, which
fails the serve start.

Usage (from serve.sh):
  python3 scripts/warmup.py --run <run dir> --port <api port> \
      --model <served model name> --scripts-dir <recipe scripts dir>
"""
import argparse
import json
import os
import random
import secrets
import sys
import time
import urllib.error
import urllib.request

# ~8K tokens estimated at 3.5 characters/token (see scripts/longctx.py).
WARMUP_TOKENS = 8192
CHARS_PER_TOKEN = 3.5

# Recipe scripts used as filler for the long prompt: local, non-secret text.
FILLER_FILES = ("env.sh", "serve_inner.sh", "smoke.sh", "guard.sh")


def mem_available_kib(meminfo_text):
    """Parse MemAvailable (KiB) out of /proc/meminfo text; None if absent."""
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def read_mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            kib = mem_available_kib(f.read())
    except OSError:
        return None
    return None if kib is None else kib / 1048576.0


def build_long_prompt(scripts_dir, nonce):
    """Nonce line + the recipe's own scripts concatenated, truncated to the
    ~8K-token character budget. Pure with respect to its inputs."""
    chunks = []
    for name in FILLER_FILES:
        path = os.path.join(scripts_dir, name)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                chunks.append(f.read())
        except OSError as exc:
            raise SystemExit(f"warmup: cannot read filler {path}: {exc}")
    body = "\n".join(chunks)
    budget = int(round(WARMUP_TOKENS * CHARS_PER_TOKEN))
    prefix = f"warmup nonce {nonce}\n"
    unit = body + "\n"
    if len(unit) < budget:  # repeat until the budget is covered, then truncate
        body = unit * (-(-budget // len(unit)))
    return prefix + body[: max(budget - len(prefix), 0)]


def post_chat(base, model, messages, max_tokens):
    """POST /v1/chat/completions; returns (http_status, body_text)."""
    body = json.dumps({"model": model, "messages": messages,
                      "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        return 0, repr(e.reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir (log: <run>/warmup.log)")
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scripts-dir", required=True,
                    help="recipe scripts dir (filler source)")
    args = ap.parse_args()

    log_path = os.path.join(args.run, "warmup.log")
    base = f"http://127.0.0.1:{args.port}"
    nonce = f"{secrets.token_hex(8)}-{random.randint(100000, 999999)}"

    def log(line):
        with open(log_path, "a") as f:
            f.write(line + "\n")

    results = []
    requests = [
        ("short_chat",
         [{"role": "user",
           "content": "What is the capital of France? One word."}]),
        ("long_8k",
         [{"role": "user",
           "content": build_long_prompt(args.scripts_dir, nonce)
                      + "\nReply with the single word OK and nothing else."}]),
    ]
    ok = True
    for name, messages in requests:
        mem_before = read_mem_available_gib()
        t0 = time.monotonic()
        status, body = post_chat(base, args.model, messages, 32)
        wall = time.monotonic() - t0
        mem_after = read_mem_available_gib()
        prompt_chars = len(messages[0]["content"])
        if status != 200:
            ok = False
            snippet = " ".join(body.split())[:300]
            log(f"warmup {name} FAILED http={status} wall_s={wall:.1f} "
                f"prompt_chars={prompt_chars} "
                f"mem_before_gib={mem_before} mem_after_gib={mem_after} "
                f"error={snippet!r}")
            print(f"warmup {name}: FAILED http={status} (see {log_path})",
                  file=sys.stderr)
        else:
            log(f"warmup {name} ok http=200 wall_s={wall:.1f} "
                f"prompt_chars={prompt_chars} "
                f"mem_before_gib={mem_before:.3f} mem_after_gib={mem_after:.3f}")
            print(f"warmup {name}: ok wall={wall:.1f}s "
                  f"mem {mem_before:.3f}->{mem_after:.3f} GiB")
        results.append({"name": name, "http": status, "wall_s": round(wall, 1),
                        "prompt_chars": prompt_chars,
                        "mem_before_gib": mem_before,
                        "mem_after_gib": mem_after})
    log("warmup " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
