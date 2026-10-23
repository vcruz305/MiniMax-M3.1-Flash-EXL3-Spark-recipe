#!/usr/bin/env python3
"""Long-context check + sampling-matched bench for the MiniMax-M3.1-Flash
EXL3 serve on ONE DGX Spark. Runs on the Spark host (reads /proc/meminfo).

Per --sizes prompt size (approximated by characters/3.5; the server's real
prompt_tokens is read back from the response usage):

  (a) needle test: a random 6-digit code is planted at ~40% depth of a
      long prompt; the model is asked for it; PASS when the reply (content
      OR reasoning field) contains the code.
  (b) TTFT and prefill tok/s via streaming (prompt_tokens / TTFT).
  (c) decode tok/s, and (when /metrics exposes the vllm spec-decode
      counters) accepted/drafted for that request.
  (d) host MemAvailable sampled every 0.5 s during each request; the
      minimum is reported.

--short-bench mode: N=3 short prose/code prompts, 256 output tokens each,
reporting tok/s, tok/step and draft acceptance.

Sampling is the pack's recommended temperature 1.0 / top_p 0.95 on every
request, and every prompt starts with a random nonce so the prefix cache
never helps. Sizes that exceed the server's max_model_len (read from
/v1/models) are refused up front — never mid-request.

Output: one JSON file (--out) plus a human summary on stdout.

  python3 scripts/longctx.py --sizes 8000,15000 --out runs/longctx.json
  python3 scripts/longctx.py --short-bench --out runs/short.json
"""
import argparse
import json
import os
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request

CHARS_PER_TOKEN = 3.5          # token estimate: chars = tokens x 3.5
NEEDLE_DEPTH_FRAC = 0.40       # needle planted at ~40% of the prompt
MEM_SAMPLE_S = 0.5
SAMPLING = {"temperature": 1.0, "top_p": 0.95}   # the pack's recommendation
NEEDLE_MAX_TOKENS = 512
PROBE_MAX_TOKENS = 256
SHORT_BENCH_TOKENS = 256

_PORT = os.environ.get("API_PORT", "8000")
BASE = f"http://127.0.0.1:{_PORT}"
MODEL = os.environ.get("SERVED_MODEL_NAME", "MiniMax-M3.1-Flash-2.15bpw")

SPEC_KEYS = ("vllm:spec_decode_num_drafts",
             "vllm:spec_decode_num_accepted_tokens",
             "vllm:spec_decode_num_draft_tokens")

# Filler for the long prompts: our own text, repeated to the requested size.
FILLER = """
The northern line was surveyed in the last year of the old empire, and the
engineers who drove it across the moor left a series of benchmark stones that
the Ordnance survey still records. Each stone carries a number, and the
numbers run in a sequence that skips every seventh entry, because the seventh
stones were never cut. A walker who counts the stones and expects a clean
sequence will therefore conclude that stones have been removed, when in fact
the gap is original. This is a useful parable about measuring systems: the
instrument was never promised to be uniform, and the absence of an expected
entry is not evidence of loss.
"""
FILLER += """
The harbour commission met once a month, and its minutes survive in a bound
volume that smells of salt. Most of the business is dull: dues, dredging,
the price of whale oil for the lamps. But in March the clerk records a
dispute about a warehouse key. The key, it emerges, opened a room that had
been bricked up in the previous decade, and the dispute turns on who holds
the authority to unbrick a door. The commission resolves nothing and moves
to the next item, which is the price of rope. Nine years later the room is
opened by a storm, and the inventory found inside answers a question that
the archive had silently stopped asking.
"""
FILLER += """
A compiler writer once listed the three lies of optimization. The first lie is
that the machine is uniform: it is not, and the cache hierarchy makes every
access a negotiation. The second lie is that the program is static: it is
not, and the branch predictor learns the program faster than the programmer
does. The third lie is that the measurement is the behavior: it is not, and a
benchmark that improves a number can degrade every program that resembles the
benchmark too little. The writer's conclusion was not fatalistic; the point
of naming the lies is to know where the accounting has to be done by hand.
"""
FILLER += """
The seed bank keeps its accessions at minus twenty degrees, and every envelope
carries a barcode that resolves to a latitude, a season and a name. The
curators know that a meaningful fraction of the names are wrong: collectors
write down what they were told, transliterate it later, and file it under a
spelling that no search will find again. The bank therefore indexes by
geometry and time, not by words, and when a breeder asks for material from a
valley whose name has changed spelling four times in a century, the query
that succeeds is the polygon of the valley, not its name.
"""
FILLER += """
The old telephone exchange printed its routing tables every night, and the
printout was long enough that the engineers measured it by weight. A routing
error in the printed table could route an entire city's traffic into a single
trunk for a whole day, so the table was checked by two clerks who read the
columns aloud to each other in a room without windows. The redundancy was not
ceremonial; the error rate of a single reader was known and unacceptable, and
the error rate of two readers who disagree aloud was low enough to ship. The
room was removed in the same decade as the mechanical exchange, and the last
chief clerk insisted that the removal was the first routing error nobody had
checked.
"""
FILLER += """
The flood of 1947 is remembered in the height of the water, marked by a plate
on the mill wall, but the flood that matters to the current design is the one
of 1968, which was lower and wider and which the plate does not record. The
difference between the two floods is the state of the upstream catchment: by
1968 the drains had been straightened, and the water arrived sooner and
spread further. Any structure designed against the 1947 plate is designed
against a flood that can no longer happen in the same shape, and the
engineers who point this out are not contradicting history but reading it.
"""
FILLER += """
The observatory kept its plate collection in folders ordered by night, and the
night folders were ordered by the assistant's mood. A researcher who wants a
specific exposure must therefore know not only the date but the assistant,
because the misfiled nights were reboxed twice, once correctly and once
conveniently, and the convenient order is the one that survives. The archive
is not corrupted; it is annotated by the shape of the hands that moved it,
and the annotation is illegible from the outside. Every long archive accrues
such a shape. The honest finding is that the shape can be mapped, and the map
is worth more than a reordering.
"""
FILLER += """
A queueing theorist was asked how long the new toll plaza would delay traffic,
and answered with a question: how long is the red light? The plaza's delay
is not a property of the plaza; it is a property of the distribution of
arrivals against the distribution of service, and the red light is a single
parameter that moves both. The designers had a target for the average delay,
and the theorist refused to certify the average, because the average is
dominated by the best mornings and experienced by nobody. The quantity that
matters is the tail: the morning that happens once a quarter and decides
whether the road keeps its permit.
"""
FILLER += """
def fold_scores(scores, weights):
    # stable fold: never let a single component dominate the sum
    total = 0.0
    for s, w in zip(scores, weights):
        if not 0.0 <= s <= 1.0:
            raise ValueError("score out of range")
        total += s * w
    return total / sum(weights)

def rank(items, key):
    # decorate-sort-undecorate, stable on ties by original index
    deco = [(key(it), i, it) for i, it in enumerate(items)]
    deco.sort(key=lambda t: (t[0], t[1]))
    return [it for _, _, it in deco]
"""
FILLER += """
class RingBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.buf = [None] * cap
        self.head = 0
        self.n = 0

    def push(self, v):
        self.buf[self.head] = v
        self.head = (self.head + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def window(self):
        if self.n == 0:
            return []
        return [self.buf[(self.head - i - 1) % self.cap] for i in range(self.n)]
"""


# ---------------------------------------------------------------------------
# pure functions (unit-tested in scripts/test_longctx.py)
# ---------------------------------------------------------------------------

def chars_for_tokens(tokens):
    """Approximate character budget for a token count (chars/3.5)."""
    return int(round(tokens * CHARS_PER_TOKEN))


def mem_available_kib(meminfo_text):
    """MemAvailable in KiB parsed from /proc/meminfo text; None if absent."""
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def size_refusal(size_tokens, max_model_len):
    """Refusal message when a size cannot be served; None when it can."""
    if not isinstance(size_tokens, int) or size_tokens <= 0:
        return f"size {size_tokens!r} is not a positive integer"
    if size_tokens > max_model_len:
        return (f"size {size_tokens} exceeds the server's max_model_len "
                f"{max_model_len}; refusing before any request is sent")
    return None


def build_prompt(filler, nonce, target_chars):
    """A prompt of ~target_chars characters: a nonce first line (so the
    prefix cache never helps) followed by the filler repeated and truncated."""
    prefix = f"context nonce {nonce}\n"
    if target_chars <= len(prefix):  # degenerate budget: truncate the nonce line
        return prefix[:target_chars]
    unit = filler if filler.endswith("\n") else filler + "\n"
    if not unit:
        unit = "\n"
    reps = -(-target_chars // len(unit))  # ceil division
    body = (unit * reps)[: target_chars - len(prefix)]
    return prefix + body


def needle_line(code):
    return (f"The maintenance code for sluice gate 7 is {code}. "
            f"Keep this code secret.")


def place_needle(text, line, frac=NEEDLE_DEPTH_FRAC):
    """Insert <line> on its own line at ~frac depth of <text> (pure)."""
    off = int(len(text) * frac)
    nl = text.find("\n", off)
    at = nl + 1 if nl >= 0 else off
    return text[:at] + line + "\n" + text[at:]


def spec_delta(before, after):
    """Per-request spec-decode deltas from two /metrics scrapes; None when
    the counters are not exposed or the scrapes failed."""
    if not before or not after:
        return None
    out = {}
    for k in SPEC_KEYS:
        if k not in before or k not in after:
            return None
        out[k] = after[k] - before[k]
    return out


# ---------------------------------------------------------------------------
# server side
# ---------------------------------------------------------------------------

def http_json(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "replace"))


class MemSampler(threading.Thread):
    """Samples host MemAvailable every MEM_SAMPLE_S seconds; keeps the min."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.samples = []
        self.min_gib = None

    def _sample(self):
        try:
            with open("/proc/meminfo") as f:
                kib = mem_available_kib(f.read())
        except OSError:
            kib = None
        if kib is not None:
            self.samples.append(kib)
            gib = kib / 1048576.0
            if self.min_gib is None or gib < self.min_gib:
                self.min_gib = gib

    def run(self):
        while not self.stop_evt.is_set():
            self._sample()
            self.stop_evt.wait(MEM_SAMPLE_S)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=5)


def stream(path, payload):
    """One streamed request. Returns (result, error): result carries wall
    time, TTFT, the accumulated per-field text, usage and finish_reason."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    usage = None
    finish = None
    fields = {}
    ntok = 0
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    j = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                ch = (j.get("choices") or [{}])[0]
                delta = ch.get("delta") or ch  # completions: .text, chat: .delta
                for f in ("text", "content", "reasoning", "reasoning_content"):
                    if delta.get(f):
                        fields[f] = fields.get(f, "") + delta[f]
                        ntok += 1
                        if ttft is None:
                            ttft = time.monotonic() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        wall = time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
    except urllib.error.URLError as e:
        return None, f"connection error: {e.reason!r}"
    return {"wall_s": wall, "ttft_s": ttft, "fields": fields,
            "chunks": ntok, "usage": usage, "finish_reason": finish}, None


def scrape_spec():
    """vllm spec-decode counters from /metrics (summed over labels)."""
    out = {}
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=60) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                m = re.match(r"^(vllm:spec_decode_num_\w+?)_total\{[^}]*\}\s+(\S+)$",
                             line)
                if m:
                    out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    except (urllib.error.URLError, OSError):
        return None
    return out or None


def get_max_model_len():
    try:
        status, j = http_json("/v1/models", timeout=120)
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"longctx: {BASE}/v1/models is not answering: {e!r}")
    data = (j or {}).get("data") or []
    if status != 200 or not data:
        raise SystemExit(f"longctx: /v1/models did not answer (HTTP {status})")
    mml = data[0].get("max_model_len")
    if not isinstance(mml, int) or mml <= 0:
        raise SystemExit("longctx: /v1/models does not expose max_model_len; "
                         "refusing to run blind (fail closed)")
    return mml


def parse_sizes(text):
    sizes = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            sizes.append(int(part))
        except ValueError:
            raise SystemExit(f"longctx: bad size {part!r} in --sizes")
    if not sizes:
        raise SystemExit("longctx: --sizes is empty")
    return sizes


def new_nonce():
    return f"{secrets.token_hex(8)}-{int(time.time() * 1000) % 1000000}"


def base_payload(max_tokens):
    p = {"model": MODEL, "max_tokens": max_tokens,
         "stream": True, "stream_options": {"include_usage": True}}
    p.update(SAMPLING)
    return p


def run_size(size, mml):
    """All measurements for one prompt size. Two requests:
    1. needle: the planted-code question (chat) -> needle verdict, TTFT,
       prefill tok/s, MemAvailable minimum.
    2. probe: same-size raw completion, ignore_eos -> decode tok/s and the
       per-request spec-decode accepted/drafted deltas, MemAvailable minimum.
    """
    chars = chars_for_tokens(size)
    out = {"size_tokens": size, "prompt_chars": chars}

    # ---- request 1: needle --------------------------------------------------
    nonce = new_nonce()
    code = f"{secrets.randbelow(900000) + 100000}"
    prompt = build_prompt(FILLER, nonce, chars)
    prompt = place_needle(prompt, needle_line(code))
    prompt += ("\n\nWhat is the maintenance code for sluice gate 7? "
               "Reply with the six digits only.")
    sampler = MemSampler()
    sampler.start()
    res, err = stream("/v1/chat/completions",
                      base_payload(NEEDLE_MAX_TOKENS)
                      | {"messages": [{"role": "user", "content": prompt}]})
    sampler.stop()
    out["needle"] = {"code": code}
    if err:
        out["needle"]["error"] = err
        out["needle"]["found"] = None
        print(f"  needle: REQUEST FAILED: {err}")
    else:
        fields = res["fields"]
        reasoning = (fields.get("reasoning", "") +
                     fields.get("reasoning_content", ""))
        content = fields.get("content", "")
        if code in reasoning:
            where = "reasoning"
        elif code in content:
            where = "content"
        else:
            where = None
        out["needle"]["found"] = where is not None
        out["needle"]["field"] = where
        out["needle"]["finish_reason"] = res["finish_reason"]
        ptoks = ((res["usage"] or {}).get("prompt_tokens"))
        out["server_prompt_tokens"] = ptoks
        out["ttft_s"] = round(res["ttft_s"], 3) if res["ttft_s"] else None
        out["prefill_tok_s"] = (round(ptoks / res["ttft_s"], 1)
                                if res["ttft_s"] and ptoks else None)
        print(f"  needle: {'PASS' if where is not None else 'FAIL'} "
              f"(field={where}) server_prompt_tokens={ptoks} "
              f"ttft={out['ttft_s']}s prefill={out['prefill_tok_s']} tok/s")
    out["mem_min_needle_gib"] = round(sampler.min_gib, 3) if sampler.min_gib else None

    # ---- request 2: decode probe --------------------------------------------
    nonce = new_nonce()
    probe_prompt = build_prompt(FILLER, nonce, chars)
    spec_before = scrape_spec()
    sampler = MemSampler()
    sampler.start()
    res, err = stream("/v1/completions",
                      base_payload(PROBE_MAX_TOKENS)
                      | {"prompt": probe_prompt, "ignore_eos": True})
    sampler.stop()
    out["decode"] = {}
    if err:
        out["decode"]["error"] = err
        print(f"  probe: REQUEST FAILED: {err}")
    else:
        ct = (res["usage"] or {}).get("completion_tokens") or res["chunks"]
        decode_s = res["wall_s"] - (res["ttft_s"] or 0.0)
        out["decode"]["completion_tokens"] = ct
        out["decode"]["decode_s"] = round(decode_s, 3)
        out["decode"]["tok_s"] = (round((ct - 1) / decode_s, 2)
                                 if decode_s > 0 else None)
        d = spec_delta(spec_before, scrape_spec())
        out["decode"]["spec_decode"] = d
        if d:
            out["decode"]["tok_per_step"] = (round(1 + d[SPEC_KEYS[1]] / d[SPEC_KEYS[0]], 3)
                                            if d[SPEC_KEYS[0]] else None)
            out["decode"]["acceptance"] = (round(d[SPEC_KEYS[1]] / d[SPEC_KEYS[2]], 3)
                                           if d[SPEC_KEYS[2]] else None)
        print(f"  probe : decode={out['decode']['tok_s']} tok/s "
              f"tok/step={out['decode'].get('tok_per_step')} "
              f"acceptance={out['decode'].get('acceptance')}")
    out["mem_min_probe_gib"] = round(sampler.min_gib, 3) if sampler.min_gib else None
    return out


SHORT_PROMPTS = [
    ("prose",
     "Write a paragraph about why a lighthouse keeper's log is harder to "
     "falsify than a bank ledger, and continue the argument for a full page."),
    ("code",
     "import socket\nimport selectors\n\n# Complete this: a single-threaded "
     "select loop that accepts connections, never blocks on a slow client,\n"
     "# and closes everything cleanly on shutdown. Continue the code."),
    ("structured",
     '{"shipment": {"id": "S-4471", "route": ["ROT", "HAM", "NYC"], '
     '"tonnage": 18.4}}\n# Validate this shipment against the rules: no '
     "transfer at HAM without a bond, tonnage over 18 needs a pilot.\n"
     "# Continue with the full audit trail."),
]


def run_short_bench():
    rows = []
    for name, prompt in SHORT_PROMPTS:
        nonce = new_nonce()
        spec_before = scrape_spec()
        sampler = MemSampler()
        sampler.start()
        res, err = stream("/v1/completions",
                          base_payload(SHORT_BENCH_TOKENS)
                          | {"prompt": f"nonce {nonce}\n" + prompt,
                             "ignore_eos": True})
        sampler.stop()
        row = {"class": name}
        if err:
            row["error"] = err
            print(f"  {name}: REQUEST FAILED: {err}")
        else:
            ct = (res["usage"] or {}).get("completion_tokens") or res["chunks"]
            decode_s = res["wall_s"] - (res["ttft_s"] or 0.0)
            row["completion_tokens"] = ct
            row["tok_s"] = round((ct - 1) / decode_s, 2) if decode_s > 0 else None
            d = spec_delta(spec_before, scrape_spec())
            row["spec_decode"] = d
            if d:
                row["tok_per_step"] = (round(1 + d[SPEC_KEYS[1]] / d[SPEC_KEYS[0]], 3)
                                      if d[SPEC_KEYS[0]] else None)
                row["acceptance"] = (round(d[SPEC_KEYS[1]] / d[SPEC_KEYS[2]], 3)
                                    if d[SPEC_KEYS[2]] else None)
            print(f"  {name}: tok/s={row['tok_s']} "
                  f"tok/step={row.get('tok_per_step')} "
                  f"acceptance={row.get('acceptance')}")
        row["mem_min_gib"] = round(sampler.min_gib, 3) if sampler.min_gib else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Long-context check + sampling-matched bench "
                    "(temperature 1.0 / top_p 0.95; nonce-prefixed prompts).")
    ap.add_argument("--sizes", default="8000,15000",
                    help="approximate prompt sizes in tokens (default 8000,15000)")
    ap.add_argument("--out", default="longctx.json",
                    help="output JSON file (default longctx.json)")
    ap.add_argument("--short-bench", action="store_true",
                    help="also run the N=3 short prose/code decode bench "
                         "(256 output tokens each)")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    mml = get_max_model_len()
    print(f"longctx: base={BASE} model={MODEL} max_model_len={mml} "
          f"sampling={SAMPLING} sizes={sizes}")
    refusals = [(s, size_refusal(s, mml)) for s in sizes]
    bad = [(s, r) for s, r in refusals if r]
    if bad:
        for s, r in bad:
            print(f"REFUSED: {r}", file=sys.stderr)
        print(f"longctx: refusing to run: {len(bad)} of {len(sizes)} sizes "
              "exceed max_model_len (or are invalid); no request was sent.",
              file=sys.stderr)
        sys.exit(2)

    results = {"meta": {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "base": BASE, "model": MODEL, "max_model_len": mml,
                       "sizes": sizes, "sampling": SAMPLING,
                       "chars_per_token": CHARS_PER_TOKEN,
                       "mem_sample_s": MEM_SAMPLE_S},
               "long": [], "short_bench": None}
    needle_failed = False
    for s in sizes:
        print(f"== size {s} tokens (~{chars_for_tokens(s)} chars) ==")
        row = run_size(s, mml)
        results["long"].append(row)
        if row["needle"].get("found") is False:
            needle_failed = True
    if args.short_bench:
        print("== short bench (N=3, 256 output tokens) ==")
        results["short_bench"] = run_short_bench()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"longctx: results written to {args.out}")

    # human summary
    print("\n== summary ==")
    for row in results["long"]:
        n = row["needle"]
        print(f"size {row['size_tokens']}: needle "
              f"{'PASS' if n.get('found') else ('FAIL' if n.get('found') is False else 'n/a')}"
              f" (field={n.get('field')}) server_prompt_tokens="
              f"{row.get('server_prompt_tokens')} ttft={row.get('ttft_s')}s "
              f"prefill={row.get('prefill_tok_s')} tok/s "
              f"decode={row.get('decode', {}).get('tok_s')} tok/s "
              f"tok/step={row.get('decode', {}).get('tok_per_step')} "
              f"acceptance={row.get('decode', {}).get('acceptance')} "
              f"mem_min={row.get('mem_min_needle_gib')}/"
              f"{row.get('mem_min_probe_gib')} GiB")
    if results["short_bench"]:
        for row in results["short_bench"]:
            print(f"short {row['class']}: tok/s={row.get('tok_s')} "
                  f"tok/step={row.get('tok_per_step')} "
                  f"acceptance={row.get('acceptance')} "
                  f"mem_min={row.get('mem_min_gib')} GiB")
    if needle_failed:
        print("longctx: at least one needle test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
