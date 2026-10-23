# bench — decode measurements for the MiniMax-M3.1-Flash serve

`bench.py <label> [--reps N] [--tokens N]` streams fixed-length decodes
(`ignore_eos`) over three prompt classes (prose / code / structured) and
diffs the engine's own spec-decode counters around each request, so the
numbers come from `/metrics`, not from the client.

Output: one JSONL row per request (`tok_s`, `ttft_s`, `decode_s`, `steps`,
`ms_step`, `tok_step`, `accept_rate`) and one final
`{"label": ..., "SUMMARY": {...}}` object with per-class medians/means.

Reading the SUMMARY:

- `tok_s_med` — what the user sees (completion tokens / decode wall).
- `ttft_s_med` — prefill speed at this context length.
- `tok_step_mean` — speculative-decode acceptance in tokens per verify step;
  1.0 with `accept_rate_med` ~ 0 means the DSpark draft is fully rejected.
  On the real pack these are the primary health signals.
- `ms_step_mean` — per-step cost.

Protocol: run only after `scripts/smoke.sh` passes and only while the
memory guard is armed. Keep every JSONL run under `runs/` as the
measurement record; the README's measured-results tables are filled from
these SUMMARY objects, never by hand.

The measured numbers in the top-level README come from Victor's runs of this
tool on the pinned stack; see the README's "Measured results" for the
per-profile medians and their provenance.
