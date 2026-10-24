# MiniMax-M3.1-Flash EXL3 (SAGE 2.15 bpw) on ONE NVIDIA DGX Spark

**Model on Hugging Face:** [vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw) (gated: request access on the model page)

> **Work in progress.** This recipe is under active development. The `q4-32k` profile is measured and is what the author serves today; larger contexts, q4 prefill speed and the tool-calling evaluation are still being worked on, and pins/profiles may change. The two-Spark 4.26 bpw path is not validated yet — PRs welcome.

A pinned, fail-closed serving recipe for
[`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw)
on **one** DGX Spark (GB10, 128 GB unified memory, linux/arm64), with vLLM
and the [vllm-exl3](https://github.com/vcruz305/vllm-exl3) plugin. Every
default in this repo is a measured boot configuration; the README's numbers
are Victor's measurements on his Spark, each labeled with the tool/log it
came from.

**Pins (all fail-closed, all verified by scripts at build/serve time):**

| what | pin | where it is read |
|---|---|---|
| model pack | [`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw) @ `99bf3a57af5d07245ea9abb5426e7dce6af6356b` (byte-verified via `scripts/SHA256SUMS` + `scripts/PACK-BLOB-SUMS`; see [Pack revision](#pack-revision)) | `scripts/env.sh` |
| base engine image | `vllm/vllm-openai:minimax-m31-flash` @ digest `sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c` | `scripts/env.sh` (`BASE_IMAGE_DIGEST`), `Dockerfile` |
| exllamav3 fork | [`vcruz305/exllamav3`](https://github.com/vcruz305/exllamav3) branch `feat/moe-act3-batched` @ `fdd77361971144e8d175e09e7b50b7100739fd95` | `scripts/env.sh` (`EXL3_REQUIRED_REV`) |
| vllm-exl3 plugin | [`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3) branch `feat/minimax-m3-exl3` @ `688220ee76d3fa49627d904082550af9458c3cd2` | `scripts/env.sh` (`PLUGIN_REQUIRED_REV`) |

Why the two fork pins are not optional: the exllamav3 pin carries the
act-3 (clamped SwiGLU-OAI) case in the batched MoE kernels — a tree without
it decodes garbage from every MiniMax MoE layer (measured cosine ~0.10
without the fix, 1.0 with it). The plugin pin carries the true-bit-width
per-expert K arrays of the unified mixed-K dispatch — upstream plugin main
builds them in half-bit units, which is wrong for every mixed-K pack in
unified mode — plus the indexer MXFP8 retention, the q4 main-KV pool, the
UMA load hygiene and the C++20 build fix the plugin needs on torch 2.14.
`scripts/build_image.sh` refuses to build and `scripts/serve.sh` refuses to
serve anything below either pin.

## Status

| Sparks | folder | engine | profile | status |
|---|---|---|---|---|
| 1x DGX Spark (GB10, 128 GB unified) | this repo | vLLM (pinned upstream image) + vllm-exl3 plugin | `q4-32k` (32K context, q4 main KV) | **measured** — see [Measured results](#measured-results) |
| 2x DGX Spark | — | — | SAGE 4.26 bpw pack ([vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-4.26bpw](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-4.26bpw)) | **not validated yet — help wanted** |

## Quick start

Run top-down on the Spark itself (aarch64, Docker + NVIDIA container toolkit,
passwordless `sudo docker`, ~120 GB free disk for the pack).

1. **Tooling.** Python 3.12, `git`, `sha256sum`, curl; the HF CLI with hf_xet:

   ```bash
   pip install -U "huggingface_hub[cli]" hf_xet
   hf auth login   # the pack repo is GATED (manual approval): request
                   # access on the model page first, then log in here
   ```

   No token value is ever stored in this repo or passed by its scripts:
   `download_and_verify.sh` never handles a token value.

2. **Clone the two pinned runtime trees** (build_image.sh builds from exactly
   these checkouts and labels the image with their HEADs):

   ```bash
   mkdir -p ~/minimax-m31-src   # or set SRC_DIR=... before every step
   git clone https://github.com/vcruz305/exllamav3.git ~/minimax-m31-src/exllamav3
   git -C ~/minimax-m31-src/exllamav3 checkout fdd77361971144e8d175e09e7b50b7100739fd95
   git clone https://github.com/vcruz305/vllm-exl3.git ~/minimax-m31-src/vllm-exl3
   git -C ~/minimax-m31-src/vllm-exl3 checkout 688220ee76d3fa49627d904082550af9458c3cd2
   ```

3. **Download and verify the pack** (~119 GB; hashes every file and refuses
   on any mismatch — see [Pack revision](#pack-revision)):

   ```bash
   bash scripts/download_and_verify.sh    # default target: ~/models/
   ```

4. **Build the serving image** (upstream engine image pinned by digest + both
   pinned trees compiled in; the image is labelled with the pins and
   serve.sh re-checks the labels):

   ```bash
   sudo docker pull vllm/vllm-openai@sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c
   bash scripts/build_image.sh
   ```

5. **Serve.** The default profile is `q4-32k` (32K context, q4 main KV);
   `PROFILE=` selects any `configs/<name>.env`:

   ```bash
   bash scripts/serve.sh                      # PROFILE=q4-32k by default
   PROFILE=fp8-16k bash scripts/serve.sh      # 16K, fp8 KV everywhere
   ```

   The serve is standalone: it waits for `/v1/models` (`SERVE_WAIT=1`), then
   runs the warm-up (one short chat request and one ~8K-token prompt,
   `max_tokens` 32) and only then prints the READY line. Each warm-up
   request's wall time and MemAvailable before/after are logged to
   `<run dir>/warmup.log`; a warm-up failure fails the start. The memory
   guard is armed before the container starts and is not optional.

6. **Smoke test** (chat, tool call, reasoning separation, and the mixed-K
   dispatch-mode gate):

   ```bash
   bash scripts/smoke.sh
   ```

7. **Stop** (container down, host sysctls restored, own guard ended):

   ```bash
   bash scripts/guard.sh stop
   ```

Optional tools kept in the repo: `scripts/longctx.py` (needle tests at
context length, prefill tok/s and TTFT per length), `bench/bench.py`
(decode measurements; see `bench/README.md`), `scripts/guard.sh status`
(one snapshot).

## Profiles

`PROFILE=` selects `configs/<name>.env`. Every profile is byte-for-byte the
flag set of a measured boot; a profile can only tighten the measured
defaults, never widen them. The guard floor each profile sets is part of
the measurement (see [Known limits](#known-limits)).

| profile | context | main KV | indexer KV | draft KV | guard floor | measured |
|---|---|---|---|---|---|---|
| `q4-32k` (default) | 32,768 | q4 (nvfp4 layout) | fp8 | fp8 | 0.2 GiB (explicit opt-in) | yes |
| `fp8-24k` | 24,576 | fp8 | fp8 | fp8 | 0.2 GiB (explicit opt-in) | yes |
| `fp8-16k` | 16,384 | fp8 | fp8 | fp8 | 0.5 GiB (public default) | yes |

48K q4 was tried and does **not** fit yet: the pool itself allocates (50,048
tokens of KV), but the first request after boot pushed host MemAvailable to
0 and the guard killed the container (receipt `boot22`: warm-up, smoke,
long-context and bench all failed). 48K is not a profile and is not
closeable by any knob in this repo.

## Measured results

All numbers below are from Victor's runs of this exact recipe (image
`m31-flash-exl3-spark:dev-fdd7736-a5e91ed` built by this Dockerfile, with
the plugin overlaid at the pinned rev) on his DGX Spark on 2026-10-23.
Provenance for every table is given per row. The receipts themselves are not
in this repo; each label names the tool/log the number came from.

### Boot and memory

| metric | fp8-16k | fp8-24k | q4-32k (default) | source |
|---|---|---|---|---|
| time to `/v1/models` (s) | 171 | 174 | 174 | run `result` (`boot_secs`) |
| KV tokens allocated (1.02x limit) | 16,640 | 24,960 | 33,408 | `container.log` (engine KV-cache line) |
| MemAvailable at ready (GiB) | 1.55 | 1.00 | 1.09 | `mem-ready` / run `result` |
| MemAvailable after warm-up (GiB) | 1.17 | 0.59 | 1.34 | `warmup.out` |
| MemAvailable minimum under load (GiB) | 1.03 | 0.44 | 0.52 | `guard.log` (0.05 s samples) |
| guard floor the run used (MiB) | 204 | 204 | 204 | `serve.out` |

The warm-up requests cost 1.8-1.9 s (short chat) and 24.1-29.4 s (~8K
prompt) per boot (`warmup.out`).

The indexer MXFP8 retention (`INDEXER_MXFP8=1`, default) was A/B-measured
on the same 16K fp8 boot: identical needle results and prefill speed
(379.2 vs 378.4 tok/s), and +0.4-0.5 GiB free memory at every stage
(steady 1.03-1.12 GiB with the knob vs 0.57-0.67 GiB without) — `longctx.log`
of both runs.

### Long-context needles (all PASS, answer in `content`)

Prefill and TTFT per tested length (`scripts/longctx.py`, `longctx.log`):

| profile | prompt tokens | prefill (tok/s) | TTFT (s) | decode probe (tok/s) |
|---|---|---|---|---|
| fp8-16k | 6,530 | 378.4 | 17.257 | 20.24 |
| fp8-16k | 12,070 | 384.3 | 31.408 | 22.24 |
| fp8-24k | 6,533 | 379.0 | 17.238 | 22.08 |
| fp8-24k | 12,075 | 381.5 | 31.652 | 22.35 |
| fp8-24k | 18,439 | 377.3 | 48.865 | 14.72 |
| q4-32k | 6,532 | 338.1 | 19.32 | 18.97 |
| q4-32k | 12,071 | 324.3 | 37.217 | 20.01 |
| q4-32k | 24,798 | 276.2 | 89.769 | 20.77 |

### Decode (`bench/bench.py`, medians over 3 prompts per class, 512 tokens each)

| profile | class | tok/s (median) | ms/step (mean) | acceptance (median) |
|---|---|---|---|---|
| fp8-16k | prose | 23.08 | 130.9 | 0.714 |
| fp8-16k | code | 19.93 | 126.9 | 0.533 |
| fp8-16k | structured | 28.65 | 129.1 | 0.936 |
| fp8-24k | prose | 25.29 | 130.8 | 0.756 |
| fp8-24k | code | 23.5 | 121.0 | 0.594 |
| fp8-24k | structured | 28.64 | 129.5 | 0.955 |
| q4-32k | prose | 21.15 | 133.1 | 0.651 |
| q4-32k | code | 29.67 | 129.6 | 0.938 |
| q4-32k | structured | 29.79 | 130.6 | 0.947 |

Sampling note: every number above was measured at the pack's own
`generation_config.json` sampling — temperature 1.0, top_p 0.95 — which the
recipe leaves in force (the serve passes no temperature/top_p override).
Acceptance varies strongly with prompt class; the per-step cost does not
(~121-133 ms/step everywhere).

### DSpark speculative decoding (k=4)

The draft ships inside the pack (`dspark/` subfolder); the engine resolves
it from the model path and no draft model path is passed anywhere. The
measured configuration is `num_speculative_tokens=3`: one verify step
covers up to 4 tokens (k=4), which is what the CUDA-graph capture sizes
1-4 pin. On GB10 the only draft pair that boots and passes is
`TRITON_ATTN` + fp8 draft KV ("auto" picks FLASHINFER, which has no block
size 128; FA fp8 draft KV is FA3/Hopper-only) — every profile pins that
pair, and the q4 gate refuses to serve without it. Measured acceptance and
effective decode rates are the bench table above (0.53-0.98 acceptance by
class, 19-30 tok/s effective); a no-draft serve was never measured on this
pack.

## Pack revision

The pack was measured at revision
`e2682d076e220e71e2a7ea7b9885e70390c27698`. The Hub repository's history
was later replaced by a commit that only removed a stray temp file; the
current head `99bf3a57af5d07245ea9abb5426e7dce6af6356b` — which is what
`MODEL_REV` pins — carries byte-identical weights: the same git blob oid and
size for every one of the 40 files, so the pinned tree and the measured
tree are the same bytes. The bytes, not the commit id, are what this
recipe pins:

- `scripts/SHA256SUMS` — sha256 of every LFS file (the 16 weight shards and
  the DSpark draft weight) at the measured revision;
- `scripts/PACK-BLOB-SUMS` — git blob sha1 of every non-LFS file.

`scripts/SHA256SUMS` + `scripts/PACK-BLOB-SUMS` remain the byte authority:
`scripts/download_and_verify.sh` downloads `MODEL_REV` and refuses unless
every file matches those hashes; it never falls back to a different tree on
its own. If the pinned revision ever stops resolving again, do not serve a
different tree — the hash check is the only acceptance criterion, and if no
commit carries the measured bytes, this recipe has no pack to serve.

## Cards tested on

| card | VRAM | status |
|---|---|---|
| NVIDIA DGX Spark (GB10, sm_121, linux/arm64) | 128 GB unified | measured (this README) |

No other card, host OS or vLLM build has been verified with this recipe.

## Quants

The EXL3 packs of this model live in
[`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw);
the table mirrors the HF model README.

| bpw | size | smallest card | top-1 vs original | mean KLD | p99 KLD | download |
|---|---|---|---|---|---|---|
| 2.15 (SAGE) | ~119 GB | DGX Spark (128 GB unified) | — | — | — | [HF](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw) |
| 4.26 (SAGE) | — | 2x DGX Spark | — | — | — | [HF](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-4.26bpw) |

top-1 = next-token agreement with the original checkpoint on a held-out set;
KLD = mean and 99th-percentile KL(reference ‖ pack) at those positions. No
held-out quality evaluation of this pack has been run, so those cells stay
em dashes rather than invented numbers; the evaluation text was not used
for calibration. Every number on this page is tied to one card, one pack
revision and one set of server settings.

## Two Sparks (not validated yet — help wanted)

The SAGE 4.26 bpw pack
([`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-4.26bpw`](https://huggingface.co/vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-4.26bpw),
public, not gated) is meant for **two** DGX Sparks.

The two-Spark part of this recipe has **not** been validated yet: there is no
measured boot and no launch config in this repo for it, and nothing above
(the pins, profiles, measured results, guard behaviour) was obtained on two
Sparks. Everything else in this repo is the one-Spark 2.15 bpw recipe.

The author is looking for help here and PRs are welcome — two-Spark launch
configs, fixes, and measured results together with the hardware and logs they
came from.

## Known limits

- **No vision weights in this pack.** MiniMax-M3.1-Flash is multimodal in
  its original release; this pack ships the language model and the DSpark
  draft only, and the serve passes `--language-model-only`. Vision requests
  are not served by this recipe.
- **262K does not fit on one Spark.** The model's full context window is
  262K; the largest context that boots and passes needles here is 32K
  (default profile). 48K was tried and does not fit yet (see
  [Profiles](#profiles)).
- **q4 prefill is slower and degrades with length.** The q4 main-KV pool
  buys the 32K context, at a prefill cost that falls from 338 to 276 tok/s
  as the prompt grows from 8K to 31K, versus ~377-384 tok/s flat on fp8
  (`longctx.log`).
- **69-scenario tool-calling eval not run.** The smoke test checks one
  well-formed tool call; the 69-scenario tool-calling evaluation has not
  been run on this stack.
- **The first request after boot costs a MAXLEN-independent memory
  transient.** ~0.33 GiB of host memory (plus ~0.18 GiB more at the first
  ~8K prefill) is consumed by the first request and recovered by
  malloc_trim only ~30 s later. This is why the warm-up must run before the
  READY line (it absorbs the transient into a logged request), and why the
  q4-32k and fp8-24k profiles set the 0.2 GiB guard floor opt-in
  (`GUARD_MEM_FLOOR_KIB=209716`, what the measured boots ran): at the
  public default 0.5 GiB floor, q4-32k's measured ready MemAvailable
  (1.09 GiB) minus that transient leaves ~0.2-0.5 GiB and the guard would
  kill the serve during the warm-up. The guard exists to keep the host
  alive; the floor is not a context-length knob.
- **One decode stream only.** `MAX_NUM_SEQS=1` on every profile: the MSA
  indexer tops out at one running sequence and the memory budget has no
  room for more.

## Credits

- [MiniMax](https://www.minimax.io) — MiniMax-M3.1-Flash, the base model
  and its DSpark draft.
- [vLLM](https://github.com/vllm-project/vllm) — the serving engine and the
  upstream MiniMax-M3.1-Flash support this image is pinned to.
- [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3) — the
  EXL3 runtime this pack is built for.
- [vllm-exl3](https://github.com/vcruz305/vllm-exl3) — the plugin that
  binds EXL3 packs into vLLM.

## License

MIT (see [LICENSE](LICENSE)). The model pack follows its own license on the
Hugging Face model page.
