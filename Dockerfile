# Serving image for vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw on ONE DGX
# Spark (GB10, sm_121, linux/arm64). Built by scripts/build_image.sh, which
# must run on an aarch64 host with passwordless docker (the Spark).
#
#   base : vllm/vllm-openai:minimax-m31-flash pinned by digest
#          (BASE_IMAGE_DIGEST in scripts/env.sh)
#   +    : vcruz305/exllamav3 branch feat/moe-act3-batched at the pinned
#          rev (EXL3_REQUIRED_REV): the act-3 clamped-SwiGLU-OAI case in the
#          batched MoE kernels. A tree without it serves garbage from every
#          MiniMax MoE layer; the build refuses such a checkout.
#   +    : vcruz305/vllm-exl3 branch feat/minimax-q4-prefill at the pinned rev
#          (PLUGIN_REQUIRED_REV): the unified mixed-K K-array fix, the
#          indexer MXFP8 retention, the q4 main-KV pool, the single-pass q4
#          prefill staging, the UMA load hygiene and the C++20 build fix for
#          torch 2.14.
#
# Both runtime trees are staged into the build context by
# scripts/build_image.sh from local checkouts of the pinned revs (git
# archive of HEAD, no untracked files); the public clone below is used ONLY
# for the aarch64 guard patcher, pinned at the same plugin rev.

ARG BASE_IMAGE_DIGEST=sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c
FROM vllm/vllm-openai@${BASE_IMAGE_DIGEST}

# Re-declared inside the stage: an ARG before FROM is not visible to RUN.
ARG BASE_IMAGE_DIGEST

ARG PLUGIN_REPO=https://github.com/vcruz305/vllm-exl3.git
ARG PLUGIN_REV=41fdb7d99a441d37938c2a1bd600ac963c18a3cb

# sm_121 (GB10). MAX_JOBS is capped for the GB10's unified memory: a larger
# fan-out OOMs the host during nvcc.
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# exllamav3 CUDA extension: built from the staged pinned tree
# (exllamav3-src). The aarch64 guard patcher ships in the plugin repo
# (tools/patch_exllamav3_aarch64.py); it is a no-op when the tree is
# already guarded, so the build asserts its message either way.
ARG LOCAL_EXL3_COMMIT=unknown
COPY exllamav3-src /opt/exllamav3-local
RUN set -eux; \
    if ! command -v git >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    git clone "$PLUGIN_REPO" /opt/vllm-exl3; \
    git -C /opt/vllm-exl3 checkout "$PLUGIN_REV"; \
    git -C /opt/vllm-exl3 rev-parse HEAD; \
    PATCHER=/opt/vllm-exl3/tools/patch_exllamav3_aarch64.py; \
    if [ -f "$PATCHER" ]; then \
        sed -i 's|raise SystemExit(f"expected one x86 pause block in {path}")|print(f"already guarded (x86 pause): {path}")|' "$PATCHER"; \
        grep -q 'already guarded (x86 pause)' "$PATCHER"; \
        python3 "$PATCHER" /opt/exllamav3-local/exllamav3/exllamav3_ext; \
    fi; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    if [ -z "$CUSPARSE_DIR" ]; then echo "ERROR: cusparse.h not found under NVIDIA Python packages" >&2; exit 2; fi; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    cd /opt/exllamav3-local; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

# vllm-exl3 plugin: compiled from the staged pinned tree
# (vllm-exl3-src). The plugin setup needs C++20 on torch 2.14 (fixed at
# the pinned rev); an older tree fails at this pip step.
ARG LOCAL_PLUGIN_COMMIT=unknown
COPY vllm-exl3-src /opt/vllm-exl3-local
RUN set -eux; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    export EXL3_EXT_INCLUDE=/opt/exllamav3-local/exllamav3/exllamav3_ext; \
    cd /opt/vllm-exl3-local; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

LABEL local.m31.exl3.base-digest="${BASE_IMAGE_DIGEST}" \
      local.m31.exl3.exllamav3="local:${LOCAL_EXL3_COMMIT}" \
      local.m31.exl3.vllm-exl3="${LOCAL_PLUGIN_COMMIT}"

ENTRYPOINT []
