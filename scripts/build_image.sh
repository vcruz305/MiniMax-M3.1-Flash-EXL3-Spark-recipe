#!/bin/bash
# Build the serving image on the Spark: upstream MiniMax-M3.1-Flash engine
# (pinned by digest) + the exllamav3 fork's act-3 fix + the vllm-exl3
# plugin, BOTH compiled from the pinned public checkouts:
#   EXL3_SRC   branch feat/moe-act3-batched (EXL3_REQUIRED_REV): adds the
#              act-3 case to the batched MoE kernels; without it every
#              MiniMax MoE layer outputs garbage.
#   PLUGIN_SRC branch feat/minimax-m3-exl3 (PLUGIN_REQUIRED_REV): the
#              unified mixed-K K-array fix, the indexer MXFP8 retention,
#              the q4 main-KV pool, the UMA load hygiene, the C++20 setup
#              fix.
#
#   bash scripts/build_image.sh                        # both from env.sh defaults
#   EXL3_SRC=/path/to/fork-checkout bash scripts/build_image.sh
#   PLUGIN_SRC=/path/to/plugin-checkout bash scripts/build_image.sh
#   FORCE=1 bash scripts/build_image.sh                # allow pre-fix pins (escape hatch)
#
# The image is tagged m31-flash-exl3-spark:dev-<exl3 short>-<plugin short>
# and labelled with both full shas; serve.sh re-checks the labels against
# the same pins and refuses any mismatch. A checkout below a pinned rev is
# refused unless FORCE=1. The clone/checkout steps inside the Dockerfile are
# pinned to the plugin rev, so their docker layer cache hits across
# rebuilds; only the two staged trees rebuild.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

FORCE="${FORCE:-0}"

if [ -z "$EXL3_REV" ]; then
  echo "REFUSING to build: EXL3_SRC=$EXL3_SRC is not a git checkout." >&2
  echo "Clone the fork at the pinned rev first (see the README quick start):" >&2
  echo "  git clone -b feat/moe-act3-batched $EXL3_SRC  # then: git -C $EXL3_SRC checkout $EXL3_REQUIRED_REV" >&2
  exit 1
fi
if [ -z "$PLUGIN_REV" ]; then
  echo "REFUSING to build: PLUGIN_SRC=$PLUGIN_SRC is not a git checkout." >&2
  echo "Clone the plugin at the pinned rev first (see the README quick start):" >&2
  echo "  git clone $PLUGIN_REPO $PLUGIN_SRC  # then: git -C $PLUGIN_SRC checkout $PLUGIN_REQUIRED_REV" >&2
  exit 1
fi
if [ -n "$(git -C "$EXL3_SRC" status --porcelain)" ]; then
  echo "REFUSING to build: $EXL3_SRC has uncommitted changes." >&2
  echo "Commit first: the image label must name a real revision." >&2
  exit 1
fi
if [ -n "$(git -C "$PLUGIN_SRC" status --porcelain)" ]; then
  echo "REFUSING to build: $PLUGIN_SRC has uncommitted changes." >&2
  echo "Commit first: the image label must name a real revision." >&2
  exit 1
fi
# Both checkouts must sit at the pinned rev or a descendant of it: building
# an older tree silently reintroduces the act-3 batched garbage (exllamav3)
# or the mixed-K unified K-array regression (plugin).
if [ "$FORCE" != "1" ]; then
  if ! git -C "$EXL3_SRC" merge-base --is-ancestor "$EXL3_REQUIRED_REV" "$EXL3_REV" 2>/dev/null; then
    echo "REFUSING to build: EXL3_SRC HEAD $EXL3_REV is not $EXL3_REQUIRED_REV" >&2
    echo "(feat/moe-act3-batched) or a descendant: the act-3 batched fix would" >&2
    echo "be missing from the compiled extension. FORCE=1 overrides." >&2
    exit 1
  fi
  if ! git -C "$PLUGIN_SRC" merge-base --is-ancestor "$PLUGIN_REQUIRED_REV" "$PLUGIN_REV" 2>/dev/null; then
    echo "REFUSING to build: PLUGIN_SRC HEAD $PLUGIN_REV is not $PLUGIN_REQUIRED_REV" >&2
    echo "(feat/minimax-m3-exl3) or a descendant: the mixed-K unified K-array" >&2
    echo "fix would be missing from the baked plugin. FORCE=1 overrides." >&2
    exit 1
  fi
fi
if [ "$(uname -m)" != "aarch64" ]; then
  echo "REFUSING to build: $(uname -m) host. The image is arm64-only (GB10)." >&2
  exit 1
fi
if ! sudo -n docker image inspect "vllm/vllm-openai@${BASE_IMAGE_DIGEST}" >/dev/null 2>&1; then
  echo "base image vllm/vllm-openai@${BASE_IMAGE_DIGEST} not present locally." >&2
  echo "Pull it first: sudo docker pull vllm/vllm-openai@${BASE_IMAGE_DIGEST}" >&2
  exit 1
fi

# Fresh, throwaway build context: the Dockerfile plus both pinned trees as
# exllamav3-src / vllm-exl3-src (the names the Dockerfile COPYs), staged
# with git archive so the context is exactly HEAD (no .git, no untracked
# files). Nothing is written into either checkout or the recipe tree.
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "$RECIPE_HOME/Dockerfile" "$CTX/Dockerfile"
mkdir "$CTX/exllamav3-src" "$CTX/vllm-exl3-src"
git -C "$EXL3_SRC" archive HEAD | tar -x -C "$CTX/exllamav3-src"
git -C "$PLUGIN_SRC" archive HEAD | tar -x -C "$CTX/vllm-exl3-src"

echo "base   : vllm/vllm-openai@$BASE_IMAGE_DIGEST"
echo "exl3   : $EXL3_SRC @ $EXL3_REV (required >= $EXL3_REQUIRED_REV)"
echo "plugin : $PLUGIN_SRC @ $PLUGIN_REV (required >= $PLUGIN_REQUIRED_REV)"
echo "profile: $PROFILE (MAXLEN=$MAXLEN KV_Q4=$KV_Q4)"
echo "image  : $IMAGE"

sudo -n docker build \
  -f "$CTX/Dockerfile" \
  --build-arg "BASE_IMAGE_DIGEST=$BASE_IMAGE_DIGEST" \
  --build-arg "PLUGIN_REPO=$PLUGIN_REPO" \
  --build-arg "PLUGIN_REV=$PLUGIN_REQUIRED_REV" \
  --build-arg "LOCAL_EXL3_COMMIT=$EXL3_REV" \
  --build-arg "LOCAL_PLUGIN_COMMIT=$PLUGIN_REV" \
  -t "$IMAGE" \
  "$CTX"

# The image must carry both checkouts' HEADs as its labels; anything else is
# not buildable by this recipe (serve.sh makes the same checks).
img_label() { sudo -n docker image inspect "$IMAGE" --format "{{index .Config.Labels \"$1\"}}" 2>/dev/null || true; }
RAW_EXL3="$(img_label local.m31.exl3.exllamav3)"
IMG_EXL3="${RAW_EXL3#local:}"
IMG_PLUGIN="$(img_label local.m31.exl3.vllm-exl3)"
if [ "$IMG_EXL3" != "$EXL3_REV" ] || [ "$IMG_PLUGIN" != "$PLUGIN_REV" ]; then
  echo "BUILD FAILED: image labels are exllamav3=${IMG_EXL3:-<none>}" >&2
  echo "(want $EXL3_REV) / vllm-exl3=${IMG_PLUGIN:-<none>} (want $PLUGIN_REV)." >&2
  exit 1
fi

# Capability check: the compiled exllamav3 extension, the plugin's compiled
# extension and the plugin package must all import in the new image, and
# CUDA must see the GB10.
CHK="$(sudo -n docker run --rm --gpus all --entrypoint python3 "$IMAGE" \
  -c 'import torch, exllamav3_ext, vllm_exl3_c, vllm_exl3; print("cap", torch.cuda.get_device_capability(), torch.__version__)' 2>&1 | tail -1)"
echo "check  : $CHK"
case "$CHK" in
  "cap (12, 1) "*) ;;
  *)
    echo "BUILD FAILED: capability check did not print 'cap (12, 1) ...' (got: $CHK)" >&2
    exit 1
    ;;
esac
sudo -n docker image inspect "$IMAGE" --format 'built {{.Id}} {{.Created}}'
