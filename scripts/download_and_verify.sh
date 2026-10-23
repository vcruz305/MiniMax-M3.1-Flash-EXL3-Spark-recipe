#!/bin/bash
# Download the SAGE 2.15bpw pack at the PINNED revision and verify every file.
#
#   bash scripts/download_and_verify.sh [/target/dir]
#
# Fail-closed, three steps:
#   1. disk: needs ~112 GiB free at the target (the pack's non-tmp files
#      total 118,985,525,140 B; the largest shard alone is 8.19 GB and xet
#      stages chunks, so the free-space check uses a 20 GiB slack).
#   2. download: hf download of the pinned revision ONLY -- no revision
#      fallback to a newer commit is allowed. If the pinned revision no
#      longer resolves on the Hub, the script refuses and points at the
#      README's "Pack revision" section; it never pulls different bytes.
#   3. verify: EVERY file must match its recorded hash -- sha256 for the 17
#      LFS files (scripts/SHA256SUMS) and the git blob sha1 for the 23
#      non-LFS files (scripts/PACK-BLOB-SUMS). Any MISSING/FAILED hash
#      refuses the serve; the model directory the serve uses is only
#      accepted once this script has passed.
#
# HF token: the pack repo is gated with manual approval; request access on
# the model page first, then log in (`hf auth login` / HF_TOKEN in the
# environment). This script never handles a token value itself;
# download_and_verify.sh only checks that huggingface_hub can see the repo
# (whoami + repo_info), never prints the token.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

DEST="${1:-$MODELS_DIR}"
PACK_DIR="$DEST/$MODEL_NAME"

echo "repo   : $MODEL_REPO @ $MODEL_REV"
echo "dest   : $PACK_DIR"
echo "prereq : gate approval (request on the model page) + hf auth login / hf CLI / hf_xet (pip install -U \"huggingface_hub[cli]\" hf_xet)"

# ---- step 0: tooling -------------------------------------------------------
python3 - <<'PYEOF'
import importlib.util, shutil, sys
missing = [m for m in ("huggingface_hub", "hf_xet") if importlib.util.find_spec(m) is None]
if missing:
    print("REFUSING to download: missing Python modules: " + ", ".join(missing), file=sys.stderr)
    print('Install them first: pip install -U "huggingface_hub[cli]" hf_xet', file=sys.stderr)
    sys.exit(3)
if shutil.which("sha256sum") is None or shutil.which("sha1sum") is None:
    print("REFUSING to download: sha256sum/sha1sum not found", file=sys.stderr)
    sys.exit(3)
PYEOF

# ---- step 1: disk ----------------------------------------------------------
NEED=$((118985525140 + 21474836480))   # pack + 20 GiB xet staging slack
AVAIL="$(df -B1 --output=avail "$DEST" 2>/dev/null | tail -1 | tr -d ' ')"
if [ -z "$AVAIL" ] || [ "$AVAIL" -lt "$NEED" ]; then
  echo "REFUSING to download: $DEST has $((AVAIL / 1000000000)) GB free; the pack needs ~112 GiB + staging slack." >&2
  exit 1
fi

# ---- step 2: download (pinned revision only, no fallback) ------------------
python3 - "$MODEL_REPO" "$MODEL_REV" "$PACK_DIR" <<'PYEOF'
import sys
from huggingface_hub.errors import GatedRepoError, RevisionNotFoundError
from huggingface_hub import model_info, snapshot_download
repo, rev, dest = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    info = model_info(repo, revision=rev)
except GatedRepoError:
    print("REFUSING: " + repo + " is gated (manual approval). Request access on the model page,", file=sys.stderr)
    print("then log in with `hf auth login`; this script never handles a token value.", file=sys.stderr)
    sys.exit(4)
except RevisionNotFoundError:
    print("REFUSING: revision " + rev + " of " + repo + " does not exist on the Hub.", file=sys.stderr)
    print("Do NOT fall back to a newer revision: the pack's measured bytes are pinned by scripts/SHA256SUMS and scripts/PACK-BLOB-SUMS.", file=sys.stderr)
    print("See README \"Pack revision\": the recorded hashes are the only acceptance criterion; no other tree may be served.", file=sys.stderr)
    sys.exit(5)
print("Hub tree at pinned rev:", info.sha)
snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir=dest,
    exclude=["*.tmp*"],
    max_workers=8,
)
print("download complete")
PYEOF

# ---- step 3: verify every file ---------------------------------------------
# 3a. LFS files: sha256sum -c against scripts/SHA256SUMS.
echo "== sha256 (LFS files) =="
( cd "$PACK_DIR" && sha256sum -c "$HERE/SHA256SUMS" --strict ) || {
  echo "REFUSING to serve this pack: LFS verification FAILED (see lines above)." >&2
  exit 6
}
# 3b. non-LFS files: recompute each git blob sha1 and compare to
# scripts/PACK-BLOB-SUMS. A missing file is a refusal too.
echo "== git blob sha1 (non-LFS files) =="
python3 - "$PACK_DIR" "$HERE/PACK-BLOB-SUMS" <<'PYEOF'
import hashlib, os, sys
pack, sums = sys.argv[1], sys.argv[2]
bad = 0
with open(sums, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        want, rel = line.split("  ", 1)
        path = os.path.join(pack, rel)
        if not os.path.isfile(path):
            print("MISSING " + rel, file=sys.stderr); bad += 1; continue
        data = open(path, "rb").read()
        h = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if h != want:
            print("FAILED " + rel + " (want " + want[:12] + ", got " + h[:12] + ")", file=sys.stderr)
            bad += 1
print(("blob check: FAILED (%d bad/missing files)" % bad) if bad else "blob check: OK (all non-LFS files match)", file=sys.stderr if bad else sys.stdout)
sys.exit(7 if bad else 0)
PYEOF

echo "VERIFIED: $PACK_DIR is byte-identical to the measured pack at $MODEL_REV."
echo "serve.sh accepts this directory (scripts/env.sh: MODEL_DIR)."
