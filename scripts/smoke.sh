#!/bin/bash
# Smoke test: four checks against a fresh serve.
#   1. plain chat  : a coherent answer with finish_reason=stop
#   2. tool call   : auto tool choice produces a well-formed tool_calls reply
#   3. reasoning separation: the model's reasoning arrives in reasoning_content,
#                    separate from the visible answer (the minimax_m3 parsers)
#   4. mixed-K gate: the serve log names the unified mixed-K dispatch (the
#                    GB10-validated mode; see the README's Correctness gates)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

API_PORT="${API_PORT:-8000}"
MODEL_ID="${SERVED_MODEL_NAME:-MiniMax-M3.1-Flash-2.15bpw}"
BASE="http://127.0.0.1:${API_PORT}/v1"

post() { # post <json-file-content> ; keeps curl's real rc
  curl -s -m 300 "$BASE/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$1" || { echo "SMOKE FAIL: no response from $BASE (curl rc=$?)" >&2; exit 1; }
}

echo "== 1/4 plain chat =="
R1=$(post '{
  "model": "'"$MODEL_ID"'",
  "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
  "max_tokens": 512
}')
echo "$R1" > /tmp/m31-smoke-1.json

echo "== 2/4 tool call =="
R2=$(post '{
  "model": "'"$MODEL_ID"'",
  "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "max_tokens": 512
}')
echo "$R2" > /tmp/m31-smoke-2.json

echo "== 3/4 reasoning separation =="
R3=$(post '{
  "model": "'"$MODEL_ID"'",
  "messages": [{"role": "user", "content": "A train leaves Lyon at 60 km/h toward Paris 465 km away. How long until it arrives? Think it through, then answer."}],
  "max_tokens": 2048
}')
echo "$R3" > /tmp/m31-smoke-3.json

echo "== 4/4 mixed-K dispatch mode gate (serve log) =="
# The requests above have exercised the MoE path, so the plugin's one-time
# mixed-K dispatch-mode log line must now be in the serve log. This recipe
# never sets VLLM_EXL3_MIXEDK_MODE: the GB10-validated mode is unified (one
# exl3_moe_mixedk launch per MoE layer per step; parity run 6: cosine 1.0 on
# the pack's real mixed-K experts). grouped (three 8-CTA launches per layer
# per step) is a documented fallback only; a serve that logs it — or logs no
# mode line at all — is NOT the validated stack: abort before reading any
# further result.
MODE_LINE="$(sudo -n docker logs "$CONTAINER" 2>&1 | grep -F 'EXL3 mixed-K MoE dispatch mode:' | head -1 || true)"
case "$MODE_LINE" in
  *"dispatch mode: unified"*)
    echo "mixed-K mode gate: $MODE_LINE"
    ;;
  *"dispatch mode: grouped"*)
    echo "SMOKE FAIL: the serve is in grouped mixed-K mode; this recipe never" >&2
    echo "sets VLLM_EXL3_MIXEDK_MODE. Find and undo the stray env and relaunch." >&2
    exit 1
    ;;
  *)
    echo "SMOKE FAIL: no 'EXL3 mixed-K MoE dispatch mode:' line in the serve log:" >&2
    echo "the runtime plugin predates the mode log or the MoE path never ran." >&2
    echo "Not the validated stack; abort." >&2
    exit 1
    ;;
esac

python3 - /tmp/m31-smoke-1.json /tmp/m31-smoke-2.json /tmp/m31-smoke-3.json <<'PY'
import json
import sys

fails = []

def load(p):
    with open(p) as f:
        return json.load(f)

# 1. plain chat: coherent answer, finish_reason stop
r1 = load(sys.argv[1])
ch1 = r1.get("choices", [{}])[0]
text1 = (ch1.get("message", {}).get("content") or "").strip()
if ch1.get("finish_reason") != "stop":
    fails.append(f"1: finish_reason={ch1.get('finish_reason')!r} (want 'stop')")
if "paris" not in text1.lower():
    fails.append(f"1: expected 'Paris' in the reply, got {text1[:120]!r}")

# 2. tool call: well-formed tool_calls with the declared function and arguments
r2 = load(sys.argv[2])
ch2 = r2.get("choices", [{}])[0]
msg2 = ch2.get("message", {})
tcs = msg2.get("tool_calls") or []
ok2 = False
if tcs:
    tc = tcs[0]
    fn = tc.get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:
        args = {}
    ok2 = fn.get("name") == "get_weather" and "paris" in json.dumps(args).lower()
if not ok2:
    fails.append(f"2: no well-formed get_weather tool call (finish_reason="
                 f"{ch2.get('finish_reason')!r}, content={str(msg2.get('content'))[:120]!r})")

# 3. reasoning separation: reasoning arrives in its own message field, apart
#    from the visible answer; the answer itself must be coherent. This vLLM
#    build names the field "reasoning" (a GB10-measured serve saw 488 chars of
#    reasoning, reasoning_tokens 129, with a correct visible answer); older
#    builds used "reasoning_content", so either is accepted.
r3 = load(sys.argv[3])
ch3 = r3.get("choices", [{}])[0]
msg3 = ch3.get("message", {})
reasoning = (msg3.get("reasoning") or msg3.get("reasoning_content") or "").strip()
content3 = (msg3.get("content") or "").strip()
if not reasoning:
    fails.append("3: no reasoning/reasoning_content in the reply (reasoning parser "
                 f"not separating?) message keys={sorted(msg3)}")
if not content3:
    fails.append("3: empty visible answer")
if reasoning and reasoning == content3:
    fails.append("3: reasoning equals content (not separated)")
if "7.75" not in content3 and "7 h" not in content3 and "hours" not in content3.lower():
    fails.append(f"3: answer does not contain the expected ~7.75 h travel time: {content3[:160]!r}")

if fails:
    for f in fails:
        print("SMOKE FAIL " + f)
    sys.exit(1)
print("SMOKE PASS: chat ok, tool call ok, reasoning separated, mixed-K unified")
PY
