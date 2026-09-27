#!/usr/bin/env bash
# Next-token KLD against the f16 reference over a text. The reference is this
# engine at --kv f16 (which matches the ported reference model), and the metric
# is KL(softmax(ref) || softmax(candidate)) averaged over positions, plus the
# fraction of positions whose argmax agrees. Needs numpy.
#
# Usage: kld.sh <model.gguf> <text-file> [tokens]
set -euo pipefail

model="${1:?usage: kld.sh <model.gguf> <text-file> [tokens]}"
text="${2:?usage: kld.sh <model.gguf> <text-file> [tokens]}"
tokens="${3:-512}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
bin="$root/engine/build/yah-run"
tok="$root/engine/build/yah-tokenize"
[[ -x "$bin" && -x "$tok" ]] || { echo "kld: build yah-run and yah-tokenize" >&2; exit 2; }

ids="$("$tok" "$model" --stdin < "$text" 2>/dev/null | sed -n '2p' | cut -d' ' -f1-"$tokens")"
[[ -n "$ids" ]] || { echo "kld: tokenizer produced no ids" >&2; exit 2; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

run() {
  local tag="$1"; shift
  "$bin" "$model" --ids "$ids" --max-context 4096 \
    --dump-all-logits "$work/$tag.bin" "$@" >/dev/null 2>&1
}
run ref --kv f16
run q8 --kv q8
run q4 --kv q4
YAH_QATTN=4 run w4a4 --kv q4
YAH_QATTN=3 run w4a3 --kv q4
YAH_KV_ROT=1 run q4rot --kv q4

python3 - "$work" "$tokens" <<'PY'
import sys, numpy as np
work, tokens = sys.argv[1], int(sys.argv[2])
def load(tag):
    a = np.fromfile('%s/%s.bin' % (work, tag), dtype=np.float32)
    return a.reshape(-1, a.size // tokens)
ref = load('ref')
def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)
P = softmax(ref)
top1 = ref.argmax(1)
print('%-10s %10s %10s %10s %10s' % ('config', 'KLD mean', 'KLD p99', 'KLD max', 'top1 same'))
for tag in ['q8', 'q4', 'q4rot', 'w4a4', 'w4a4rot', 'w4a3']:
    try:
        x = load(tag)
    except OSError:
        continue
    Q = softmax(x)
    kld = (P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(1)
    print('%-10s %10.6f %10.6f %10.6f %9.2f%%' % (
        tag, kld.mean(), np.percentile(kld, 99), kld.max(),
        100.0 * (top1 == x.argmax(1)).mean()))
PY
