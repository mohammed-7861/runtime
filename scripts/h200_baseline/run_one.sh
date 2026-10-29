#!/bin/bash
# One K2.6 baseline run inside the gitm-submit container: fresh engine, warmup
# decode, timed decode, run record. submit.py writes this folder to
# /out/h200_baseline and runs it; finalize runs on the host after the health check.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/out}
CFG="$HERE/k26_h200.json"
export HF_HUB_OFFLINE=1          # weights are resident on the node; never download them

python3 "$HERE/write_record.py" init --config "$CFG" --out "$OUT"
python3 "$HERE/write_record.py" fetch --config "$CFG" --dest "$OUT/trace.jsonl"

set +e
python3 "$HERE/measure_once.py" --config "$CFG" --trace "$OUT/trace.jsonl" \
    --out "$OUT/measure.json" 2>&1 | tee "$OUT/engine.log"
rc=${PIPESTATUS[0]}
set -e

python3 "$HERE/write_record.py" measure --out "$OUT"
rm -f "$OUT/trace.jsonl"         # pinned by URL + sha256 in inputs.json
exit "$rc"
