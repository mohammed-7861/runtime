"""One noise-floor sample: build a fresh engine, decode once to warm it, decode once timed.

The timed decode is the loop's own throughput probe (``_throughput`` in
``gitm/workloads.py``): the prefix cache is emptied outside the timer, the
whole batch goes to ``LLM.generate``, and throughput is generated tokens over
wall seconds. The batch is a Mooncake trace's request shapes, built the way
``GITM_VLLM_TRAFFIC`` builds it (``gitm.traffic.offline.load_batch``), with
``ignore_eos`` so every engine does exactly the same work.

Runs inside the gitm-submit container. Writes ``measure.json``; the record
itself is ``write_record.py``'s job.

    python3 measure_once.py --config k26_h200.json --trace trace.jsonl --out measure.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def _stack_versions(torch) -> dict[str, str | None]:
    """Driver, CUDA and NCCL versions: the run-record table keeps one column each."""
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30).stdout.split("\n")[0].strip()
    except (OSError, subprocess.SubprocessError):
        driver = ""
    try:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:  # noqa: BLE001 - absent or not built with NCCL
        nccl = None
    return {"driver_version": driver or None, "cuda_version": torch.version.cuda, "nccl_version": nccl}


def _stats(values: list[float]) -> dict[str, float | None]:
    """Mean, median, p95, p99 (nearest rank) of the values that exist."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return {"mean": None, "median": None, "p95": None, "p99": None}

    def rank(q: float) -> float:
        return xs[min(len(xs) - 1, max(0, round(q * len(xs)) - 1))]

    return {"mean": statistics.fmean(xs), "median": statistics.median(xs),
            "p95": rank(0.95), "p99": rank(0.99)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--trace", required=True, help="local copy of the trace in the config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from gitm.tracer.vllm_stats import request_records_from_outputs
    from gitm.traffic.offline import load_batch
    from gitm.workloads import _cold_prefix_cache, _max_model_len, _vocab_size, sync_device

    t0 = time.perf_counter()
    llm = LLM(model=cfg["model"], revision=cfg["hf_revision"], **cfg["engine_kwargs"])
    build_s = time.perf_counter() - t0

    traffic = cfg["traffic"]
    batch = load_batch(f"{traffic['source']}:{args.trace}", n=int(traffic["requests"]),
                       vocab_size=_vocab_size(llm), max_model_len=_max_model_len(llm))
    if batch.sha256 != traffic["sha256"]:
        raise SystemExit(f"trace sha256 {batch.sha256} != config {traffic['sha256']}")
    prompts = [{"prompt_token_ids": ids} for ids in batch.prompt_token_ids]
    params = [SamplingParams(max_tokens=o, temperature=0.0, ignore_eos=True)
              for o in batch.output_tokens]

    def decode() -> tuple[float, float, int, list]:
        _cold_prefix_cache(llm)          # outside the timer, as in _throughput
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        sync_device()
        elapsed = time.perf_counter() - start
        return toks / max(elapsed, 1e-9), elapsed, toks, outs

    warm_tps, warm_s, warm_toks, _ = decode()
    tps, elapsed, toks, outs = decode()

    records = request_records_from_outputs(outs)
    expected = sum(batch.output_tokens)
    measurement = {
        "throughput_tok_s": tps,
        "decode_s": elapsed,
        "generated_tokens": toks,
        "expected_tokens": expected,
        "requests": len(prompts),
        "warmup": {"throughput_tok_s": warm_tps, "decode_s": warm_s, "generated_tokens": warm_toks},
        "ttft_s": _stats([r.ttft_s for r in records]),
        "tpot_s": _stats([r.tpot_s for r in records]),
        "engine_build_s": build_s,
    }
    out = {
        "measurement": measurement,
        "traffic": batch.summary(),
        "engine_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        **_stack_versions(torch),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"measured {tps:.1f} tok/s ({toks} tokens in {elapsed:.1f}s; warmup {warm_tps:.1f} tok/s)",
          flush=True)
    if toks != expected:
        raise SystemExit(f"generated {toks} tokens, expected {expected}: requests did not run to length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
