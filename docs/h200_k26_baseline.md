# Baseline series: Kimi K2.6 on 8x H200 (ENG-95)

The rows the noise floor is computed from. Repeated identical runs of one tuned
K2.6 config on one H200 node, each submitted through `gitm-submit` and written
as one run record with its node-health result. The floor statistic itself
(Rule 3) is computed by the harness (ENG-101), not here; [`noise-floor.md`]
describes it. Reviewer: Sebastian.

**Status (Oct 29):** series complete. 11 runs (15–25, jobs 202–213), all healthy,
all `baseline`, in `/var/lib/gitm/results/<job>/` on `gitmachine-2`. Ready for review.

How it got here:

- **Oct 27:** #189 made `gitm-node-health` mandatory, but it wasn't installed on either node, so every job was refused.
- **Oct 28:** runs 1–7 ran with `--no-health`, so their health is unknown.
- **After that:** the checker was installed. Its `power_cap` rule failed every full-load run, because "SW power cap active" is normal under load. Runs 8–14 were discarded and requeued on that rule.
- **The fix:** #222 fixed the rule on main, but the nodes still have the #189 copy, and only root can update it. So `submit.py` ships main's checker (`--health-ref`, default #222) into each job's own results folder and points `HEALTH_BIN` at it. gitm-submit's pre/sample/post/summarize/gate steps are unchanged, and kernel-log reads go through the node's `gitm-dmesg-xid` sudo helper. Each record names the checker it used (`gitm.node_health.checker`).
- **Once the nodes are updated:** `--node-health` uses the installed copy.

**Store:** `write_record.py rows --sql` emits exactly the columns of
`run_records` (Pamela's dump, Oct 27). Its `state` CHECK doesn't yet allow
`baseline`, so those inserts fail until `'baseline'` is added (asked Pamela).

## What one row is

One run is one `gitm-submit` job, which means one fresh container and one
fresh engine:

1. Build the engine with the frozen config.
2. Run one warmup decode of the batch. It is recorded but not counted.
3. Run one timed decode: throughput = generated tokens ÷ wall seconds, the loop's probe (`_throughput` in `gitm/workloads.py`).
4. Shut down. The host's node-health post-check runs, then `write_record.py finalize` folds its result into the record.

A floor sample is the ratio of two consecutive runs' throughput, minus 1. It
needs a fresh engine on each side, because repeated decodes on one engine miss
the build-to-build noise ([`noise-floor.md`] §1). The harness's own baseline
and candidate runs are separate jobs too, so job-to-job noise is the noise it
has to beat.

## Protocol

| | |
|---|---|
| Hardware | 8x H200, one node: `gitmachine-2`, where K2.6 is resident |
| Model | `moonshotai/Kimi-K2.6` at `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` (`kimi-k2.6-int4`), loaded offline (`HF_HUB_OFFLINE=1`) |
| Engine | vLLM from the node's default image. The version is recorded per run (`engine_version`) and the image ref in `inputs.json` |
| Config | TP 8, `gpu_memory_utilization 0.92`, `max_num_batched_tokens 8192`, `max_num_seqs 256`, `max_model_len 65536`. KV auto (bf16), no EP, no speculative decoding, untraced. Flags ported from the MI355X floor ([`autoresearch_noise_floor_mi355x.md`]) |
| Load | Mooncake `conversation_trace.jsonl` @ `kvcache-ai/Mooncake` `3cca71da`, sha256 `b8cbb061…08b9df`; first 256 usable requests as an offline batch (`gitm.traffic.offline`), `ignore_eos`. 10 requests over 64k are skipped, never trimmed. Mean input 10,965 tokens, mean output 367; 2.81M input / 94,011 output tokens per decode |
| Metric | Output tokens/s of the timed decode; TTFT and TPOT mean, median, p95 and p99 alongside |
| Target | 11 healthy runs for now, giving 10 consecutive pairs. Extend to 21 (the 20 samples of H200 job 100) if the harness wants more; new runs pair on from run 25 |
| gitm | shipped in the job from this repo @ `bbc6c075` (the repo is private, so pip can't fetch it) |
| Node health | `deploy/h200/gitm-node-health` @ `48ccd32` (#222), shipped in the job |
| Scripts | `scripts/h200_baseline/` (sha256 of each recorded per run). Submit with `py scripts/h200_baseline/submit.py --count N` |

The config is in one file, `scripts/h200_baseline/k26_h200.json`. Every run
reads it, and `inputs.json` records it.

## Where the rows are

**The series data stays on the node.** It is not in this repo. On `gitmachine-2`:

```
/var/lib/gitm/results/202/   run 15
/var/lib/gitm/results/204/   run 16  …through…
/var/lib/gitm/results/213/   run 25        (job 203 is a diagnostic, not a run)
```

To turn them into `run_records` rows, run this on the node (as a job, or once `/mnt/gitm-runs` is mounted):

```
cd /var/lib/gitm/results
python3 202/h200_baseline/write_record.py rows 202 204 205 206 207 208 209 210 211 212 213 \
    --out /tmp/rows.jsonl --sql /tmp/rows.sql
```

Runs 1–7 (jobs 170–176, no health) and the discarded attempts of runs 8–14
(`/var/lib/gitm/results/discarded/181`, `189`–`195`, old `power_cap` rule) are still
there for reference. They are not part of the series.

Per run, in `<results>/<slurm job id>/`:

- `inputs.json`: run ID (ENG-76 format), job id, node, model, revision, engine version, flags, trace, image, script hashes.
- `record.json`: the ENG-72 run-record fields. `state` is `baseline`, or `invalid` when the node was unhealthy or the decode didn't complete. `node_healthy_for_window` comes from `node_health_window.json`. Lever/canary/floor fields are null. Measurements are under `gitm.measurement`.
- `engine.log`, `measure.json`, `node_health.jsonl`, `node_health_window.json`.

`<results>` is `/var/lib/gitm/results` on the node until `/mnt/gitm-runs` is
mounted. Once it is, `finalize` also copies each run to
`/mnt/gitm-runs/<run id>/`. `write_record.py rows <dir> --out rows.jsonl` turns
the records into store rows. Columns follow ENG-72 until the store schema
(ENG-82) is published, then only `STORE_COLUMNS` changes.

Every job's log ends with a `GITM_RECORD {...}` line (run ID, state, health,
throughput), so the status page's log tail shows the result.

## Run log

| # | Slurm job | Run ID | State | Healthy | Throughput (tok/s) | Note |
|---|---|---|---|---|---|---|
| smoke | 148 | `20261028T033315Z-kimi-k26-gm2-ff23` | baseline | unknown (`--no-health`) | 435.1 (warmup 434.0) | Pipeline test, not counted. 94,011 tokens in 216.1 s. Engine force-killed at shutdown, 1 leaked shared-memory object |
| 1 | 170 | `…-kimi-k26-gm2-1fa0` | baseline | unknown (`--no-health`) | 433.64 | Batch 1 (Oct 28) |
| 2 | 171 | `…-kimi-k26-gm2-2d11` | baseline | unknown | 434.21 | |
| 3 | 172 | `…-kimi-k26-gm2-78df` | baseline | unknown | 433.47 | |
| 4 | 173 | `…-kimi-k26-gm2-7842` | baseline | unknown | 434.67 | |
| 5 | 174 | `…-kimi-k26-gm2-bc81` | baseline | unknown | 434.20 | |
| 6 | 175 | `…-kimi-k26-gm2-d0c3` | baseline | unknown | 434.15 | |
| 7 | 176 | `…-kimi-k26-gm2-d41c` | baseline | unknown | 435.06 | |
| 15 | 202 | see `record.json` | baseline | **true** (122 samples, no failed signals) | 434.37 | First run with main's checker shipped in the job (#222, blob `467eca8`); runs 8–14 (jobs 181, 189–195) were all discarded on the old `power_cap` rule and cancelled |

| 16 | 204 | `…-2b25` | baseline | true | 434.60 | |
| 17 | 205 | `…-f0c3` | baseline | true | 435.23 | |
| 18 | 206 | `…-9407` | baseline | true | 434.67 | |
| 19 | 207 | `…-7aeb` | baseline | true | 434.82 | |
| 20 | 208 | `…-aab2` | baseline | true | 434.02 | |
| 21 | 209 | `…-a597` | baseline | true | 434.82 | |
| 22 | 210 | `…-2c6b` | baseline | true | 434.40 | |
| 23 | 211 | `…-8d11` | baseline | true | 434.87 | |
| 24 | 212 | `…-7624` | baseline | true | 434.58 | |
| 25 | 213 | `…-53aa` | baseline | true | 434.53 | |

**The series (Oct 29):** runs 15–25, 11 runs, all healthy with main's checker, all 94,011
tokens, none discarded or requeued. Run 15 is `…-0a66`. They replace runs 1–7, which stay
on disk for reference but aren't part of the series. Jobs 214–223 were cancelled before
running. Sanity check only (ENG-101 computes the floor): mean 434.63 tok/s, sd 0.32,
consecutive-pair RMS 0.116% over 10 pairs, largest single change 0.18%.

Batch 1 sanity check (not the floor; ENG-101 computes that): mean 434.20 tok/s,
consecutive-pair RMS 0.17% over 6 pairs. Full run IDs are in each run's
`record.json` under `/var/lib/gitm/results/<job>/`; `/mnt/gitm-runs` is still
not mounted.

## Decisions to confirm

- **Offline probe, not a server replay.** This matches the loop's keep rule and the existing floor. A server replay (PR #215, Option A) would give a floor in a different metric. The harness has to measure its runs the same way for the floor to apply (Isaiah, Rishith).
- **One warmup decode per run, then one timed decode.** The harness should do the same.
- **Trace:** Mooncake conversation, the first 256 usable requests. No trace is locked for the Kimi baseline yet (Sebastian).

[`noise-floor.md`]: https://github.com/GitM-Labs/runtime/pull/197
[`autoresearch_noise_floor_mi355x.md`]: autoresearch_noise_floor_mi355x.md
