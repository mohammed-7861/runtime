#!/usr/bin/env python3
"""Run record for one K2.6 baseline run: ``inputs.json`` + ``record.json``.

``inputs.json`` is what was run (provenance); ``record.json`` is what came out,
in the run-record fields of the loop design (ENG-72). A baseline run has no
lever and no canary, so those fields are null and ``state`` is ``baseline``.
The noise floor is not computed here: the harness (ENG-101) computes it from
these rows.

    write_record.py fetch    --config C --dest trace.jsonl   # in the container
    write_record.py init     --config C --out DIR            # in the container
    write_record.py measure  --out DIR                       # in the container
    write_record.py finalize DIR [--image REF]               # on the host, after the health check
    write_record.py rows     PATH... --out rows.jsonl        # anywhere: store rows

Stdlib only, Python 3.9+, so ``finalize`` runs on the node itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

#: The run-record fields of the loop design (ENG-72), in its order.
RECORD_FIELDS = (
    "beat_baseline", "delta", "iterations", "wall_time_s", "node_healthy_for_window",
    "lever_id", "knob_values", "canary_instance", "control_instances", "window_ids",
    "noise_floor", "threshold", "min_effect", "slo_guard", "state", "policy_id",
    "degradations", "baseline_config", "candidate_config",
)

#: Pamela's ``run_records`` table (ENG-82; Postgres dump of 2026-10-27), in its
#: order, minus ``inserted_at`` (the database fills it).
TABLE_COLUMNS = (
    "run_id", "window_ids", "lever_id", "knob_values", "canary_instance_id",
    "control_instance_ids", "beat_baseline", "delta", "iterations", "wall_time_s",
    "node_healthy_for_window", "noise_floor", "threshold", "min_effect", "slo_guard",
    "state", "policy_id", "degradations", "baseline_config", "candidate_config",
    "driver_version", "cuda_version", "nccl_version",
)

#: The table's ``state`` CHECK. ``baseline`` isn't in it yet: asked Pamela to add it.
TABLE_STATES = ("proposed", "canary", "promoted", "rolled_back", "invalid", "inconclusive")

#: Record field -> table column, where the names differ.
STORE_COLUMNS = {f: f for f in RECORD_FIELDS} | {
    "canary_instance": "canary_instance_id",
    "control_instances": "control_instance_ids",
}

#: Software-stack versions: in the record and the table, read from measure.json.
VERSION_FIELDS = ("driver_version", "cuda_version", "nccl_version")

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/mnt/gitm-runs")
DURABLE_FILES = ("record.json", "inputs.json", "engine.log", "measure.json",
                 "node_health_window.json", "node_health.jsonl")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def model_slug(model: str) -> str:
    """'moonshotai/Kimi-K2.6' -> 'kimi-k26' (ENG-76)."""
    return re.sub(r"[^a-z0-9-]", "", model.split("/")[-1].lower())


def node_short(host: str) -> str:
    """'gitmachine-2' -> 'gm2'; anything else -> its own sanitized short name."""
    host = host.split(".")[0]
    m = re.fullmatch(r"gitmachine-(\d+)", host)
    return f"gm{m.group(1)}" if m else re.sub(r"[^a-z0-9]", "", host.lower()) or "node"


def mint_run_id(model: str, host: str, when: datetime | None = None) -> str:
    """``<UTC>-<model-slug>-<node>-<rand4>``, as agreed in ENG-76."""
    when = when or datetime.now(timezone.utc)
    return f"{when:%Y%m%dT%H%M%SZ}-{model_slug(model)}-{node_short(host)}-{secrets.token_hex(2)}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def degradation(what: str, detail: str) -> dict:
    return {"what": what, "detail": detail}


# --------------------------------------------------------------------------
# container side
# --------------------------------------------------------------------------

def cmd_fetch(args) -> int:
    traffic = json.loads(Path(args.config).read_text())["traffic"]
    dest = Path(args.dest)
    with urllib.request.urlopen(traffic["url"], timeout=120) as r:
        dest.write_bytes(r.read())
    got = _sha256(dest)
    if got != traffic["sha256"]:
        dest.unlink()
        sys.exit(f"write_record: trace sha256 {got} != pinned {traffic['sha256']}")
    return 0


def build_inputs(cfg: dict, env: dict, host: str, scripts: dict[str, str]) -> dict:
    return {
        "run_id": mint_run_id(cfg["model"], host),
        "series": cfg["series"],
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "submitter": env.get("SLURM_JOB_USER") or env.get("USER"),
        "node": host.split(".")[0],
        "model": cfg["model"],
        "hf_revision": cfg["hf_revision"],
        "engine": cfg["engine"],
        "engine_version": None,
        "engine_flags": cfg["engine_kwargs"],
        "traffic": cfg["traffic"],
        "gitm_ref": env.get("GITM_REF"),
        "image": None,
        "scripts_sha256": scripts,
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
    }


def cmd_init(args) -> int:
    out = Path(args.out)
    here = Path(args.config).parent
    scripts = {p.name: _sha256(p) for p in sorted(here.iterdir()) if p.is_file()}
    cfg = json.loads(Path(args.config).read_text())
    _write(out / "inputs.json", build_inputs(cfg, dict(os.environ), socket.gethostname(), scripts))
    return 0


def build_record(inputs: dict, measure: dict | None) -> dict:
    """The ENG-72 record for a baseline run. ``state`` is ``invalid`` when the
    measurement is missing or the batch did not run to length."""
    rec: dict = {f: None for f in RECORD_FIELDS}
    degr: list[dict] = []
    m = (measure or {}).get("measurement")
    if not m:
        state = "invalid"
        degr.append(degradation("measurement", "no measure.json: the engine build or decode failed"))
    elif m["generated_tokens"] != m["expected_tokens"]:
        state = "invalid"
        degr.append(degradation("measurement", f"generated {m['generated_tokens']} tokens, "
                                               f"expected {m['expected_tokens']}"))
    else:
        state = "baseline"
    skipped = ((measure or {}).get("traffic") or {}).get("skipped") or {}
    if skipped:
        degr.append(degradation("traffic", f"requests skipped, not trimmed: {skipped}"))
    rec.update({
        "iterations": 1 if m else 0,
        "wall_time_s": m["decode_s"] if m else None,
        "control_instances": [inputs["run_id"]],
        "window_ids": [],
        "slo_guard": {
            "bounds": None,
            "ttft_p95_s": m["ttft_s"]["p95"] if m else None,
            "tpot_p95_s": m["tpot_s"]["p95"] if m else None,
        },
        "state": state,
        "degradations": degr,
        "baseline_config": {
            "model": inputs["model"], "hf_revision": inputs["hf_revision"],
            "engine": inputs["engine"],
            "engine_version": (measure or {}).get("engine_version"),
            "engine_kwargs": inputs["engine_flags"],
        },
    })
    for f in VERSION_FIELDS:
        rec[f] = (measure or {}).get(f)
    rec["run_id"] = inputs["run_id"]
    rec["series"] = inputs["series"]
    rec["gitm.measurement"] = m
    rec["gitm.traffic"] = (measure or {}).get("traffic")
    rec["gitm.gpu"] = {"name": (measure or {}).get("gpu_name"), "count": (measure or {}).get("gpu_count")}
    return rec


def cmd_measure(args) -> int:
    out = Path(args.out)
    inputs = _read(out / "inputs.json")
    if inputs is None:
        sys.exit("write_record: no inputs.json; run init first")
    measure = _read(out / "measure.json")
    rec = build_record(inputs, measure)
    _write(out / "record.json", rec)
    inputs["engine_version"] = (measure or {}).get("engine_version")
    inputs["finished_at"] = _now()
    inputs["status"] = "measured" if rec["state"] == "baseline" else "failed"
    _write(out / "inputs.json", inputs)
    return 0


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

def apply_health(rec: dict, window: dict | None, checker: bool = True) -> dict:
    """Fold the node-health window in; an unhealthy or missing window invalidates the run.

    ``checker=False`` is a run submitted while the node had no health checker
    installed: health is unknown (null), not unhealthy, and the record says so."""
    if not checker and window is None:
        rec["node_healthy_for_window"] = None
        rec["degradations"].append(degradation(
            "node_health", "gitm-node-health not installed on the node; health unknown for this window"))
        return rec
    if window is None:
        rec["node_healthy_for_window"] = None
        rec["state"] = "invalid"
        rec["degradations"].append(degradation("node_health", "no node_health_window.json"))
        return rec
    healthy = window.get("healthy") is True
    rec["node_healthy_for_window"] = healthy
    rec["gitm.node_health"] = {"healthy": healthy, "failed_signals": window.get("failed_signals") or [],
                               "job_id": window.get("job_id"), "node": window.get("node")}
    if not healthy:
        rec["state"] = "invalid"
    return rec


def image_info(ref: str | None) -> dict | None:
    if not ref:
        return None
    info: dict = {"ref": ref}
    p = Path(ref)
    if p.is_file():                      # a pre-imported .sqsh: size and mtime identify it cheaply
        st = p.stat()
        info.update(size=st.st_size, mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"))
    return info


def cmd_finalize(args) -> int:
    d = Path(args.dir)
    inputs = _read(d / "inputs.json")
    if inputs is None:
        print("GITM_RECORD " + json.dumps({"state": "invalid", "detail": "no inputs.json"}), flush=True)
        return 1
    rec = _read(d / "record.json") or build_record(inputs, None)
    rec = apply_health(rec, _read(d / "node_health_window.json"), checker=not args.no_health)
    shipped = _read(d / "nh" / "checker.json")   # submit.py shipped main's checker with the job
    if shipped is not None:
        rec.setdefault("gitm.node_health", {})["checker"] = shipped
    inputs["image"] = image_info(args.image)

    root = Path(args.results_root)
    if root.is_dir() and os.access(root, os.W_OK):
        dest = root / inputs["run_id"]
        dest.mkdir(parents=True, exist_ok=True)
        rec["gitm.durable_path"] = str(dest)
    else:
        dest = None
        rec["degradations"].append(degradation(
            "storage", f"{root} not mounted or not writable; results stay in {d}"))
    _write(d / "record.json", rec)
    _write(d / "inputs.json", inputs)
    if dest is not None:
        for name in DURABLE_FILES:
            if (d / name).is_file():
                shutil.copy2(d / name, dest / name)

    m = rec.get("gitm.measurement") or {}
    print("GITM_RECORD " + json.dumps({
        "run_id": rec["run_id"], "state": rec["state"],
        "healthy": rec["node_healthy_for_window"],
        "throughput_tok_s": m.get("throughput_tok_s"), "dir": str(dest or d)}), flush=True)
    return 0


# --------------------------------------------------------------------------
# store rows
# --------------------------------------------------------------------------

def store_row(rec: dict) -> dict:
    """One ``run_records`` row: exactly the table's columns. Everything else
    (measurements, provenance) stays in record.json / inputs.json."""
    row = {STORE_COLUMNS[f]: rec.get(f) for f in RECORD_FIELDS}
    row.update({f: rec.get(f) for f in VERSION_FIELDS}, run_id=rec["run_id"])
    if row["window_ids"] is None:
        row["window_ids"] = []                       # NOT NULL in the table
    return {c: row[c] for c in TABLE_COLUMNS}


def _sql_value(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):  # noqa: UP038 - py3.9 on the node
        return repr(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return "'" + json.dumps(v).replace("'", "''") + "'"


def insert_sql(row: dict) -> str:
    """An INSERT for one row (paste into Adminer). Arrays go in as text[] / uuid[]."""
    vals = []
    for c in TABLE_COLUMNS:
        v = row[c]
        if c in ("window_ids", "control_instance_ids"):
            kind = "uuid[]" if c == "window_ids" else "text[]"
            vals.append("NULL" if v is None else
                        f"ARRAY[{', '.join(_sql_value(x) for x in v)}]::{kind}")
        else:
            vals.append(_sql_value(v))
    cols = ", ".join(f'"{c}"' for c in TABLE_COLUMNS)
    return f'INSERT INTO "public"."run_records" ({cols}) VALUES ({", ".join(vals)});'


def cmd_rows(args) -> int:
    paths: list[Path] = []
    for p in map(Path, args.paths):
        paths += sorted(p.rglob("record.json")) if p.is_dir() else [p]
    seen: set[str] = set()
    rows = []
    for p in paths:
        rec = _read(p)
        if not rec or rec.get("run_id") in seen:
            continue
        seen.add(rec["run_id"])
        rows.append(store_row(rec))
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    if args.sql:
        Path(args.sql).write_text("".join(insert_sql(r) + "\n" for r in rows))
    off = sorted({r["state"] for r in rows} - set(TABLE_STATES))
    print(f"{len(rows)} rows -> {args.out}" + (f" (+ {args.sql})" if args.sql else ""))
    if off:
        print(f"note: state {off} not in run_records' CHECK yet; inserts will fail until it is added",
              file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="write_record.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch")
    s.add_argument("--config", required=True)
    s.add_argument("--dest", required=True)
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("init")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("measure")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_measure)
    s = sub.add_parser("finalize")
    s.add_argument("dir")
    s.add_argument("--image")
    s.add_argument("--results-root", default=RESULTS_ROOT)
    s.add_argument("--no-health", action="store_true",
                   help="the node has no health checker: record health as unknown, not unhealthy")
    s.set_defaults(fn=cmd_finalize)
    s = sub.add_parser("rows")
    s.add_argument("paths", nargs="+", help="record.json files or folders to search")
    s.add_argument("--out", required=True)
    s.add_argument("--sql", help="also write INSERTs for run_records to this file")
    s.set_defaults(fn=cmd_rows)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
