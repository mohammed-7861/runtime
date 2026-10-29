"""The ENG-95 baseline run record: run id, ENG-72 fields, health folding, store rows."""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / "scripts" / "h200_baseline"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wr = _load("write_record")
submit = _load("submit")
CFG = json.loads((DIR / "k26_h200.json").read_text())


def _measure(tokens: int = 94011, expected: int = 94011) -> dict:
    stats = {"mean": 0.1, "median": 0.1, "p95": 0.2, "p99": 0.3}
    return {
        "measurement": {"throughput_tok_s": 1000.0, "decode_s": 94.0, "generated_tokens": tokens,
                        "expected_tokens": expected, "requests": 256,
                        "warmup": {"throughput_tok_s": 900.0}, "ttft_s": stats, "tpot_s": stats,
                        "engine_build_s": 600.0},
        "traffic": {"requests": 256, "skipped": {"over_max_model_len": 10}},
        "engine_version": "0.x", "gpu_name": "NVIDIA H200", "gpu_count": 8,
        "driver_version": "570.0", "cuda_version": "12.8", "nccl_version": "2.26.2",
    }


def _inputs() -> dict:
    return wr.build_inputs(CFG, {"SLURM_JOB_ID": "42", "SLURM_JOB_USER": "mohammed"},
                           "gitmachine-2", {"run_one.sh": "abc"})


def test_run_id_follows_eng76():
    when = datetime(2026, 10, 28, 14, 15, 0, tzinfo=timezone.utc)
    rid = wr.mint_run_id("moonshotai/Kimi-K2.6", "gitmachine-2.internal", when)
    assert re.fullmatch(r"20261028T141500Z-kimi-k26-gm2-[0-9a-f]{4}", rid)


def test_baseline_record_has_every_eng72_field_and_no_lever():
    rec = wr.build_record(_inputs(), _measure())
    assert set(wr.RECORD_FIELDS) <= rec.keys()
    assert rec["state"] == "baseline"
    assert rec["iterations"] == 1 and rec["wall_time_s"] == 94.0
    for f in ("beat_baseline", "delta", "lever_id", "knob_values", "canary_instance",
              "noise_floor", "threshold", "min_effect", "policy_id", "candidate_config"):
        assert rec[f] is None, f
    assert rec["control_instances"] == [rec["run_id"]]
    assert rec["baseline_config"]["engine_kwargs"]["tensor_parallel_size"] == 8
    assert rec["slo_guard"]["ttft_p95_s"] == 0.2
    assert any(d["what"] == "traffic" for d in rec["degradations"])


@pytest.mark.parametrize("measure", [None, _measure(tokens=90000)])
def test_failed_or_short_measurement_is_invalid(measure):
    assert wr.build_record(_inputs(), measure)["state"] == "invalid"


def test_finalize_folds_health_and_marks_unhealthy_invalid(tmp_path, capsys):
    inputs = _inputs()
    wr._write(tmp_path / "inputs.json", inputs)
    wr._write(tmp_path / "record.json", wr.build_record(inputs, _measure()))
    wr._write(tmp_path / "node_health_window.json",
              {"healthy": False, "failed_signals": ["thermal"], "job_id": "42", "node": "gitmachine-2"})
    wr.main(["finalize", str(tmp_path), "--image", "vllm/vllm-openai:latest",
             "--results-root", str(tmp_path / "missing")])
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["node_healthy_for_window"] is False and rec["state"] == "invalid"
    assert rec["gitm.node_health"]["failed_signals"] == ["thermal"]
    assert any(d["what"] == "storage" for d in rec["degradations"])
    assert json.loads((tmp_path / "inputs.json").read_text())["image"]["ref"] == "vllm/vllm-openai:latest"
    assert '"state": "invalid"' in capsys.readouterr().out


def test_finalize_copies_to_durable_root_when_mounted(tmp_path):
    run, root = tmp_path / "run", tmp_path / "gitm-runs"
    run.mkdir()
    root.mkdir()
    inputs = _inputs()
    wr._write(run / "inputs.json", inputs)
    wr._write(run / "record.json", wr.build_record(inputs, _measure()))
    wr._write(run / "node_health_window.json", {"healthy": True, "failed_signals": []})
    wr.main(["finalize", str(run), "--results-root", str(root)])
    dest = root / inputs["run_id"]
    rec = json.loads((dest / "record.json").read_text())
    assert rec["state"] == "baseline" and rec["node_healthy_for_window"] is True
    assert (dest / "inputs.json").is_file()


def test_finalize_without_a_record_writes_an_invalid_one(tmp_path):
    wr._write(tmp_path / "inputs.json", _inputs())
    wr.main(["finalize", str(tmp_path), "--results-root", str(tmp_path / "missing")])
    assert json.loads((tmp_path / "record.json").read_text())["state"] == "invalid"


def test_rows_map_every_field_once_per_run(tmp_path):
    for i in range(2):
        d = tmp_path / str(i)
        d.mkdir()
        inputs = _inputs()
        wr._write(d / "inputs.json", inputs)
        wr._write(d / "record.json", wr.build_record(inputs, _measure()))
    out, sql = tmp_path / "rows.jsonl", tmp_path / "rows.sql"
    wr.main(["rows", str(tmp_path), "--out", str(out), "--sql", str(sql)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert list(row) == list(wr.TABLE_COLUMNS)          # exactly run_records' columns
        assert row["control_instance_ids"] == [row["run_id"]] and row["window_ids"] == []
        assert row["cuda_version"] == "12.8" and row["nccl_version"] == "2.26.2"
    inserts = sql.read_text().splitlines()
    assert len(inserts) == 2 and all(s.startswith('INSERT INTO "public"."run_records"') for s in inserts)
    assert "ARRAY[]::uuid[]" in inserts[0]


def test_every_record_field_maps_to_a_table_column():
    mapped = set(wr.STORE_COLUMNS.values()) | set(wr.VERSION_FIELDS) | {"run_id"}
    assert mapped == set(wr.TABLE_COLUMNS)


def test_finalize_without_checker_leaves_health_unknown_not_invalid(tmp_path):
    inputs = _inputs()
    wr._write(tmp_path / "inputs.json", inputs)
    wr._write(tmp_path / "record.json", wr.build_record(inputs, _measure()))
    wr.main(["finalize", str(tmp_path), "--no-health", "--results-root", str(tmp_path / "missing")])
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["node_healthy_for_window"] is None and rec["state"] == "baseline"
    assert any(d["what"] == "node_health" for d in rec["degradations"])


def _script() -> str:
    gs = submit.load_gitm_submit()
    return gs.build_script(type("A", (), {
        "gitm_ref": "x", "command": ["true"], "capture_args": None, "vllm_args": None,
        "env": [], "image": None, "model": CFG["model"]})())


def test_submit_runs_finalize_after_the_health_check():
    patched = submit.with_finalize(_script())
    assert patched.endswith('exit "$rc"\n')
    assert patched.index("write_record.py\" finalize") > patched.index('"$HEALTH_BIN" summarize')
    assert "--no-health" not in patched


def test_unhealthy_run_is_recorded_before_the_gate_discards_it():
    patched = submit.with_finalize(_script())
    post = patched.index('"$HEALTH_BIN" post')
    gate = patched.index('"$HEALTH_BIN" gate', post)
    assert post < patched.index("write_record.py\" finalize", post) < gate
    assert patched.count("write_record.py\" finalize") == 2   # unhealthy exit + normal exit


def test_gitm_ships_with_the_job_instead_of_pip(monkeypatch):
    monkeypatch.setattr(submit, "gitm_source", lambda ref: b"tgz-bytes")
    patched = submit.with_shipped_gitm(submit.with_finalize(_script()), "abc123")
    assert "pip install" not in patched and "PYTHONPATH=/out/gitm_src" in patched
    # written on the host before the container starts, not inside its command line
    assert patched.index("<<'GITM_SRC'") < patched.index("\nsrun --container-image=")
    assert patched.index("<<'GITM_SRC'") > patched.index('OUT_HOST="$RESULTS/$SLURM_JOB_ID"')


def test_shipped_checker_replaces_only_the_node_copy(monkeypatch):
    monkeypatch.setattr(submit, "health_source", lambda ref: (b"#!/usr/bin/env python3\n", "b" * 40))
    script = submit.with_finalize(_script())
    patched = submit.with_shipped_health(script, "abc123")
    ship = patched.index("<<'GITM_NH'")
    assert patched.index('OUT_HOST="$RESULTS/$SLURM_JOB_ID"') < ship
    assert patched.index('HEALTH_BIN=$(command -v gitm-node-health') < ship
    override = patched.index('HEALTH_BIN="$OUT_HOST/nh/gitm-node-health"')
    assert ship < override < patched.index('"$HEALTH_BIN" pre')
    # one block inserted before the executable check; the rest of gitm-submit's script is untouched
    at = script.index('if [ ! -x "$HEALTH_BIN" ]; then')
    added = len(patched) - len(script)
    assert patched[:at] == script[:at] and patched[at + added:] == script[at:]


def test_health_source_is_the_file_in_git():
    body, blob = submit.health_source("48ccd329bb23da59ea7b713e4dba99eed7fd9e6a")
    assert blob.startswith("467eca8") and b"check_throttle" in body


def test_finalize_records_which_checker_ran(tmp_path):
    inputs = _inputs()
    wr._write(tmp_path / "inputs.json", inputs)
    wr._write(tmp_path / "record.json", wr.build_record(inputs, _measure()))
    wr._write(tmp_path / "node_health_window.json", {"healthy": True, "failed_signals": []})
    (tmp_path / "nh").mkdir()
    wr._write(tmp_path / "nh" / "checker.json", {"ref": "abc", "blob": "467eca8"})
    wr.main(["finalize", str(tmp_path), "--results-root", str(tmp_path / "missing")])
    rec = json.loads((tmp_path / "record.json").read_text())
    assert rec["gitm.node_health"]["checker"]["blob"] == "467eca8"
    assert rec["state"] == "baseline"


def test_no_health_drops_only_the_hard_stop():
    script = _script()
    patched = submit.without_health_gate(submit.with_finalize(script, no_health=True))
    assert "required node-health checker is not executable" in script
    assert "required node-health checker is not executable" not in patched
    assert patched.count('"$HEALTH_BIN" pre') == script.count('"$HEALTH_BIN" pre')  # still runs if installed
    assert "finalize \"$OUT_HOST\" --image \"$IMAGE\" --no-health" in patched
