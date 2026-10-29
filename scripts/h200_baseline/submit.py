#!/usr/bin/env python3
"""Submit K2.6 baseline runs (ENG-95) through deploy/h200/gitm-submit.

Each submit is one run: one job, one fresh engine, one timed decode. The job
carries this folder's scripts and the gitm source at --gitm-ref with it (the
repo is private, so pip can't fetch it), and the host runs ``write_record.py
finalize`` after gitm-submit's node-health post-check, so the record holds the
health result.

    py scripts/h200_baseline/submit.py --count 1 --dry-run
    py scripts/h200_baseline/submit.py --count 7
    py scripts/h200_baseline/submit.py --count 7 --no-health   # nodes without gitm-node-health

The node-health checker is shipped from --health-ref too, until the nodes'
installed copy catches up with main (--node-health uses the installed one).

SLURM_JWT / SLURM_API come from the environment, else from ``KEY=VALUE`` lines
in the repo's .env.
"""

from __future__ import annotations

import argparse
import base64
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRIPTS = ("k26_h200.json", "measure_once.py", "run_one.sh", "write_record.py")
FINALIZE = ('python3 "$OUT_HOST/h200_baseline/write_record.py" finalize "$OUT_HOST" '
            '--image "$IMAGE"{flags} || true\n')
#: The hard stop #189 put in front of every job. Until the nodes have the
#: checker installed, --no-health drops just this; the pre/post checks below it
#: are already skipped when the checker isn't there.
HEALTH_GATE = ('if [ ! -x "$HEALTH_BIN" ]; then\n'
               '  echo "gitm-submit: required node-health checker is not executable: $HEALTH_BIN" >&2\n'
               '  exit 1\n'
               'fi\n')


def load_env() -> None:
    try:
        lines = (REPO / ".env").read_text().splitlines()
    except OSError:
        return
    for line in lines:
        m = re.fullmatch(r"\s*(?:export\s+)?(SLURM_JWT|SLURM_API)=(\S+)\s*", line)
        if m and not os.environ.get(m.group(1)):
            os.environ[m.group(1)] = m.group(2)


def load_gitm_submit():
    path = REPO / "deploy" / "h200" / "gitm-submit"
    loader = importlib.machinery.SourceFileLoader("gitm_submit", str(path))
    spec = importlib.util.spec_from_loader("gitm_submit", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def payload() -> str:
    """Shell that recreates this folder at /out/h200_baseline, then runs one baseline run."""
    lines = ["set -euo pipefail", "mkdir -p /out/h200_baseline"]
    for name in SCRIPTS:
        b64 = base64.b64encode((HERE / name).read_bytes()).decode()
        lines.append(f"echo {b64} | base64 -d > /out/h200_baseline/{name}")
    lines.append("bash /out/h200_baseline/run_one.sh")
    return "\n".join(lines)


def with_finalize(script: str, no_health: bool = False) -> str:
    """Run finalize on the host after the health post-check, before the job exits."""
    tail = '\nexit "$rc"\n'
    if not script.endswith(tail):
        sys.exit("submit.py: gitm-submit's batch script changed shape; update with_finalize()")
    finalize = FINALIZE.format(flags=" --no-health" if no_health else "")
    script = script[: -len(tail)] + "\n" + finalize + tail.lstrip("\n")
    # An unhealthy post-check moves the run to results/discarded/ and exits
    # before the line above. Record it first, so a discarded attempt still has
    # its measurement and failed signals (state invalid) instead of nothing.
    gate = '    "$HEALTH_BIN" gate --window "$NH_WINDOW"'
    at = script.rfind(gate)
    if at == -1 or at < script.find("post --job"):
        sys.exit("submit.py: gitm-submit's post-check changed shape; update with_finalize()")
    return script[:at] + "    " + finalize + script[at:]


def gitm_source(ref: str) -> bytes:
    """The ``gitm`` package at ``ref``, as a .tar.gz, from the local clone."""
    try:
        return subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", ref, "gitm"],
                              check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"submit.py: can't archive gitm at {ref} ({e.stderr.decode().strip()}); "
                 "fetch that commit first")


def with_shipped_gitm(script: str, ref: str) -> str:
    """Use gitm from the job itself instead of pip-installing it from GitHub.

    GitM-Labs/runtime is private, so gitm-submit's ``pip install`` of its archive
    URL 404s and the job dies. The host side of the job writes the source to
    the results folder (mounted at /out), and the container puts it first on
    PYTHONPATH. It goes in the batch script, which can hold megabytes, not the
    container command line, which caps one argument at 128 KB."""
    m = re.search(r'pip install -q "gitm-labs\[nvidia\] @ [^"]+"', script)
    if not m or "\nsrun --container-image=" not in script:
        sys.exit("submit.py: gitm-submit's install step changed shape; update with_shipped_gitm()")
    use = ("mkdir -p /out/gitm_src && tar -xzf /out/gitm_src.tgz -C /out/gitm_src && "
           "export PYTHONPATH=/out/gitm_src && "
           "python3 -c \"import gitm; print('gitm', gitm.__version__, gitm.__file__)\"")
    script = script.replace(m.group(0), use)
    b64 = base64.encodebytes(gitm_source(ref)).decode()
    ship = f"# gitm @ {ref}, shipped with the job\nbase64 -d > \"$OUT_HOST/gitm_src.tgz\" <<'GITM_SRC'\n{b64}GITM_SRC\n"
    return script.replace("\nsrun --container-image=", "\n" + ship + "srun --container-image=", 1)


def health_source(ref: str) -> tuple[bytes, str]:
    """``deploy/h200/gitm-node-health`` at ``ref`` and its git blob id."""
    spec = f"{ref}:deploy/h200/gitm-node-health"
    try:
        blob = subprocess.run(["git", "-C", str(REPO), "rev-parse", spec],
                              check=True, capture_output=True, text=True).stdout.strip()
        body = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", blob],
                              check=True, capture_output=True).stdout
    except subprocess.CalledProcessError:
        sys.exit(f"submit.py: can't read {spec}; fetch that commit first")
    return body, blob


def with_shipped_health(script: str, ref: str) -> str:
    """Run the node-health checker from ``ref`` instead of the node's installed copy.

    Only root can update /usr/local/bin/gitm-node-health, and the nodes can lag
    main (e.g. #222's power_cap fix). The job writes the checker into its own
    results folder and points HEALTH_BIN at it; the pre/sample/post/summarize/gate
    steps are gitm-submit's, unchanged. The file keeps its name because the
    sampler finds its own process by it. Kernel-log reads still go through the
    node's sudo helper, as with the installed copy."""
    anchor = 'if [ ! -x "$HEALTH_BIN" ]; then\n'
    out_host = 'OUT_HOST="$RESULTS/$SLURM_JOB_ID"'
    at = script.find(anchor)
    if at == -1 or "HEALTH_BIN=$(command -v gitm-node-health" not in script[:at] or script.find(out_host) > at:
        sys.exit("submit.py: gitm-submit's health setup changed shape; update with_shipped_health()")
    body, blob = health_source(ref)
    info = json.dumps({"source": "deploy/h200/gitm-node-health", "ref": ref, "blob": blob,
                       "path": "nh/gitm-node-health"})
    b64 = base64.encodebytes(body).decode()
    ship = (f"# gitm-node-health @ {ref} (blob {blob}), shipped with the job\n"
            'mkdir -p "$OUT_HOST/nh"\n'
            f"base64 -d > \"$OUT_HOST/nh/gitm-node-health\" <<'GITM_NH'\n{b64}GITM_NH\n"
            'chmod 755 "$OUT_HOST/nh/gitm-node-health"\n'
            f"cat > \"$OUT_HOST/nh/checker.json\" <<'GITM_NH_INFO'\n{info}\nGITM_NH_INFO\n"
            'HEALTH_BIN="$OUT_HOST/nh/gitm-node-health"\n'
            'echo "submit.py: node health from $HEALTH_BIN (' + f"{ref[:12]}, blob {blob[:12]}" + ')"\n')
    return script[:at] + ship + script[at:]


def without_health_gate(script: str) -> str:
    """Let the job run on a node with no health checker; it says so in its log."""
    if HEALTH_GATE not in script:
        sys.exit("submit.py: gitm-submit's health gate changed shape; update without_health_gate()")
    return script.replace(HEALTH_GATE, 'if [ ! -x "$HEALTH_BIN" ]; then\n'
                          '  echo "submit.py --no-health: node-health checker not installed; '
                          'health recorded as unknown" >&2\nfi\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--count", type=int, default=1, help="runs (jobs) to queue")
    ap.add_argument("--start", type=int, default=1, help="first job-name number")
    ap.add_argument("--gitm-ref", default="bbc6c07587cece4dfa42ef6a282bcb253fc2a942")
    ap.add_argument("--image", help="container image (default: the node's pre-loaded vLLM image)")
    ap.add_argument("--max-hours", type=float, default=1.5)
    ap.add_argument("--no-health", action="store_true",
                    help="run without the node-health gate (only while the nodes lack the checker)")
    ap.add_argument("--health-ref", default="48ccd329bb23da59ea7b713e4dba99eed7fd9e6a",
                    help="ship gitm-node-health from this ref (default: #222, the power_cap fix)")
    ap.add_argument("--node-health", action="store_true",
                    help="use the node's installed gitm-node-health instead of shipping one")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.no_health and not args.node_health:
        args.node_health = True          # nothing to ship a checker for

    load_env()
    cfg = json.loads((HERE / "k26_h200.json").read_text())
    gs = load_gitm_submit()
    original = gs.build_script

    def build(a):
        script = with_shipped_gitm(with_finalize(original(a), args.no_health), args.gitm_ref)
        if args.no_health:
            return without_health_gate(script)
        return script if args.node_health else with_shipped_health(script, args.health_ref)

    gs.build_script = build
    for i in range(args.start, args.start + args.count):
        ns = argparse.Namespace(
            name=f"k26-floor-{i:02d}", model=cfg["model"], max_hours=args.max_hours,
            vllm_args=None, capture_args=None, image=args.image, gitm_ref=args.gitm_ref,
            env=[f"GITM_REF={args.gitm_ref}"], node=cfg["node"], partition="gpu",
            results_dir="/var/lib/gitm/results", dry_run=args.dry_run,
            command=["bash", "-c", payload()])
        gs.cmd_run(ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
