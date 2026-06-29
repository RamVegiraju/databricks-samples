#!/usr/bin/env python3
"""Poll an AI Runtime CLI run to completion, then print its Job and MLflow links.

`air` submits each workload as a one-time Databricks Job run and tracks it in an
MLflow experiment. This helper waits for that run to finish and prints clickable
links so you can inspect metrics, the logged model, and the job run page.

Auth reuses your Databricks CLI profile (the same one passed to `air`), so no
extra credentials are needed. It shells out to the `databricks` CLI only.

Usage:
    python watch_run.py <job_run_id> -p my-workspace
    python watch_run.py <job_run_id> -p my-workspace --experiment ai-runtime-cli-intro
"""

import argparse
import json
import subprocess
import sys
import time
from urllib.parse import urlparse

TERMINAL = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}


def db(args: list[str], profile: str) -> dict:
    """Call `databricks <args> -p <profile>` and parse JSON stdout."""
    res = subprocess.run(
        ["databricks", *args, "-p", profile],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"databricks {' '.join(args)} failed:\n{res.stderr.strip()}")
    return json.loads(res.stdout)


def get_run(run_id: str, profile: str) -> dict:
    return db(["jobs", "get-run", run_id], profile)


def resolve_mlflow_run(run_id: str, experiment: str, profile: str) -> tuple[str | None, str | None]:
    """Return (experiment_id, mlflow_run_id) for the given job run, if found."""
    me = db(["current-user", "me"], profile)
    email = me["userName"]
    exp_path = experiment if experiment.startswith("/") else f"/Users/{email}/{experiment}"

    exp = db(
        ["api", "get", "/api/2.0/mlflow/experiments/get-by-name",
         "--json", json.dumps({"experiment_name": exp_path})],
        profile,
    )
    experiment_id = exp["experiment"]["experiment_id"]

    # air tags each MLflow run with the originating job run id.
    search = db(
        ["api", "post", "/api/2.0/mlflow/runs/search", "--json", json.dumps({
            "experiment_ids": [experiment_id],
            "filter": f"tags.`mlflow.databricks.jobRunID` = '{run_id}'",
            "max_results": 1,
        })],
        profile,
    )
    runs = search.get("runs", [])
    mlflow_run_id = runs[0]["info"]["run_id"] if runs else None
    return experiment_id, mlflow_run_id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id", help="Job run ID printed by `air run`")
    p.add_argument("-p", "--profile", default="my-workspace", help="Databricks CLI profile")
    p.add_argument("--experiment", default="ai-runtime-cli-intro",
                   help="MLflow experiment name (matches experiment_name in train.yaml)")
    p.add_argument("--interval", type=int, default=15, help="Poll interval in seconds")
    p.add_argument("--timeout", type=int, default=1800, help="Give up after this many seconds")
    args = p.parse_args()

    print(f"Watching run {args.run_id} (profile: {args.profile})...")
    start = time.time()
    run = get_run(args.run_id, args.profile)
    while True:
        state = run.get("state", {})
        life = state.get("life_cycle_state", "?")
        result = state.get("result_state", "")
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>4}s] {life}{(' / ' + result) if result else ''}")
        if life in TERMINAL:
            break
        if time.time() - start > args.timeout:
            print(f"Timed out after {args.timeout}s; run is still {life}.", file=sys.stderr)
            return 2
        time.sleep(args.interval)
        run = get_run(args.run_id, args.profile)

    result = run.get("state", {}).get("result_state", "UNKNOWN")
    duration_s = run.get("run_duration", 0) // 1000
    job_url = run.get("run_page_url", "")
    host = f"{urlparse(job_url).scheme}://{urlparse(job_url).netloc}" if job_url else ""

    # Never let an MLflow lookup failure hide the Job link, which prints below.
    exp_id = mlflow_run_id = None
    try:
        exp_id, mlflow_run_id = resolve_mlflow_run(args.run_id, args.experiment, args.profile)
    except Exception as e:  # noqa: BLE001 - best-effort link resolution
        print(f"(could not resolve MLflow links: {e})", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Result:   {result}")
    print(f"Duration: {duration_s // 60}m {duration_s % 60}s")
    print("=" * 60)
    print(f"Job run:           {job_url}")
    if exp_id:
        print(f"MLflow experiment: {host}/ml/experiments/{exp_id}")
    if exp_id and mlflow_run_id:
        print(f"MLflow run:        {host}/ml/experiments/{exp_id}/runs/{mlflow_run_id}")
    elif exp_id:
        print("MLflow run:        (no run tagged with this job run id yet)")
    print("=" * 60)

    return 0 if result == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
