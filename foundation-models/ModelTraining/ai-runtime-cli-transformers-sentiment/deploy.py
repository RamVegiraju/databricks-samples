"""Step 2 — deploy (decoupled from training, runs locally on your machine).

Takes the model artifacts that train.py registered to Unity Catalog and serves
them on a Databricks Model Serving endpoint. This is a control-plane operation
(just SDK calls) — it needs NO GPU and no cluster, so it runs locally against
your Databricks CLI profile instead of inside the training job. That keeps the
expensive A10 dedicated to training only.

Resolves the exact version behind the registry alias train.py set (--alias /
MODEL_ALIAS), creates or updates the endpoint, then polls until it is READY,
printing endpoint state every --interval seconds (default 90s). Endpoint builds
for transformers models can take ~10-20 min on the first create.

The fine-tuned DistilBERT is small, so it serves on CPU (workload_type=CPU) at
the smallest workload size with scale-to-zero — cheapest footprint for a
prototype (the first request after idle pays a cold start).

    python deploy.py -p my-workspace                       # deploy @champion
    python deploy.py -p my-workspace --version 3           # pin an exact version
    python deploy.py -p my-workspace --no-scale-to-zero    # keep always-on
"""

import argparse
import os
import sys
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--profile",
                   default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT"),
                   help="Databricks CLI profile (OAuth) to authenticate with.")
    p.add_argument("--model-name",
                   default=os.environ.get("MODEL_NAME", "custom_ml.models.distilbert_sentiment"),
                   help="Unity Catalog model: <catalog>.<schema>.<model>.")
    p.add_argument("--alias", default=os.environ.get("MODEL_ALIAS", "champion"),
                   help="Registry alias set by train.py; resolves to the exact version to serve.")
    p.add_argument("--version", default=None,
                   help="Serve an exact version instead of resolving --alias.")
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT_NAME", "air-transformers-sentiment"),
                   help="Serving endpoint name.")
    p.add_argument("--workload-size", default=os.environ.get("WORKLOAD_SIZE", "Small"),
                   choices=["Small", "Medium", "Large"])
    p.add_argument("--workload-type", default=os.environ.get("WORKLOAD_TYPE", "CPU"),
                   help="CPU (default; the model is small), or a GPU type e.g. GPU_SMALL.")
    p.add_argument("--scale-to-zero", dest="scale_to_zero",
                   default=_env_bool("SCALE_TO_ZERO", True), action="store_true",
                   help="Scale to zero when idle (default).")
    p.add_argument("--no-scale-to-zero", dest="scale_to_zero", action="store_false")
    p.add_argument("--timeout-min", type=int, default=40,
                   help="How long to wait for the endpoint to become READY.")
    p.add_argument("--interval", type=int, default=90,
                   help="Seconds between readiness polls.")
    return p.parse_args()


def version_for_alias(w: WorkspaceClient, model_name: str, alias: str) -> str:
    # Resolve via the authenticated WorkspaceClient (Unity Catalog), not an
    # MlflowClient — the latter uses a separate auth path that ignores the CLI
    # profile and 401s when deploy.py runs locally.
    return str(w.model_versions.get_by_alias(full_name=model_name, alias=alias).version)


def poll_until_ready(w: WorkspaceClient, name: str, timeout_min: int,
                     interval: int) -> None:
    deadline = time.monotonic() + timeout_min * 60
    while time.monotonic() < deadline:
        try:
            ep = w.serving_endpoints.get(name)
        except ResourceDoesNotExist:
            raise SystemExit(
                f"Endpoint '{name}' no longer exists (deleted during build?). Aborting.")
        state = ep.state
        ready = state.ready.value if state and state.ready else "?"
        update = state.config_update.value if state and state.config_update else "?"
        # Surface the served-entity build message ("Container creation pending",
        # "Deploying served entities", etc.) so long builds are legible.
        detail = ""
        pending = ep.pending_config.served_entities if ep.pending_config else None
        if pending and pending[0].state and pending[0].state.deployment_state_message:
            detail = f" :: {pending[0].state.deployment_state_message}"
        print(f"  [{time.strftime('%H:%M:%S')}] ready={ready} config_update={update}{detail}")
        if ready == "READY" and update != "IN_PROGRESS":
            return
        if update == "UPDATE_FAILED":
            raise SystemExit(
                f"Endpoint {name} update FAILED. Check the serving logs in the UI.")
        time.sleep(interval)
    raise SystemExit(f"Timed out after {timeout_min} min waiting for {name}.")


def main() -> None:
    # Stream progress live even when stdout is redirected/piped (not a TTY),
    # so the every-90s poll lines show up in tee'd logs and CI as they happen.
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    w = WorkspaceClient(profile=args.profile)

    version = args.version or version_for_alias(w, args.model_name, args.alias)
    src = f"v{version}" if args.version else f"v{version} (@{args.alias})"
    print(f"Deploying {args.model_name} {src} -> endpoint '{args.endpoint}' "
          f"({args.workload_type}/{args.workload_size}, scale_to_zero={args.scale_to_zero})")

    served = ServedEntityInput(
        entity_name=args.model_name,
        entity_version=str(version),
        workload_size=args.workload_size,
        workload_type=ServingModelWorkloadType(args.workload_type),
        scale_to_zero_enabled=args.scale_to_zero,
    )

    exists = True
    try:
        w.serving_endpoints.get(args.endpoint)
    except ResourceDoesNotExist:
        exists = False

    if exists:
        print("Endpoint exists — updating served model to this version.")
        w.serving_endpoints.update_config(name=args.endpoint, served_entities=[served])
    else:
        print("Creating endpoint.")
        w.serving_endpoints.create(
            name=args.endpoint,
            config=EndpointCoreConfigInput(name=args.endpoint, served_entities=[served]),
        )

    print(f"Waiting for endpoint to become READY (polling every {args.interval}s; "
          f"first build can take ~10-20 min)...")
    poll_until_ready(w, args.endpoint, args.timeout_min, args.interval)

    host = w.config.host.rstrip("/")
    print("\nEndpoint READY.")
    print(f"  Name:  {args.endpoint}")
    print(f"  URL:   {host}/serving-endpoints/{args.endpoint}/invocations")
    print(f"  UI:    {host}/ml/endpoints/{args.endpoint}")
    print(f"\nInvoke it:  python invoke_endpoint.py -p {args.profile} --endpoint {args.endpoint}")


if __name__ == "__main__":
    main()
