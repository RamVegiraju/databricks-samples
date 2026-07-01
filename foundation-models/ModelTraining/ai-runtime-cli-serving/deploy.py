"""Deploy a Unity Catalog model version to a Databricks Model Serving endpoint.

Creates the endpoint if it doesn't exist, or updates it in place with the new
version. Everything is parameterized; the only required-in-practice input is the
model name (which defaults to the same one train.yaml registers).

    python deploy.py -p my-workspace                       # deploy latest version
    python deploy.py -p my-workspace --version 3           # pin a version
    python deploy.py -p my-workspace \
      --model-name custom_ml.models.air_mlp_classifier \
      --endpoint-name air-mlp-classifier --workload-size Small --workload-type CPU
"""

import argparse
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-p", "--profile", default="DEFAULT",
                   help="Databricks CLI profile (OAuth) to authenticate with.")
    p.add_argument("--model-name", default="custom_ml.models.air_mlp_classifier",
                   help="Unity Catalog model: <catalog>.<schema>.<model>.")
    p.add_argument("--version", default=None,
                   help="Model version to serve. Default: latest registered.")
    p.add_argument("--endpoint-name", default="air-mlp-classifier",
                   help="Serving endpoint name.")
    p.add_argument("--workload-size", default="Small",
                   choices=["Small", "Medium", "Large"])
    p.add_argument("--workload-type", default="CPU",
                   help="CPU, or a GPU type e.g. GPU_SMALL.")
    p.add_argument("--scale-to-zero", dest="scale_to_zero", action="store_true",
                   default=True, help="Scale to zero when idle (default).")
    p.add_argument("--no-scale-to-zero", dest="scale_to_zero",
                   action="store_false")
    p.add_argument("--timeout-min", type=int, default=40,
                   help="How long to wait for the endpoint to become READY.")
    p.add_argument("--interval", type=int, default=20,
                   help="Seconds between readiness polls.")
    return p.parse_args()


def latest_version(w: WorkspaceClient, model_name: str) -> str:
    versions = [int(v.version) for v in w.model_versions.list(model_name)]
    if not versions:
        raise SystemExit(f"No versions found for {model_name}. Train first.")
    return str(max(versions))


def poll_until_ready(w: WorkspaceClient, name: str, timeout_min: int,
                     interval: int) -> None:
    deadline = time.monotonic() + timeout_min * 60
    while time.monotonic() < deadline:
        try:
            ep = w.serving_endpoints.get(name)
        except ResourceDoesNotExist:
            raise SystemExit(
                f"Endpoint '{name}' no longer exists (deleted during build?). "
                f"Aborting.")
        ready = ep.state.ready.value if ep.state and ep.state.ready else "?"
        update = (ep.state.config_update.value
                  if ep.state and ep.state.config_update else "?")
        print(f"  state: ready={ready} config_update={update}")
        if ready == "READY" and update != "IN_PROGRESS":
            return
        if update == "UPDATE_FAILED":
            raise SystemExit(f"Endpoint {name} update FAILED. "
                             f"Check the serving logs in the UI.")
        time.sleep(interval)
    raise SystemExit(f"Timed out after {timeout_min} min waiting for {name}.")


def main() -> None:
    args = parse_args()
    w = WorkspaceClient(profile=args.profile)

    version = args.version or latest_version(w, args.model_name)
    print(f"Deploying {args.model_name} v{version} -> endpoint "
          f"'{args.endpoint_name}' ({args.workload_type}/{args.workload_size}, "
          f"scale_to_zero={args.scale_to_zero})")

    served = ServedEntityInput(
        entity_name=args.model_name,
        entity_version=version,
        workload_size=args.workload_size,
        workload_type=ServingModelWorkloadType(args.workload_type),
        scale_to_zero_enabled=args.scale_to_zero,
    )

    exists = True
    try:
        w.serving_endpoints.get(args.endpoint_name)
    except Exception:
        exists = False

    if exists:
        print("Endpoint exists — updating config to the new version.")
        w.serving_endpoints.update_config(
            name=args.endpoint_name, served_entities=[served])
    else:
        print("Creating endpoint.")
        w.serving_endpoints.create(
            name=args.endpoint_name,
            config=EndpointCoreConfigInput(
                name=args.endpoint_name, served_entities=[served]),
        )

    print("Waiting for endpoint to become READY (first build can take ~10-20 min)...")
    poll_until_ready(w, args.endpoint_name, args.timeout_min, args.interval)

    host = w.config.host.rstrip("/")
    print("\nEndpoint READY.")
    print(f"  Name:  {args.endpoint_name}")
    print(f"  URL:   {host}/serving-endpoints/{args.endpoint_name}/invocations")
    print(f"  UI:    {host}/ml/endpoints/{args.endpoint_name}")
    print(f"\nInvoke it:  python invoke.py -p {args.profile} "
          f"--endpoint-name {args.endpoint_name}")


if __name__ == "__main__":
    main()
