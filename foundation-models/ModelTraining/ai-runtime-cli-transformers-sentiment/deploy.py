"""Step 2 — deploy.

Resolves the model version behind the registry alias that step 1 just set
(--alias / MODEL_ALIAS) and creates or updates a Databricks Model Serving
endpoint pointing at it, then waits until the endpoint is ready. Runs inside the
same `air` job as training, using the job's ambient Databricks credentials.

The fine-tuned DistilBERT is small, so it serves on CPU: the smallest workload
size with scale-to-zero enabled — cheapest footprint for a prototype (the first
request after idle pays a cold start).
"""

import argparse
import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.tracking import MlflowClient


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name",
                   default=os.environ.get("MODEL_NAME", "main.default.distilbert_sentiment"))
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT_NAME", "air-transformers-sentiment"))
    p.add_argument("--alias", default=os.environ.get("MODEL_ALIAS", "champion"),
                   help="Registry alias set by train.py; resolves to the exact version to serve")
    p.add_argument("--workload-size", default=os.environ.get("WORKLOAD_SIZE", "Small"),
                   choices=["Small", "Medium", "Large"])
    p.add_argument("--scale-to-zero", dest="scale_to_zero",
                   default=_env_bool("SCALE_TO_ZERO", True), action="store_true")
    p.add_argument("--no-scale-to-zero", dest="scale_to_zero", action="store_false")
    return p.parse_args()


def version_for_alias(model_name: str, alias: str) -> str:
    client = MlflowClient(registry_uri="databricks-uc")
    mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    return mv.version


def main() -> None:
    args = parse_args()
    mlflow.set_registry_uri("databricks-uc")

    version = version_for_alias(args.model_name, args.alias)
    print(f"Deploying {args.model_name} v{version} (@{args.alias}) to endpoint '{args.endpoint}'...")

    w = WorkspaceClient()
    entity = ServedEntityInput(
        entity_name=args.model_name,
        entity_version=str(version),
        workload_size=args.workload_size,
        scale_to_zero_enabled=args.scale_to_zero,
    )

    existing = {e.name for e in w.serving_endpoints.list()}
    if args.endpoint in existing:
        print("Endpoint exists; updating served model (this can take several minutes)...")
        w.serving_endpoints.update_config_and_wait(name=args.endpoint, served_entities=[entity])
    else:
        print("Creating endpoint (this can take several minutes)...")
        w.serving_endpoints.create_and_wait(
            name=args.endpoint,
            config=EndpointCoreConfigInput(served_entities=[entity]),
        )

    print(f"Endpoint '{args.endpoint}' is ready, serving {args.model_name} v{version} (@{args.alias}).")


if __name__ == "__main__":
    main()
