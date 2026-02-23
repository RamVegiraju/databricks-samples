import os
import mlflow

mlflow.set_tracking_uri("databricks")

exp_path = os.getenv("DBX_EXPERIMENT_PATH")
if not exp_path:
    username = os.getenv("DATABRICKS_USERNAME") or os.getenv("DBX_USERNAME")
    if not username:
        raise ValueError(
            "Set DBX_EXPERIMENT_PATH (e.g. /Users/<you>/ResponsesAgentInterface) "
            "or set DATABRICKS_USERNAME/DBX_USERNAME."
        )
    exp_path = f"/Users/{username}/ResponsesAgentInterface"

exp = mlflow.set_experiment(exp_path)
print("TRACKING_URI:", mlflow.get_tracking_uri())
print("EXPERIMENT:", exp.name, f"(id={exp.experiment_id})")

with mlflow.start_run() as run:
    info = mlflow.pyfunc.log_model(
        artifact_path="agent_model",
        python_model="agent.py",
    )
    print("RUN_ID:", run.info.run_id)
    print("MODEL_URI:", info.model_uri)
    print(f'SERVE_CMD: mlflow models serve -m "{info.model_uri}" -p 8000 --env-manager local')