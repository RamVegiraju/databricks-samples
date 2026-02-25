import os
import mlflow

# 1) Ensure tracking is Databricks + experiment set
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

# 2) Enable OpenAI instrumentation (MLflow will create traces/spans)
mlflow.openai.autolog()  # if available in your mlflow version

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=os.environ["DATABRICKS_BASE_URL"],
)

resp = client.chat.completions.create(
  model="databricks-claude-sonnet-4-6",
  messages=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is Databricks?"},
  ],
  max_tokens=256,
)

print(resp.choices[0].message.content)