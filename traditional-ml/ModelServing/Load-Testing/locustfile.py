# Locust load test for a Databricks Model Serving endpoint.
# Config comes from env vars (DATABRICKS_TOKEN, SERVING_ENDPOINT_NAME);
# the workspace host is passed via the --host CLI flag.

import os

from locust import HttpUser, task, between

TOKEN = os.environ["DATABRICKS_TOKEN"]
ENDPOINT_NAME = os.environ["SERVING_ENDPOINT_NAME"]
INVOCATION_PATH = f"/serving-endpoints/{ENDPOINT_NAME}/invocations"

# Scoring payload matching the model's input schema.
PAYLOAD = {
    "dataframe_split": {
        "columns": ["feature_1", "feature_2", "feature_3"],
        "data": [[0.15, 3.8, 7.0]],
    }
}


class ServingUser(HttpUser):
    wait_time = between(0.1, 0.5)  # think-time between requests per user

    def on_start(self):
        # Set the auth header once on the session.
        self.client.headers.update({"Authorization": f"Bearer {TOKEN}"})

    @task
    def score(self):
        self.client.post(INVOCATION_PATH, json=PAYLOAD, name="score")
