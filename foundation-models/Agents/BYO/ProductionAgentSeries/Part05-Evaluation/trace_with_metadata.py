import os
import mlflow
from openai import OpenAI

# --- Setup ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

client = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=os.environ["DATABRICKS_BASE_URL"],
)

# --- Traced function with user/session metadata ---
# @mlflow.trace owns the trace lifecycle here — do NOT also enable
# mlflow.openai.autolog() in this file.  autolog creates its own top-level
# trace for each OpenAI call; when both are active, you end up with two
# separate traces per invocation and the metadata update only lands on
# the @mlflow.trace one, leaving the autolog trace without user/session
# columns in the Databricks UI.
#
# mlflow.trace.user  → dedicated "User" column in the Traces view
# mlflow.trace.session → dedicated "Session" column in the Traces view
# Both are queryable via:
#   mlflow.search_traces(filter_string="metadata.`mlflow.trace.user` = 'user-1'")
@mlflow.trace
def chat(user_message: str, user_id: str, session_id: str) -> str:
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )

    resp = client.chat.completions.create(
        model="databricks-claude-sonnet-4-6",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=256,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # Simulate multiple users across different sessions
    calls = [
        ("What is MLflow?",                      "user-1", "session-aaa"),
        ("Explain Databricks Unity Catalog.",    "user-2", "session-bbb"),
        ("What is a vector index?",              "user-1", "session-aaa"),
    ]

    for question, user_id, session_id in calls:
        answer = chat(question, user_id, session_id)
        print(f"[{user_id} | {session_id}] Q: {question}")
        print(f"  A: {answer}\n")
