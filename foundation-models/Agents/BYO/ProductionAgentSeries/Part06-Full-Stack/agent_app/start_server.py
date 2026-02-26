import os
import sys

# Ensure agent_app/ is on the path so relative imports within agent/ work.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agent import server  # noqa: F401 — import registers @invoke/@stream handlers
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Maps traces to the git commit SHA they were generated from (optional).
# Silently skipped if the deployed environment has no .git directory.
try:
    setup_mlflow_git_based_version_tracking()
except Exception:
    pass


def main() -> None:
    # Accepts CLI args: --port, --workers, --reload
    # Defaults to port 8000, which is the Databricks Apps listener port.
    # Example: python start_server.py --reload --port 8000
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
