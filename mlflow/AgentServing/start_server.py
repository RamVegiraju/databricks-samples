import os
import sys

# Ensure local imports work whether executed from this directory or repo root.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import agent  # noqa: F401 - Import registers @invoke/@stream functions
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Optional, but useful for mapping traces back to commit SHA.
setup_mlflow_git_based_version_tracking()


def main() -> None:
    # Use CLI args for port/workers/reload:
    # python3 start_server.py --reload --port 8000 --workers 1
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()