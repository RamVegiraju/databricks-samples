from mlflow.genai.agent_server import AgentServer
import agent # This import triggers the @invoke registration

# Initialize the server with a name
agent_server = AgentServer("LangChainToolAgent")

# Expose the FastAPI app object for Uvicorn
app = agent_server.app

if __name__ == "__main__":
    # Run the server
    agent_server.run(app_import_string="start_server:app", port=8000)