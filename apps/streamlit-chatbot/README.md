# Streamlit Chatbot on Databricks Apps

A minimal streaming chatbot powered by `databricks-gpt-5-5` (Unity AI Gateway) and deployed on **Databricks Apps**. Intended as a quick-demo template, not a production reference.

## Files

- `app.py` — Streamlit UI with `st.write_stream` rendering token deltas from the serving endpoint.
- `app.yaml` — App config: launch command + `SERVING_ENDPOINT` env var.
- `requirements.txt` — `streamlit`, `mlflow`, `databricks-sdk`.

## Run locally

```bash
pip install -r requirements.txt
export SERVING_ENDPOINT="databricks-gpt-5-5"
databricks auth login --host https://<your-workspace>.cloud.databricks.com
streamlit run app.py
```

## Deploy to Databricks Apps

1. Authenticate the CLI (one-time):

   ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
   ```

2. Create the app (one-time):

   ```bash
   databricks apps create streamlit-gpt-chatbot
   ```

3. Sync this folder to your workspace:

   ```bash
   databricks sync --watch . /Workspace/Users/<your-email>/streamlit-gpt-chatbot
   ```

   (Drop `--watch` for a one-shot upload.)

4. Deploy the app from the synced source path:

   ```bash
   databricks apps deploy streamlit-gpt-chatbot \
     --source-code-path /Workspace/Users/<your-email>/streamlit-gpt-chatbot
   ```

5. Open the app URL printed by the deploy command and start chatting.

To swap models, change `SERVING_ENDPOINT` in `app.yaml` to any chat-completions-compatible endpoint (e.g. `databricks-claude-opus-4`) and redeploy.

## References

- [Databricks `streamlit-chatbot-app` template](https://github.com/databricks/app-templates/tree/main/streamlit-chatbot-app)
- [Databricks docs — Build a chat app](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app)
