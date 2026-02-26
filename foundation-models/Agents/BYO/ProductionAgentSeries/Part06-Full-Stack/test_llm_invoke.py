from openai import OpenAI
import os

# To get a DATABRICKS_TOKEN, click the "Generate Access Token" button or follow https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://7474655512364318.ai-gateway.cloud.databricks.com/mlflow/v1"
)

chat_completion = client.chat.completions.create(
  messages=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is Databricks?"},
  ],
  model="databricks-gpt-oss-120b",
  max_tokens=1024
)

print(chat_completion.choices[0].message.content)