"""
One-time script to apply schema.sql to the Lakebase instance.
Run once after provisioning:  python3 apply_schema.py
"""

import os
import uuid
import psycopg2
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")

w = WorkspaceClient(
    host=os.environ["DATABRICKS_HOST"],
    token=os.environ["DATABRICKS_TOKEN"],
)

cred = w.database.generate_database_credential(
    request_id=str(uuid.uuid4()),
    instance_names=[os.environ["LAKEBASE_INSTANCE_NAME"]],
)

conn = psycopg2.connect(
    host=os.environ["LAKEBASE_HOST"],
    port=int(os.environ.get("LAKEBASE_PORT", 5432)),
    dbname=os.environ["LAKEBASE_DATABASE"],
    user=os.environ["LAKEBASE_USER"],
    password=cred.token,
    sslmode=os.environ.get("LAKEBASE_SSLMODE", "require"),
    connect_timeout=15,
)
conn.autocommit = True

with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("pgvector extension ready")

conn.autocommit = False
with open(SCHEMA_FILE) as f:
    ddl = f.read()

with conn.cursor() as cur:
    cur.execute(ddl)
conn.commit()
conn.close()

print("Schema applied. Tables: users, sessions, messages, long_term_memory")
