import sys, logging
from databricks.sdk import WorkspaceClient

def validate_connection():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    try:
        w = WorkspaceClient()  # uses unified auth
        me = w.current_user.me()  # non-mutating call
        print(f"OK: Connected to {w.config.host} as {me.user_name}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    validate_connection()