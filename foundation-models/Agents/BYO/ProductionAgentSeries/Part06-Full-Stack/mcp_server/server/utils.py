import contextvars
import os
from databricks.sdk import WorkspaceClient

header_store = contextvars.ContextVar("header_store")


def get_workspace_client():
    return WorkspaceClient()


def get_user_authenticated_workspace_client():
    # Local: default auth
    if "DATABRICKS_APP_NAME" not in os.environ:
        return WorkspaceClient()

    # In Databricks App: token is forwarded by the platform
    headers = header_store.get({})
    token = headers.get("x-forwarded-access-token")

    if not token:
        raise ValueError("Missing x-forwarded-access-token header")

    return WorkspaceClient(token=token, auth_type="pat")
