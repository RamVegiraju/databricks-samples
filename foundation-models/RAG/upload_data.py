import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceAlreadyExists
from databricks.sdk.service.catalog import VolumeType  # <--- Added this import

w = WorkspaceClient()

# Configuration
CATALOG = "demo_rag_catalog"
SCHEMA = "demo_rag_data"
VOLUME = "pdf_source_files"
# ADJUST THIS TO YOUR METASTORE STORAGE LOCATION
STORAGE_LOCATION = "s3://{ENTER METASTORE STORAGE LOCATION}/rag_data_root" 
LOCAL_DATA_DIR = "./data" 

def setup_rag_infrastructure():
    # 1. CATALOG: Check/Create
    try:
        w.catalogs.get(CATALOG)
        print(f"ℹ️ Catalog '{CATALOG}' already exists.")
    except NotFound:
        print(f"Creating catalog: {CATALOG}...")
        w.catalogs.create(name=CATALOG, storage_root=STORAGE_LOCATION)
        print(f"✅ Catalog '{CATALOG}' created.")

    # 2. SCHEMA: Check/Create
    try:
        w.schemas.get(f"{CATALOG}.{SCHEMA}")
        print(f"ℹ️ Schema '{SCHEMA}' already exists.")
    except NotFound:
        print(f"Creating schema: {SCHEMA}...")
        w.schemas.create(name=SCHEMA, catalog_name=CATALOG)
        print(f"✅ Schema '{SCHEMA}' created.")

    # 3. VOLUME: Check/Create
    # Listing volumes to check for existence
    volumes = w.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)
    volume_exists = any(v.name == VOLUME for v in volumes)
    
    if volume_exists:
        print(f"ℹ️ Volume '{VOLUME}' already exists.")
    else:
        try:
            print(f"Creating volume: {VOLUME}...")
            w.volumes.create(
                catalog_name=CATALOG,
                schema_name=SCHEMA,
                name=VOLUME,
                volume_type=VolumeType.MANAGED  # <--- Changed from "MANAGED" to Enum
            )
            print(f"✅ Volume '{VOLUME}' created.")
        except ResourceAlreadyExists:
            print(f"ℹ️ Volume '{VOLUME}' already exists (race condition).")

    # 4. UPLOAD: Idempotent Upload
    if os.path.exists(LOCAL_DATA_DIR):
        volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
        
        pdf_files = [f for f in os.listdir(LOCAL_DATA_DIR) if f.lower().endswith(".pdf")]
        print(f"📂 Found {len(pdf_files)} PDFs. Starting sync...")

        for filename in pdf_files:
            local_file = os.path.join(LOCAL_DATA_DIR, filename)
            target_file = f"{volume_path}/{filename}"
            try:
                with open(local_file, "rb") as f:
                    # overwrite=True makes the file sync non-blocking
                    w.files.upload(target_file, f, overwrite=True)
                print(f"🚀 Synced: {filename}")
            except Exception as e:
                print(f"⚠️ Failed to upload {filename}: {e}")
    else:
        print(f"❌ Local directory '{LOCAL_DATA_DIR}' not found.")

if __name__ == "__main__":
    setup_rag_infrastructure()