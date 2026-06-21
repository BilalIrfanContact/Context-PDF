import os

from .common import (
    get_storage_bucket,
    get_storage_client,
    map_persistence_error,
)


def upload_file_to_storage(
    *,
    user_id: str,
    document_id: str,
    filename: str,
    data: bytes,
    content_type: str,
) -> str:
    bucket = get_storage_bucket()
    safe_name = os.path.basename(filename) or "document"
    storage_path = f"{user_id}/{document_id}/{safe_name}"

    try:
        storage = get_storage_client().from_(bucket)
        try:
            storage.upload(
                storage_path,
                data,
                file_options={"content-type": content_type, "upsert": "false"},
            )
        except TypeError:
            storage.upload(
                storage_path,
                data,
                {"content-type": content_type, "upsert": "false"},
            )
    except Exception as exc:
        raise map_persistence_error("Failed to upload document to Supabase Storage", exc) from exc

    return f"{bucket}/{storage_path}"


def delete_storage_object(storage_url: str) -> None:
    bucket, _, object_path = storage_url.partition("/")
    if not bucket or not object_path:
        return

    try:
        get_storage_client().from_(bucket).remove([object_path])
    except Exception as exc:
        if "not found" not in str(exc).lower():
            raise map_persistence_error("Failed to delete document from Supabase Storage", exc) from exc
