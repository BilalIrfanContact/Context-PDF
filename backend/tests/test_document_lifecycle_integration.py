import json
import unittest
from unittest.mock import patch

from fastapi import FastAPI

from backend.routers import documents, upload
from backend.services.internal_auth import require_authenticated_user
from backend.services.persistence.common import PersistenceError


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(upload.router)
    app.include_router(documents.router)

    async def override_user():
        return "user-a"

    app.dependency_overrides[require_authenticated_user] = override_user
    return app


def _multipart_request_body(
    *,
    field_name: str,
    filename: str,
    content_type: str,
    payload: bytes,
    boundary: str,
) -> bytes:
    lines = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        payload,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(lines)


async def _request_asgi(
    app: FastAPI,
    *,
    method: str,
    path: str,
    body: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    response_status = 500
    response_headers: dict[str, str] = {}
    response_body = bytearray()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}

        sent = True
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    async def send(message):
        nonlocal response_status
        if message["type"] == "http.response.start":
            response_status = message["status"]
            response_headers.update(
                {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
            )
            return

        if message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    await app(scope, receive, send)
    return response_status, response_headers, bytes(response_body)


class DocumentLifecycleHttpIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = _build_test_app()

    async def asyncTearDown(self):
        self.app.dependency_overrides.clear()

    async def test_upload_endpoint_returns_ready_contract_on_success(self):
        boundary = "boundary123"
        body = _multipart_request_body(
            field_name="file",
            filename="report.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4",
            boundary=boundary,
        )

        with (
            patch("backend.services.document_lifecycle.extract_text_from_pdf", return_value="alpha beta"),
            patch("backend.services.document_lifecycle.chunk_text", return_value=["alpha", "beta"]),
            patch("backend.services.document_lifecycle.build_vector_store", return_value=2),
            patch(
                "backend.services.document_lifecycle.upload_file_to_storage",
                return_value="documents/user-a/doc-1/report.pdf",
            ),
            patch("backend.services.document_lifecycle.insert_document") as insert_document_mock,
            patch("backend.services.document_lifecycle.uuid.uuid4", return_value="doc-1"),
        ):
            status, headers, response_body = await _request_asgi(
                self.app,
                method="POST",
                path="/upload",
                body=body,
                headers=[
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                    ),
                    (b"content-length", str(len(body)).encode("utf-8")),
                ],
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "status": "success",
                "lifecycle_status": "ready",
                "document_id": "doc-1",
                "chunk_count": 2,
                "stored_count": 2,
            },
        )
        insert_document_mock.assert_called_once_with(
            document_id="doc-1",
            user_id="user-a",
            filename="report.pdf",
            storage_url="documents/user-a/doc-1/report.pdf",
        )

    async def test_upload_endpoint_returns_structured_storage_failure_and_rolls_back_index(self):
        boundary = "boundary123"
        body = _multipart_request_body(
            field_name="file",
            filename="report.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4",
            boundary=boundary,
        )

        with (
            patch("backend.services.document_lifecycle.extract_text_from_pdf", return_value="alpha beta"),
            patch("backend.services.document_lifecycle.chunk_text", return_value=["alpha", "beta"]),
            patch("backend.services.document_lifecycle.build_vector_store", return_value=2),
            patch(
                "backend.services.document_lifecycle.upload_file_to_storage",
                side_effect=PersistenceError("Failed to upload document to Supabase Storage"),
            ),
            patch("backend.services.document_lifecycle.delete_vector_store") as delete_vector_store_mock,
            patch("backend.services.document_lifecycle.uuid.uuid4", return_value="doc-1"),
        ):
            status, headers, response_body = await _request_asgi(
                self.app,
                method="POST",
                path="/upload",
                body=body,
                headers=[
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                    ),
                    (b"content-length", str(len(body)).encode("utf-8")),
                ],
            )

        self.assertEqual(status, 502)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "detail": {
                    "message": "Failed to upload document to Supabase Storage",
                    "lifecycle_status": "failed",
                    "failure_stage": "storage",
                    "reason_code": "storage_upload_failed",
                    "cleanup_status": "completed",
                }
            },
        )
        delete_vector_store_mock.assert_called_once_with("doc-1")

    async def test_upload_endpoint_accepts_markdown_files(self):
        boundary = "boundary123"
        body = _multipart_request_body(
            field_name="file",
            filename="notes.md",
            content_type="text/plain",
            payload=b"# Notes\n\nalpha beta",
            boundary=boundary,
        )

        with (
            patch("backend.services.document_lifecycle.chunk_text", return_value=["alpha", "beta"]),
            patch("backend.services.document_lifecycle.build_vector_store", return_value=2),
            patch(
                "backend.services.document_lifecycle.upload_file_to_storage",
                return_value="documents/user-a/doc-1/notes.md",
            ) as upload_file_to_storage_mock,
            patch("backend.services.document_lifecycle.insert_document") as insert_document_mock,
            patch("backend.services.document_lifecycle.uuid.uuid4", return_value="doc-1"),
        ):
            status, headers, response_body = await _request_asgi(
                self.app,
                method="POST",
                path="/upload",
                body=body,
                headers=[
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                    ),
                    (b"content-length", str(len(body)).encode("utf-8")),
                ],
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "status": "success",
                "lifecycle_status": "ready",
                "document_id": "doc-1",
                "chunk_count": 2,
                "stored_count": 2,
            },
        )
        upload_file_to_storage_mock.assert_called_once_with(
            user_id="user-a",
            document_id="doc-1",
            filename="notes.md",
            data=b"# Notes\n\nalpha beta",
            content_type="text/markdown",
        )
        insert_document_mock.assert_called_once_with(
            document_id="doc-1",
            user_id="user-a",
            filename="notes.md",
            storage_url="documents/user-a/doc-1/notes.md",
        )

    async def test_upload_endpoint_rejects_empty_markdown_files(self):
        boundary = "boundary123"
        body = _multipart_request_body(
            field_name="file",
            filename="empty.md",
            content_type="text/markdown",
            payload=b"",
            boundary=boundary,
        )

        status, headers, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/upload",
            body=body,
            headers=[
                (
                    b"content-type",
                    f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                ),
                (b"content-length", str(len(body)).encode("utf-8")),
            ],
        )

        self.assertEqual(status, 400)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "detail": {
                    "message": "No extractable text found in the Markdown file.",
                    "lifecycle_status": "rejected",
                    "failure_stage": "validation",
                    "reason_code": "no_extractable_text",
                    "cleanup_status": "not-needed",
                }
            },
        )

    async def test_upload_endpoint_rejects_unreadable_pdf_files(self):
        boundary = "boundary123"
        body = _multipart_request_body(
            field_name="file",
            filename="broken.pdf",
            content_type="application/octet-stream",
            payload=b"not-a-real-pdf",
            boundary=boundary,
        )

        with patch(
            "backend.services.document_lifecycle.extract_text_from_pdf",
            side_effect=ValueError("Could not parse PDF"),
        ):
            status, headers, response_body = await _request_asgi(
                self.app,
                method="POST",
                path="/upload",
                body=body,
                headers=[
                    (
                        b"content-type",
                        f"multipart/form-data; boundary={boundary}".encode("utf-8"),
                    ),
                    (b"content-length", str(len(body)).encode("utf-8")),
                ],
            )

        self.assertEqual(status, 400)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "detail": {
                    "message": "The PDF file could not be read.",
                    "lifecycle_status": "rejected",
                    "failure_stage": "validation",
                    "reason_code": "unreadable_document",
                    "cleanup_status": "not-needed",
                }
            },
        )

    async def test_delete_endpoint_returns_deleted_contract_on_success(self):
        with (
            patch(
                "backend.routers.documents.require_user_document",
                return_value={"id": "doc-1", "user_id": "user-a", "storage_url": "documents/user-a/doc-1/report.pdf"},
            ),
            patch(
                "backend.services.document_lifecycle.list_document_conversation_ids",
                return_value=["convo-1"],
            ),
            patch("backend.services.document_lifecycle.delete_storage_object") as delete_storage_object_mock,
            patch("backend.services.document_lifecycle.delete_document_record") as delete_document_record_mock,
            patch("backend.services.document_lifecycle.delete_messages_for_conversation") as delete_messages_mock,
            patch("backend.services.document_lifecycle.delete_user_document_conversations") as delete_conversations_mock,
            patch("backend.services.document_lifecycle.delete_vector_store") as delete_vector_store_mock,
        ):
            status, headers, response_body = await _request_asgi(
                self.app,
                method="DELETE",
                path="/documents/doc-1",
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "deleted": True,
                "lifecycle_status": "deleted",
                "cleanup_status": "completed",
            },
        )
        delete_storage_object_mock.assert_called_once_with("documents/user-a/doc-1/report.pdf")
        delete_document_record_mock.assert_called_once_with(document_id="doc-1", user_id="user-a")
        delete_messages_mock.assert_called_once_with("convo-1")
        delete_conversations_mock.assert_called_once_with(user_id="user-a", document_id="doc-1")
        delete_vector_store_mock.assert_called_once_with("doc-1")

    async def test_delete_endpoint_returns_structured_storage_failure_without_metadata_delete(self):
        with (
            patch(
                "backend.routers.documents.require_user_document",
                return_value={"id": "doc-1", "user_id": "user-a", "storage_url": "documents/user-a/doc-1/report.pdf"},
            ),
            patch(
                "backend.services.document_lifecycle.list_document_conversation_ids",
                return_value=["convo-1"],
            ),
            patch(
                "backend.services.document_lifecycle.delete_storage_object",
                side_effect=PersistenceError("Failed to delete document from Supabase Storage"),
            ),
            patch("backend.services.document_lifecycle.delete_document_record") as delete_document_record_mock,
            patch("backend.services.document_lifecycle.delete_messages_for_conversation") as delete_messages_mock,
            patch("backend.services.document_lifecycle.delete_user_document_conversations") as delete_conversations_mock,
            patch("backend.services.document_lifecycle.delete_vector_store") as delete_vector_store_mock,
        ):
            status, headers, response_body = await _request_asgi(
                self.app,
                method="DELETE",
                path="/documents/doc-1",
            )

        self.assertEqual(status, 502)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            json.loads(response_body),
            {
                "detail": {
                    "message": "Failed to delete document from Supabase Storage",
                    "lifecycle_status": "failed",
                    "failure_stage": "storage",
                    "reason_code": "storage_delete_failed",
                    "cleanup_status": "partial",
                }
            },
        )
        delete_document_record_mock.assert_not_called()
        delete_messages_mock.assert_not_called()
        delete_conversations_mock.assert_not_called()
        delete_vector_store_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
