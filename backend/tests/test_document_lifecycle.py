import unittest
from unittest.mock import AsyncMock, patch

from backend.services.document_lifecycle import delete_document, upload_document
from backend.services.persistence.common import PersistenceError


class DocumentLifecycleTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_upload_document_returns_completed_result_on_success(self):
        file = AsyncMock()
        file.content_type = "application/pdf"
        file.filename = "report.pdf"
        file.read.return_value = b"%PDF"

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
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.document_id, "doc-1")
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.stored_count, 2)
        self.assertEqual(result.cleanup_status, "not-needed")
        self.assertEqual(result.to_response().lifecycle_status, "ready")
        insert_document_mock.assert_called_once_with(
            document_id="doc-1",
            user_id="user-a",
            filename="report.pdf",
            storage_url="documents/user-a/doc-1/report.pdf",
        )

    async def test_upload_document_cleans_up_indexed_chunks_when_storage_upload_fails(self):
        file = AsyncMock()
        file.content_type = "application/pdf"
        file.filename = "report.pdf"
        file.read.return_value = b"%PDF"

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
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "storage")
        self.assertEqual(result.http_status, 502)
        self.assertEqual(result.cleanup_status, "completed")
        self.assertEqual(result.detail, "Failed to upload document to Supabase Storage")
        self.assertEqual(
            result.to_error_detail(),
            {
                "message": "Failed to upload document to Supabase Storage",
                "lifecycle_status": "failed",
                "failure_stage": "storage",
                "reason_code": "storage_upload_failed",
                "cleanup_status": "completed",
            },
        )
        delete_vector_store_mock.assert_called_once_with("doc-1")

    async def test_upload_document_returns_structured_failure_when_indexing_raises(self):
        file = AsyncMock()
        file.content_type = "application/pdf"
        file.filename = "report.pdf"
        file.read.return_value = b"%PDF"

        with (
            patch("backend.services.document_lifecycle.extract_text_from_pdf", return_value="alpha beta"),
            patch("backend.services.document_lifecycle.chunk_text", return_value=["alpha", "beta"]),
            patch(
                "backend.services.document_lifecycle.build_vector_store",
                side_effect=RuntimeError("Embedding provider unavailable"),
            ),
            patch("backend.services.document_lifecycle.delete_vector_store") as delete_vector_store_mock,
            patch("backend.services.document_lifecycle.uuid.uuid4", return_value="doc-1"),
        ):
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "indexing")
        self.assertEqual(result.reason_code, "indexing_failed")
        self.assertEqual(result.http_status, 500)
        self.assertEqual(result.cleanup_status, "completed")
        self.assertEqual(result.detail, "Embedding provider unavailable")
        delete_vector_store_mock.assert_called_once_with("doc-1")

    async def test_upload_document_cleans_up_storage_and_index_when_metadata_persist_fails(self):
        file = AsyncMock()
        file.content_type = "application/pdf"
        file.filename = "report.pdf"
        file.read.return_value = b"%PDF"

        with (
            patch("backend.services.document_lifecycle.extract_text_from_pdf", return_value="alpha beta"),
            patch("backend.services.document_lifecycle.chunk_text", return_value=["alpha", "beta"]),
            patch("backend.services.document_lifecycle.build_vector_store", return_value=2),
            patch(
                "backend.services.document_lifecycle.upload_file_to_storage",
                return_value="documents/user-a/doc-1/report.pdf",
            ),
            patch(
                "backend.services.document_lifecycle.insert_document",
                side_effect=PersistenceError("Failed to persist document metadata"),
            ),
            patch("backend.services.document_lifecycle.delete_vector_store") as delete_vector_store_mock,
            patch("backend.services.document_lifecycle.delete_storage_object") as delete_storage_object_mock,
            patch("backend.services.document_lifecycle.uuid.uuid4", return_value="doc-1"),
        ):
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "metadata")
        self.assertEqual(result.reason_code, "metadata_persist_failed")
        self.assertEqual(result.http_status, 502)
        self.assertEqual(result.cleanup_status, "completed")
        self.assertEqual(result.detail, "Failed to persist document metadata")
        delete_vector_store_mock.assert_called_once_with("doc-1")
        delete_storage_object_mock.assert_called_once_with("documents/user-a/doc-1/report.pdf")

    async def test_upload_document_rejects_non_pdf_files_before_processing(self):
        file = AsyncMock()
        file.content_type = "text/plain"
        file.filename = "notes.txt"

        with (
            patch("backend.services.document_lifecycle.extract_text_from_pdf") as extract_text_mock,
            patch("backend.services.document_lifecycle.build_vector_store") as build_vector_store_mock,
        ):
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.failure_stage, "validation")
        self.assertEqual(result.reason_code, "invalid_file_type")
        self.assertEqual(result.http_status, 400)
        self.assertEqual(result.detail, "Only PDF and Markdown files are supported.")
        extract_text_mock.assert_not_called()
        build_vector_store_mock.assert_not_called()

    async def test_upload_document_accepts_markdown_by_extension_and_stores_markdown_content_type(self):
        file = AsyncMock()
        file.content_type = "application/octet-stream"
        file.filename = "notes.md"
        file.read.return_value = b"# Notes\n\nalpha beta"

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
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.document_id, "doc-1")
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

    async def test_upload_document_rejects_empty_markdown_files(self):
        file = AsyncMock()
        file.content_type = "text/markdown"
        file.filename = "empty.md"
        file.read.return_value = b""

        with (
            patch("backend.services.document_lifecycle.build_vector_store") as build_vector_store_mock,
            patch("backend.services.document_lifecycle.upload_file_to_storage") as upload_file_to_storage_mock,
        ):
            result = await upload_document(file=file, user_id="user-a")

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.failure_stage, "validation")
        self.assertEqual(result.reason_code, "no_extractable_text")
        self.assertEqual(result.detail, "No extractable text found in the Markdown file.")
        build_vector_store_mock.assert_not_called()
        upload_file_to_storage_mock.assert_not_called()

    async def test_delete_document_runs_through_single_lifecycle_path(self):
        call_order: list[str] = []

        def record(name: str):
            def inner(*args, **kwargs):
                del args, kwargs
                call_order.append(name)

            return inner

        with (
            patch(
                "backend.services.document_lifecycle.list_document_conversation_ids",
                return_value=["convo-1", "convo-2"],
            ),
            patch(
                "backend.services.document_lifecycle.delete_messages_for_conversation",
                side_effect=lambda conversation_id: call_order.append(f"message:{conversation_id}"),
            ),
            patch(
                "backend.services.document_lifecycle.delete_user_document_conversations",
                side_effect=record("conversations"),
            ),
            patch(
                "backend.services.document_lifecycle.delete_vector_store",
                side_effect=record("index"),
            ),
            patch(
                "backend.services.document_lifecycle.delete_storage_object",
                side_effect=record("storage"),
            ),
            patch(
                "backend.services.document_lifecycle.delete_document_record",
                side_effect=record("metadata"),
            ),
        ):
            result = delete_document(
                document_id="doc-1",
                user_id="user-a",
                storage_url="documents/user-a/doc-1/report.pdf",
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.cleanup_status, "completed")
        self.assertEqual(
            call_order,
            [
                "storage",
                "metadata",
                "message:convo-1",
                "message:convo-2",
                "conversations",
                "index",
            ],
        )
        self.assertEqual(result.to_response().lifecycle_status, "deleted")

    async def test_delete_document_returns_not_started_when_conversation_lookup_fails(self):
        with patch(
            "backend.services.document_lifecycle.list_document_conversation_ids",
            side_effect=PersistenceError("Failed to load document conversations"),
        ):
            result = delete_document(document_id="doc-1", user_id="user-a")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "conversations")
        self.assertEqual(result.reason_code, "conversation_lookup_failed")
        self.assertEqual(result.cleanup_status, "not-started")
        self.assertEqual(result.http_status, 502)

    async def test_delete_document_returns_partial_failure_when_storage_delete_fails(self):
        with (
            patch(
                "backend.services.document_lifecycle.list_document_conversation_ids",
                return_value=["convo-1"],
            ),
            patch("backend.services.document_lifecycle.delete_messages_for_conversation"),
            patch("backend.services.document_lifecycle.delete_user_document_conversations"),
            patch("backend.services.document_lifecycle.delete_vector_store"),
            patch(
                "backend.services.document_lifecycle.delete_storage_object",
                side_effect=PersistenceError("Failed to delete document from Supabase Storage"),
            ),
        ):
            result = delete_document(
                document_id="doc-1",
                user_id="user-a",
                storage_url="documents/user-a/doc-1/report.pdf",
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "storage")
        self.assertEqual(result.cleanup_status, "partial")
        self.assertEqual(result.http_status, 502)
        self.assertEqual(
            result.to_error_detail(),
            {
                "message": "Failed to delete document from Supabase Storage",
                "lifecycle_status": "failed",
                "failure_stage": "storage",
                "reason_code": "storage_delete_failed",
                "cleanup_status": "partial",
            },
        )


if __name__ == "__main__":
    unittest.main()
