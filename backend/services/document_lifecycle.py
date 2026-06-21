import uuid
from dataclasses import dataclass
from os.path import splitext
from typing import Callable, Literal

from fastapi import HTTPException, UploadFile

from ..models.schemas import DeleteDocumentResponse, UploadResponse
from .markdown_extractor import extract_text_from_markdown
from .pdf_extractor import extract_text_from_pdf
from .persistence import PersistenceError
from .persistence.conversations_repository import (
    delete_user_document_conversations,
    list_document_conversation_ids,
)
from .persistence.documents_repository import delete_document_record, insert_document
from .persistence.messages_repository import delete_messages_for_conversation
from .persistence.storage_repository import delete_storage_object, upload_file_to_storage
from .text_chunker import chunk_text
from .vector_store import build_vector_store, delete_vector_store

UploadFailureStage = Literal["validation", "indexing", "storage", "metadata"]
UploadCleanupStatus = Literal["not-needed", "completed", "failed"]
UploadLifecycleStatus = Literal["completed", "rejected", "failed"]
UploadReasonCode = Literal[
    "invalid_file_type",
    "unreadable_document",
    "no_extractable_text",
    "no_usable_chunks",
    "indexing_failed",
    "no_chunks_stored",
    "storage_upload_failed",
    "metadata_persist_failed",
]
DeleteFailureStage = Literal["conversations", "indexing", "storage", "metadata"]
DeleteCleanupStatus = Literal["not-started", "partial", "completed"]
DeleteLifecycleStatus = Literal["completed", "failed"]
DeleteReasonCode = Literal[
    "conversation_lookup_failed",
    "storage_delete_failed",
    "metadata_delete_failed",
    "conversation_cleanup_failed",
    "indexing_cleanup_failed",
]


@dataclass(frozen=True)
class UploadLifecycleResult:
    status: UploadLifecycleStatus
    http_status: int
    document_id: str | None = None
    chunk_count: int = 0
    stored_count: int = 0
    detail: str | None = None
    failure_stage: UploadFailureStage | None = None
    reason_code: UploadReasonCode | None = None
    cleanup_status: UploadCleanupStatus = "not-needed"

    def to_response(self) -> UploadResponse:
        if self.status != "completed" or not self.document_id:
            raise ValueError("Only completed upload lifecycle results can be converted to responses.")

        return UploadResponse(
            status="success",
            lifecycle_status="ready",
            document_id=self.document_id,
            chunk_count=self.chunk_count,
            stored_count=self.stored_count,
        )

    def to_http_exception(self) -> HTTPException:
        if self.status == "completed":
            raise ValueError("Completed upload lifecycle results cannot be converted to errors.")

        return HTTPException(status_code=self.http_status, detail=self.to_error_detail())

    def to_error_detail(self) -> dict[str, str]:
        return {
            "message": self.detail or "Upload failed.",
            "lifecycle_status": "rejected" if self.status == "rejected" else "failed",
            "failure_stage": self.failure_stage or "validation",
            "reason_code": self.reason_code or "invalid_file_type",
            "cleanup_status": self.cleanup_status,
        }


@dataclass(frozen=True)
class DeleteLifecycleResult:
    status: DeleteLifecycleStatus
    http_status: int
    detail: str | None = None
    failure_stage: DeleteFailureStage | None = None
    reason_code: DeleteReasonCode | None = None
    cleanup_status: DeleteCleanupStatus = "not-started"

    def to_response(self) -> DeleteDocumentResponse:
        if self.status != "completed":
            raise ValueError("Only completed delete lifecycle results can be converted to responses.")

        return DeleteDocumentResponse(
            deleted=True,
            lifecycle_status="deleted",
            cleanup_status="completed",
        )

    def to_http_exception(self) -> HTTPException:
        if self.status == "completed":
            raise ValueError("Completed delete lifecycle results cannot be converted to errors.")

        return HTTPException(status_code=self.http_status, detail=self.to_error_detail())

    def to_error_detail(self) -> dict[str, str]:
        return {
            "message": self.detail or "Delete failed.",
            "lifecycle_status": "failed",
            "failure_stage": self.failure_stage or "metadata",
            "reason_code": self.reason_code or "metadata_delete_failed",
            "cleanup_status": self.cleanup_status,
        }


@dataclass(frozen=True)
class SupportedUploadKind:
    label: str
    content_type: str
    fallback_filename: str
    extract_text: Callable[[bytes], str]


def _resolve_upload_kind(file: UploadFile) -> SupportedUploadKind | None:
    filename = (file.filename or "").lower()
    extension = splitext(filename)[1]
    content_type = (file.content_type or "").lower()

    if content_type == "application/pdf" or extension == ".pdf":
        return SupportedUploadKind(
            label="PDF",
            content_type="application/pdf",
            fallback_filename="document.pdf",
            extract_text=extract_text_from_pdf,
        )

    if content_type == "text/markdown" or extension == ".md":
        return SupportedUploadKind(
            label="Markdown",
            content_type="text/markdown",
            fallback_filename="document.md",
            extract_text=extract_text_from_markdown,
        )

    return None


def _cleanup_failed_upload(document_id: str, storage_url: str | None = None) -> UploadCleanupStatus:
    cleanup_failed = False

    try:
        delete_vector_store(document_id)
    except Exception:
        cleanup_failed = True

    if storage_url:
        try:
            delete_storage_object(storage_url)
        except PersistenceError:
            cleanup_failed = True

    return "failed" if cleanup_failed else "completed"


def _delete_failure(
    *,
    detail: str,
    failure_stage: DeleteFailureStage,
    reason_code: DeleteReasonCode,
    cleanup_status: DeleteCleanupStatus,
    http_status: int = 502,
) -> DeleteLifecycleResult:
    return DeleteLifecycleResult(
        status="failed",
        http_status=http_status,
        detail=detail,
        failure_stage=failure_stage,
        reason_code=reason_code,
        cleanup_status=cleanup_status,
    )


async def upload_document(file: UploadFile, user_id: str) -> UploadLifecycleResult:
    upload_kind = _resolve_upload_kind(file)
    if upload_kind is None:
        return UploadLifecycleResult(
            status="rejected",
            http_status=400,
            detail="Only PDF and Markdown files are supported.",
            failure_stage="validation",
            reason_code="invalid_file_type",
        )

    data = await file.read()
    try:
        text = upload_kind.extract_text(data)
    except Exception:
        return UploadLifecycleResult(
            status="rejected",
            http_status=400,
            detail=f"The {upload_kind.label} file could not be read.",
            failure_stage="validation",
            reason_code="unreadable_document",
        )

    if not text:
        return UploadLifecycleResult(
            status="rejected",
            http_status=400,
            detail=f"No extractable text found in the {upload_kind.label} file.",
            failure_stage="validation",
            reason_code="no_extractable_text",
        )

    chunks = [chunk.strip() for chunk in chunk_text(text) if chunk and chunk.strip()]
    if not chunks:
        return UploadLifecycleResult(
            status="rejected",
            http_status=400,
            detail=f"No usable text chunks were created from the {upload_kind.label} file.",
            failure_stage="validation",
            reason_code="no_usable_chunks",
        )

    document_id = str(uuid.uuid4())
    try:
        stored_count = build_vector_store(document_id=document_id, chunks=chunks)
    except Exception as exc:
        cleanup_status = _cleanup_failed_upload(document_id)
        detail = str(exc) or "Failed to index document chunks."
        if cleanup_status == "failed":
            detail = f"{detail} Cleanup may be required for partially indexed chunks."

        return UploadLifecycleResult(
            status="failed",
            http_status=500,
            document_id=document_id,
            chunk_count=len(chunks),
            detail=detail,
            failure_stage="indexing",
            reason_code="indexing_failed",
            cleanup_status=cleanup_status,
        )

    if stored_count == 0:
        cleanup_status = _cleanup_failed_upload(document_id)
        detail = "Chunks were created but not stored. Check OpenAI key and embedding setup."
        if cleanup_status == "failed":
            detail = f"{detail} Cleanup may be required for partially indexed chunks."

        return UploadLifecycleResult(
            status="failed",
            http_status=500,
            document_id=document_id,
            chunk_count=len(chunks),
            detail=detail,
            failure_stage="indexing",
            reason_code="no_chunks_stored",
            cleanup_status=cleanup_status,
        )

    filename = file.filename or upload_kind.fallback_filename

    try:
        storage_url = upload_file_to_storage(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            data=data,
            content_type=upload_kind.content_type,
        )
    except PersistenceError as exc:
        cleanup_status = _cleanup_failed_upload(document_id)
        detail = str(exc)
        if cleanup_status == "failed":
            detail = f"{detail} Cleanup may be required for indexed chunks."

        return UploadLifecycleResult(
            status="failed",
            http_status=502,
            document_id=document_id,
            chunk_count=len(chunks),
            stored_count=stored_count,
            detail=detail,
            failure_stage="storage",
            reason_code="storage_upload_failed",
            cleanup_status=cleanup_status,
        )

    try:
        insert_document(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            storage_url=storage_url,
        )
    except PersistenceError as exc:
        cleanup_status = _cleanup_failed_upload(document_id, storage_url=storage_url)
        detail = str(exc)
        if cleanup_status == "failed":
            detail = f"{detail} Cleanup may be required for uploaded document artifacts."

        return UploadLifecycleResult(
            status="failed",
            http_status=502,
            document_id=document_id,
            chunk_count=len(chunks),
            stored_count=stored_count,
            detail=detail,
            failure_stage="metadata",
            reason_code="metadata_persist_failed",
            cleanup_status=cleanup_status,
        )

    return UploadLifecycleResult(
        status="completed",
        http_status=200,
        document_id=document_id,
        chunk_count=len(chunks),
        stored_count=stored_count,
    )


def delete_document(document_id: str, user_id: str, storage_url: str | None = None) -> DeleteLifecycleResult:
    try:
        conversation_ids = list_document_conversation_ids(user_id=user_id, document_id=document_id)
    except PersistenceError as exc:
        return _delete_failure(
            detail=str(exc),
            failure_stage="conversations",
            reason_code="conversation_lookup_failed",
            cleanup_status="not-started",
        )

    if storage_url:
        try:
            delete_storage_object(storage_url)
        except PersistenceError as exc:
            return _delete_failure(
                detail=str(exc),
                failure_stage="storage",
                reason_code="storage_delete_failed",
                cleanup_status="partial",
            )

    # Delete metadata last so failed cleanup remains visible and retryable.
    try:
        delete_document_record(document_id=document_id, user_id=user_id)
    except PersistenceError as exc:
        return _delete_failure(
            detail=str(exc),
            failure_stage="metadata",
            reason_code="metadata_delete_failed",
            cleanup_status="partial",
        )

    try:
        for conversation_id in conversation_ids:
            delete_messages_for_conversation(conversation_id)
        delete_user_document_conversations(user_id=user_id, document_id=document_id)
    except PersistenceError as exc:
        return _delete_failure(
            detail=str(exc),
            failure_stage="conversations",
            reason_code="conversation_cleanup_failed",
            cleanup_status="partial",
        )

    try:
        delete_vector_store(document_id)
    except Exception as exc:
        return _delete_failure(
            detail=str(exc) or "Failed to delete indexed document chunks.",
            failure_stage="indexing",
            reason_code="indexing_cleanup_failed",
            cleanup_status="partial",
            http_status=500,
        )

    return DeleteLifecycleResult(
        status="completed",
        http_status=200,
        cleanup_status="completed",
    )
