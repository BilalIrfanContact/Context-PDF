from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ErrorDetailResponse(BaseModel):
    detail: str


class UploadResponse(BaseModel):
    status: Literal["success"]
    lifecycle_status: Literal["ready"] = Field(..., description="Lifecycle state for the upload flow")
    document_id: str = Field(..., description="UUID for the uploaded document")
    chunk_count: int = Field(..., description="Number of chunks stored for retrieval")
    stored_count: int = Field(..., description="Number of chunks persisted in Chroma")


class UploadErrorDetail(BaseModel):
    message: str
    lifecycle_status: Literal["failed", "rejected"]
    failure_stage: Literal["validation", "indexing", "storage", "metadata"]
    reason_code: Literal[
        "invalid_file_type",
        "unreadable_document",
        "no_extractable_text",
        "no_usable_chunks",
        "indexing_failed",
        "no_chunks_stored",
        "storage_upload_failed",
        "metadata_persist_failed",
    ]
    cleanup_status: Literal["not-needed", "completed", "failed"]


class UploadErrorResponse(BaseModel):
    detail: UploadErrorDetail


class ChatRequest(BaseModel):
    document_id: str
    question: Optional[str] = None
    message: Optional[str] = None
    conversation_id: Optional[str] = None


class AnswerCitation(BaseModel):
    chunk_id: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    intent: Literal["summary", "qa"]
    retrieval_mode: Literal["head", "semantic"]
    answer_status: Literal["answered", "insufficient_context"]
    citations: List[AnswerCitation]


class ConversationCreateRequest(BaseModel):
    document_id: str


class ConversationCreateResponse(BaseModel):
    conversation_id: str


class ConversationRecord(BaseModel):
    id: str
    user_id: str
    document_id: str
    created_at: Optional[str] = None


class ConversationsResponse(BaseModel):
    conversations: List[ConversationRecord]


class MessageRecord(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: Optional[str] = None


class ConversationMessagesResponse(BaseModel):
    messages: List[MessageRecord]


class DocumentRecord(BaseModel):
    id: str
    user_id: str
    filename: str
    storage_url: str
    uploaded_at: Optional[str] = None


class DocumentsResponse(BaseModel):
    documents: List[DocumentRecord]


class DeleteErrorDetail(BaseModel):
    message: str
    lifecycle_status: Literal["failed"]
    failure_stage: Literal["conversations", "indexing", "storage", "metadata"]
    reason_code: Literal[
        "conversation_lookup_failed",
        "storage_delete_failed",
        "metadata_delete_failed",
        "conversation_cleanup_failed",
        "indexing_cleanup_failed",
    ]
    cleanup_status: Literal["not-started", "partial", "completed"]


class DeleteErrorResponse(BaseModel):
    detail: DeleteErrorDetail


class DeleteDocumentResponse(BaseModel):
    deleted: Literal[True]
    lifecycle_status: Literal["deleted"] = Field(..., description="Lifecycle state for the delete flow")
    cleanup_status: Literal["completed"] = Field(..., description="Cleanup outcome for the delete flow")
