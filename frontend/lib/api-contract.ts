// Generated from backend FastAPI OpenAPI schema. Do not edit by hand.

export interface AnswerCitation {
  "chunk_id": string;
  "excerpt": string;
}

export interface Body_upload_pdf_upload_post {
  "file": Blob;
}

export interface ChatRequest {
  "conversation_id"?: string | null;
  "document_id": string;
  "message"?: string | null;
  "question"?: string | null;
}

export interface ChatResponse {
  "answer": string;
  "answer_status": "answered" | "insufficient_context";
  "citations": AnswerCitation[];
  "intent": "summary" | "qa";
  "retrieval_mode": "head" | "semantic";
}

export interface ConversationCreateRequest {
  "document_id": string;
}

export interface ConversationCreateResponse {
  "conversation_id": string;
}

export interface ConversationMessagesResponse {
  "messages": MessageRecord[];
}

export interface ConversationRecord {
  "created_at"?: string | null;
  "document_id": string;
  "id": string;
  "user_id": string;
}

export interface ConversationsResponse {
  "conversations": ConversationRecord[];
}

export interface DeleteDocumentResponse {
  "cleanup_status": "completed";
  "deleted": true;
  "lifecycle_status": "deleted";
}

export interface DeleteErrorDetail {
  "cleanup_status": "not-started" | "partial" | "completed";
  "failure_stage": "conversations" | "indexing" | "storage" | "metadata";
  "lifecycle_status": "failed";
  "message": string;
  "reason_code": "conversation_lookup_failed" | "storage_delete_failed" | "metadata_delete_failed" | "conversation_cleanup_failed" | "indexing_cleanup_failed";
}

export interface DeleteErrorResponse {
  "detail": DeleteErrorDetail;
}

export interface DocumentRecord {
  "filename": string;
  "id": string;
  "storage_url": string;
  "uploaded_at"?: string | null;
  "user_id": string;
}

export interface DocumentsResponse {
  "documents": DocumentRecord[];
}

export interface ErrorDetailResponse {
  "detail": string;
}

export interface HTTPValidationError {
  "detail"?: ValidationError[];
}

export interface MessageRecord {
  "content": string;
  "conversation_id": string;
  "created_at"?: string | null;
  "id": string;
  "role": string;
}

export interface UploadErrorDetail {
  "cleanup_status": "not-needed" | "completed" | "failed";
  "failure_stage": "validation" | "indexing" | "storage" | "metadata";
  "lifecycle_status": "failed" | "rejected";
  "message": string;
  "reason_code": "invalid_file_type" | "unreadable_document" | "no_extractable_text" | "no_usable_chunks" | "indexing_failed" | "no_chunks_stored" | "storage_upload_failed" | "metadata_persist_failed";
}

export interface UploadErrorResponse {
  "detail": UploadErrorDetail;
}

export interface UploadResponse {
  "chunk_count": number;
  "document_id": string;
  "lifecycle_status": "ready";
  "status": "success";
  "stored_count": number;
}

export interface ValidationError {
  "loc": (string | number)[];
  "msg": string;
  "type": string;
}

export type UploadPdfRequestBody = Body_upload_pdf_upload_post;
export type UploadPdfResponse = UploadResponse;
export type UploadPdfErrorResponse = UploadErrorResponse | ErrorDetailResponse | HTTPValidationError;
export type ChatRequestBody = ChatRequest;
export type ChatResponseBody = ChatResponse;
export type CreateConversationRequestBody = ConversationCreateRequest;
export type CreateConversationResponseBody = ConversationCreateResponse;
export type GetUserConversationsResponseBody = ConversationsResponse;
export type GetConversationMessagesResponseBody = ConversationMessagesResponse;
export type GetUserDocumentsResponseBody = DocumentsResponse;
export type DeleteUserDocumentResponseBody = DeleteDocumentResponse;
export type DeleteUserDocumentErrorResponse = ErrorDetailResponse | DeleteErrorResponse | HTTPValidationError;
