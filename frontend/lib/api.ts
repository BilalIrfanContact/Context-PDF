import type {
  ChatResponseBody,
  ChatRequestBody,
  ConversationRecord,
  CreateConversationRequestBody,
  CreateConversationResponseBody,
  DeleteErrorDetail,
  DeleteUserDocumentResponseBody,
  DocumentRecord,
  GetConversationMessagesResponseBody,
  GetUserConversationsResponseBody,
  GetUserDocumentsResponseBody,
  MessageRecord,
  UploadErrorDetail,
  UploadPdfResponse
} from "./api-contract";

const API_BASE = "/api";

async function getErrorMessage(res: Response, fallback: string) {
  const error = await res.json().catch(() => ({ detail: fallback }));
  if (typeof error?.detail === "string") {
    return error.detail;
  }

  if (typeof error?.detail?.message === "string") {
    return error.detail.message;
  }

  return fallback;
}

type UploadLifecycleStatus = UploadErrorDetail["lifecycle_status"] | UploadPdfResponse["lifecycle_status"];
type UploadFailureStage = UploadErrorDetail["failure_stage"];
type UploadCleanupStatus = UploadErrorDetail["cleanup_status"];
type UploadReasonCode = UploadErrorDetail["reason_code"];
type DeleteLifecycleStatus = DeleteErrorDetail["lifecycle_status"] | DeleteUserDocumentResponseBody["lifecycle_status"];
type DeleteFailureStage = DeleteErrorDetail["failure_stage"];
type DeleteCleanupStatus = DeleteErrorDetail["cleanup_status"] | DeleteUserDocumentResponseBody["cleanup_status"];
type DeleteReasonCode = DeleteErrorDetail["reason_code"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseUploadReasonCode(
  detail: Partial<UploadErrorDetail> & Record<string, unknown>
): UploadReasonCode | null {
  if (
    detail.reason_code === "invalid_file_type" ||
    detail.reason_code === "no_extractable_text" ||
    detail.reason_code === "no_usable_chunks" ||
    detail.reason_code === "indexing_failed" ||
    detail.reason_code === "no_chunks_stored" ||
    detail.reason_code === "storage_upload_failed" ||
    detail.reason_code === "metadata_persist_failed"
  ) {
    return detail.reason_code;
  }

  if (detail.failure_stage === "storage") {
    return "storage_upload_failed";
  }

  if (detail.failure_stage === "metadata") {
    return "metadata_persist_failed";
  }

  if (detail.failure_stage === "indexing") {
    return typeof detail.message === "string" &&
      detail.message.includes("Chunks were created but not stored")
      ? "no_chunks_stored"
      : "indexing_failed";
  }

  if (detail.failure_stage === "validation" && typeof detail.message === "string") {
    if (detail.message.includes("Only PDF and Markdown files are supported")) {
      return "invalid_file_type";
    }

    if (detail.message.includes("No extractable text found")) {
      return "no_extractable_text";
    }

    if (detail.message.includes("No usable text chunks were created")) {
      return "no_usable_chunks";
    }
  }

  return null;
}

function parseDeleteReasonCode(
  detail: Partial<DeleteErrorDetail> & Record<string, unknown>
): DeleteReasonCode | null {
  if (
    detail.reason_code === "conversation_lookup_failed" ||
    detail.reason_code === "storage_delete_failed" ||
    detail.reason_code === "metadata_delete_failed" ||
    detail.reason_code === "conversation_cleanup_failed" ||
    detail.reason_code === "indexing_cleanup_failed"
  ) {
    return detail.reason_code;
  }

  if (detail.failure_stage === "storage") {
    return "storage_delete_failed";
  }

  if (detail.failure_stage === "metadata") {
    return "metadata_delete_failed";
  }

  if (detail.failure_stage === "indexing") {
    return "indexing_cleanup_failed";
  }

  if (detail.failure_stage === "conversations") {
    return detail.cleanup_status === "not-started"
      ? "conversation_lookup_failed"
      : "conversation_cleanup_failed";
  }

  return null;
}

export class UploadFlowError extends Error {
  lifecycleStatus: UploadLifecycleStatus;
  failureStage: UploadFailureStage | null;
  reasonCode: UploadReasonCode | null;
  cleanupStatus: UploadCleanupStatus | null;

  constructor(
    message: string,
    options?: {
      lifecycleStatus?: UploadLifecycleStatus;
      failureStage?: UploadFailureStage | null;
      reasonCode?: UploadReasonCode | null;
      cleanupStatus?: UploadCleanupStatus | null;
    }
  ) {
    super(message);
    this.name = "UploadFlowError";
    this.lifecycleStatus = options?.lifecycleStatus ?? "failed";
    this.failureStage = options?.failureStage ?? null;
    this.reasonCode = options?.reasonCode ?? null;
    this.cleanupStatus = options?.cleanupStatus ?? null;
  }
}

export class DeleteFlowError extends Error {
  lifecycleStatus: DeleteLifecycleStatus;
  failureStage: DeleteFailureStage | null;
  reasonCode: DeleteReasonCode | null;
  cleanupStatus: DeleteCleanupStatus | null;

  constructor(
    message: string,
    options?: {
      lifecycleStatus?: DeleteLifecycleStatus;
      failureStage?: DeleteFailureStage | null;
      reasonCode?: DeleteReasonCode | null;
      cleanupStatus?: DeleteCleanupStatus | null;
    }
  ) {
    super(message);
    this.name = "DeleteFlowError";
    this.lifecycleStatus = options?.lifecycleStatus ?? "failed";
    this.failureStage = options?.failureStage ?? null;
    this.reasonCode = options?.reasonCode ?? null;
    this.cleanupStatus = options?.cleanupStatus ?? null;
  }
}

export type PersistedDocument = DocumentRecord;
export type PersistedConversation = ConversationRecord;
export type PersistedMessage = MessageRecord;

export async function uploadPdf(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: { message: "Upload failed." } }));
    const detail = error?.detail;

    if (isRecord(detail)) {
      throw new UploadFlowError(
        typeof detail.message === "string" ? detail.message : "Upload failed.",
        {
          lifecycleStatus:
            detail.lifecycle_status === "rejected" || detail.lifecycle_status === "failed"
              ? detail.lifecycle_status
              : "failed",
          failureStage:
            detail.failure_stage === "validation" ||
            detail.failure_stage === "indexing" ||
            detail.failure_stage === "storage" ||
            detail.failure_stage === "metadata"
              ? detail.failure_stage
              : null,
          reasonCode: parseUploadReasonCode(detail),
          cleanupStatus:
            detail.cleanup_status === "not-needed" ||
            detail.cleanup_status === "completed" ||
            detail.cleanup_status === "failed"
              ? detail.cleanup_status
              : null
        }
      );
    }

    throw new UploadFlowError(typeof detail === "string" ? detail : "Upload failed.");
  }

  return res.json() as Promise<UploadPdfResponse>;
}

export async function createConversation(documentId: string) {
  const body: CreateConversationRequestBody = { document_id: documentId };

  const res = await fetch(`${API_BASE}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Failed to create conversation."));
  }

  return res.json() as Promise<CreateConversationResponseBody>;
}

export async function getUserDocuments() {
  const res = await fetch(`${API_BASE}/documents`, {
    method: "GET",
    cache: "no-store"
  });

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Failed to load documents."));
  }

  const data = (await res.json()) as GetUserDocumentsResponseBody;
  return data.documents;
}

export async function deleteUserDocument(documentId: string) {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE"
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: { message: "Failed to delete document." } }));
    const detail = error?.detail;

    if (isRecord(detail)) {
      throw new DeleteFlowError(
        typeof detail.message === "string" ? detail.message : "Failed to delete document.",
        {
          lifecycleStatus: detail.lifecycle_status === "failed" ? "failed" : "failed",
          failureStage:
            detail.failure_stage === "conversations" ||
            detail.failure_stage === "indexing" ||
            detail.failure_stage === "storage" ||
            detail.failure_stage === "metadata"
              ? detail.failure_stage
              : null,
          reasonCode: parseDeleteReasonCode(detail),
          cleanupStatus:
            detail.cleanup_status === "not-started" ||
            detail.cleanup_status === "partial" ||
            detail.cleanup_status === "completed"
              ? detail.cleanup_status
              : null
        }
      );
    }

    throw new DeleteFlowError(typeof detail === "string" ? detail : "Failed to delete document.");
  }

  return res.json() as Promise<DeleteUserDocumentResponseBody>;
}

export async function getUserConversations(documentId?: string) {
  const query = new URLSearchParams();
  if (documentId) {
    query.set("document_id", documentId);
  }

  const res = await fetch(`${API_BASE}/conversations?${query.toString()}`, {
    method: "GET",
    cache: "no-store"
  });

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Failed to load conversations."));
  }

  const data = (await res.json()) as GetUserConversationsResponseBody;
  return data.conversations;
}

export async function getConversationMessages(conversationId: string) {
  const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}/messages`, {
    method: "GET",
    cache: "no-store"
  });

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Failed to load messages."));
  }

  const data = (await res.json()) as GetConversationMessagesResponseBody;
  return data.messages;
}

export async function askQuestion(input: {
  documentId: string;
  conversationId: string;
  message: string;
}) {
  const body: ChatRequestBody = {
    document_id: input.documentId,
    conversation_id: input.conversationId,
    message: input.message
  };

  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    throw new Error(await getErrorMessage(res, "Chat failed."));
  }

  return res.json() as Promise<ChatResponseBody>;
}
