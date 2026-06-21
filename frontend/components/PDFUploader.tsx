import { useEffect, useRef, useState } from "react";
import { UploadFlowError, uploadPdf } from "../lib/api";
import { isSupportedUploadFile, UPLOAD_FILE_ACCEPT } from "../lib/uploadValidation";
import type { UploadBootstrapResult } from "./home/types";

type PDFUploaderProps = {
  onUploaded: (
    documentId: string,
    meta: { fileName: string; fileSize: string; chunkCount: number; storedCount: number }
  ) => Promise<UploadBootstrapResult>;
  onClear: () => void;
  activeDocumentId: string | null;
  resetSignal: number;
};

type UploadStage = "idle" | "uploading" | "finalizing" | "ready" | "attention" | "error";

export default function PDFUploader({
  onUploaded,
  onClear,
  activeDocumentId,
  resetSignal
}: PDFUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<UploadStage>("idle");
  const [fileInfo, setFileInfo] = useState<{ name: string; size: string } | null>(null);

  useEffect(() => {
    setStatus("");
    setLoading(false);
    setStage("idle");
    setFileInfo(null);
  }, [resetSignal]);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    const value = bytes / Math.pow(1024, i);
    return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
  };

  const handleFileChange = async (file: File | null) => {
    if (!file) return;

    if (!isSupportedUploadFile(file)) {
      setStatus("Please select a valid PDF or Markdown file.");
      setStage("error");
      return;
    }

    try {
      setLoading(true);
      setStage("uploading");
      const nextFileInfo = { name: file.name, size: formatBytes(file.size) };
      setFileInfo(nextFileInfo);
      setStatus("Uploading your document...");

      const response = await uploadPdf(file);
      setStage("finalizing");
      setStatus(
        response.lifecycle_status === "ready"
          ? "Indexing complete. Preparing your chat workspace..."
          : "Preparing your document..."
      );
      const uploadResult = await onUploaded(response.document_id, {
        fileName: file.name,
        fileSize: nextFileInfo.size,
        chunkCount: response.chunk_count,
        storedCount: response.stored_count
      });

      if (uploadResult.status === "cancelled") {
        return;
      }

      if (uploadResult.status === "ready") {
        setStage("ready");
        setStatus("Document indexed and ready for questions.");
        return;
      }

      setStage("attention");
      setStatus(uploadResult.message);
    } catch (error) {
      const message = getUploadErrorMessage(error);
      setStatus(message);
      setStage("error");
    } finally {
      setLoading(false);
    }
  };

  const UploadIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );

  return (
    <div className="uploader-wrap">
      <div className="upload-container">
        <button
          type="button"
          className="upload-input-bar"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Choose document file"
        >
          <div className="upload-input-icon">
            <UploadIcon />
          </div>
          <div className="upload-input-content">
            <span className="upload-input-text">Choose a PDF or Markdown file to analyze...</span>
          </div>
        </button>

        {stage === "uploading" || stage === "finalizing" ? (
          <div className="upload-progress-container">
            <div className="upload-progress-fill" />
          </div>
        ) : null}
      </div>

      <div className="upload-suggestions">
        <button type="button" className="suggestion-pill">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
          Summarize
        </button>
        <button type="button" className="suggestion-pill">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          Explain
        </button>
        <button type="button" className="suggestion-pill">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          Quick summary
        </button>
        <button type="button" className="suggestion-pill">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Find answers
        </button>
      </div>

      {stage === "uploading" || stage === "finalizing" ? (
        <div className="loader-container">
          <div className="spinner" />
          <span className="loader-text">
            {stage === "uploading" ? "Uploading and indexing your document..." : "Preparing your chat workspace..."}
          </span>
        </div>
      ) : null}

      {fileInfo && stage !== "uploading" ? (
        <div className="upload-meta">
          <span style={{color: "var(--color-near-black)"}}>{fileInfo.name}</span>
          <span className="uploader-status">{fileInfo.size}</span>
        </div>
      ) : null}

      <input
        ref={fileInputRef}
        type="file"
        accept={UPLOAD_FILE_ACCEPT}
        style={{display: 'none'}}
        onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
      />

      {status && (stage === "error" || stage === "attention") ? (
        <p
          className="uploader-status"
          style={{color: stage === "error" ? 'var(--color-error)' : "var(--color-near-black)", marginTop: '8px'}}
        >
          {status}
        </p>
      ) : null}

      {activeDocumentId ? (
        <button type="button" onClick={onClear} className="btn-ghost" style={{marginTop: "16px"}}>
          Clear document
        </button>
      ) : null}
    </div>
  );
}

function getUploadErrorMessage(error: unknown) {
  if (!(error instanceof Error)) {
    return "Upload failed.";
  }

  if (!(error instanceof UploadFlowError)) {
    return error.message;
  }

  if (
    error.reasonCode === "storage_upload_failed" ||
    error.reasonCode === "metadata_persist_failed"
  ) {
    const recoveryNote =
      error.cleanupStatus === "completed"
        ? " Partial upload data was rolled back."
        : error.cleanupStatus === "failed"
          ? " Cleanup may be incomplete. Retry once and remove any partial document entry if it appears."
          : "";

    return `${error.message}${recoveryNote}`;
  }

  if (error.reasonCode === "no_chunks_stored") {
    return "Document indexing did not store any chunks. Check the embedding configuration and try again.";
  }

  return error.message;
}
