const SUPPORTED_UPLOAD_EXTENSIONS = new Set([".pdf", ".md"]);
const SUPPORTED_UPLOAD_MIME_TYPES = new Set(["application/pdf", "text/markdown"]);

export const UPLOAD_FILE_ACCEPT = ".pdf,.md,text/markdown,application/pdf";

export function isSupportedUploadFile(file: Pick<File, "name" | "type">) {
  const normalizedType = file.type.toLowerCase();
  if (SUPPORTED_UPLOAD_MIME_TYPES.has(normalizedType)) {
    return true;
  }

  const normalizedName = file.name.toLowerCase();
  return Array.from(SUPPORTED_UPLOAD_EXTENSIONS).some((extension) => normalizedName.endsWith(extension));
}
