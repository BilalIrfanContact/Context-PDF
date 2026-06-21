import test from "node:test";
import assert from "node:assert/strict";

import { UPLOAD_FILE_ACCEPT, isSupportedUploadFile } from "./uploadValidation";

test("accept string includes both pdf and markdown filters", () => {
  assert.equal(UPLOAD_FILE_ACCEPT, ".pdf,.md,text/markdown,application/pdf");
});

test("accepts markdown files by extension when the browser reports a generic mime type", () => {
  assert.equal(isSupportedUploadFile({ name: "notes.md", type: "application/octet-stream" }), true);
});

test("accepts markdown files by mime type", () => {
  assert.equal(isSupportedUploadFile({ name: "notes.tmp", type: "text/markdown" }), true);
});

test("rejects unsupported file types", () => {
  assert.equal(isSupportedUploadFile({ name: "notes.txt", type: "text/plain" }), false);
});
