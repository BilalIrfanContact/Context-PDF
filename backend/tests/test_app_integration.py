import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from urllib.parse import urlsplit
from unittest.mock import Mock, patch

from fastapi import FastAPI, Request

from backend.routers import chat, conversations, documents, upload
from backend.services.internal_auth import require_authenticated_user
from backend.services.rag_pipeline import AnswerDecision, INSUFFICIENT_CONTEXT_ANSWER


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(upload.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(documents.router)

    async def override_user(request: Request) -> str:
        return request.headers.get("x-test-user", "user-a")

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

    split = urlsplit(path)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": split.path,
        "raw_path": split.path.encode("utf-8"),
        "query_string": split.query.encode("utf-8"),
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    await app(scope, receive, send)
    return response_status, response_headers, bytes(response_body)


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakePostgrestQuery:
    def __init__(self, state: "InMemoryAppState", table: str):
        self.state = state
        self.table = table
        self._action = "select"
        self._filters: list[tuple[str, object]] = []
        self._selected_fields: list[str] | None = None
        self._limit: int | None = None
        self._order_field: str | None = None
        self._order_desc = False
        self._insert_payload: dict | None = None

    def select(self, fields: str):
        self._action = "select"
        self._selected_fields = [field.strip() for field in fields.split(",")]
        return self

    def eq(self, field: str, value):
        self._filters.append((field, value))
        return self

    def limit(self, value: int):
        self._limit = value
        return self

    def order(self, field: str, desc: bool = False):
        self._order_field = field
        self._order_desc = desc
        return self

    def insert(self, payload: dict):
        self._action = "insert"
        self._insert_payload = dict(payload)
        return self

    def delete(self):
        self._action = "delete"
        return self

    def execute(self):
        if self._action == "insert":
            return FakeResponse(self.state.insert_row(self.table, self._insert_payload or {}))

        if self._action == "delete":
            self.state.delete_rows(self.table, self._filters)
            return FakeResponse([])

        rows = self.state.select_rows(
            self.table,
            filters=self._filters,
            selected_fields=self._selected_fields,
            order_field=self._order_field,
            order_desc=self._order_desc,
            limit=self._limit,
        )
        return FakeResponse(rows)


class FakePostgrestClient:
    def __init__(self, state: "InMemoryAppState"):
        self.state = state

    def from_(self, table: str) -> FakePostgrestQuery:
        return FakePostgrestQuery(self.state, table)


class InMemoryAppState:
    def __init__(self):
        self.documents: dict[str, dict] = {
            "doc-a": {
                "id": "doc-a",
                "user_id": "user-a",
                "filename": "alpha.pdf",
                "storage_url": "documents/user-a/doc-a/alpha.pdf",
                "uploaded_at": "2026-06-11T10:00:00Z",
            },
            "doc-b": {
                "id": "doc-b",
                "user_id": "user-b",
                "filename": "beta.pdf",
                "storage_url": "documents/user-b/doc-b/beta.pdf",
                "uploaded_at": "2026-06-11T10:05:00Z",
            },
        }
        self.conversations: dict[str, dict] = {
            "convo-b": {
                "id": "convo-b",
                "user_id": "user-b",
                "document_id": "doc-b",
                "created_at": "2026-06-11T10:10:00Z",
            }
        }
        self.messages: list[dict] = [
            {
                "id": "msg-b-1",
                "conversation_id": "convo-b",
                "role": "user",
                "content": "private question",
                "created_at": "2026-06-11T10:11:00Z",
            }
        ]
        self.storage_deleted: list[str] = []
        self.storage_uploaded: list[str] = []
        self.vector_deleted: list[str] = []
        self.conversation_counter = 0
        self.message_counter = 0
        self.upload_counter = 0
        self.document_counter = 0

    @staticmethod
    def _matches_filters(row: dict, filters: list[tuple[str, object]]) -> bool:
        return all(row.get(field) == value for field, value in filters)

    @staticmethod
    def _project_row(row: dict, selected_fields: list[str] | None) -> dict:
        if not selected_fields:
            return dict(row)
        return {field: row.get(field) for field in selected_fields}

    def select_rows(
        self,
        table: str,
        *,
        filters: list[tuple[str, object]],
        selected_fields: list[str] | None,
        order_field: str | None,
        order_desc: bool,
        limit: int | None,
    ) -> list[dict]:
        if table == "documents":
            rows = list(self.documents.values())
        elif table == "conversations":
            rows = list(self.conversations.values())
        elif table == "messages":
            rows = list(self.messages)
        else:
            raise AssertionError(f"Unexpected table: {table}")

        rows = [row for row in rows if self._matches_filters(row, filters)]
        if order_field:
            rows = sorted(rows, key=lambda item: item.get(order_field) or "", reverse=order_desc)
        if limit is not None:
            rows = rows[:limit]
        return [self._project_row(row, selected_fields) for row in rows]

    def insert_row(self, table: str, payload: dict) -> list[dict]:
        row = dict(payload)

        if table == "documents":
            self.document_counter += 1
            row.setdefault("uploaded_at", f"2026-06-11T13:{self.document_counter:02d}:00Z")
            self.documents[row["id"]] = row
        elif table == "conversations":
            self.conversation_counter += 1
            self.conversations[row["id"]] = {
                **row,
                "created_at": f"2026-06-11T11:{self.conversation_counter:02d}:00Z",
            }
            row = self.conversations[row["id"]]
        elif table == "messages":
            self.message_counter += 1
            self.messages.append(
                {
                    **row,
                    "created_at": f"2026-06-11T12:{self.message_counter:02d}:00Z",
                }
            )
            row = self.messages[-1]
        else:
            raise AssertionError(f"Unexpected table: {table}")

        return [dict(row)]

    def delete_rows(self, table: str, filters: list[tuple[str, object]]) -> None:
        if table == "documents":
            for document_id, row in list(self.documents.items()):
                if self._matches_filters(row, filters):
                    del self.documents[document_id]
            return

        if table == "conversations":
            for conversation_id, row in list(self.conversations.items()):
                if self._matches_filters(row, filters):
                    del self.conversations[conversation_id]
            return

        if table == "messages":
            self.messages = [row for row in self.messages if not self._matches_filters(row, filters)]
            return

        raise AssertionError(f"Unexpected table: {table}")

    def upload_storage_object(
        self,
        *,
        user_id: str,
        document_id: str,
        filename: str,
        data: bytes,
        content_type: str,
    ) -> str:
        del data, content_type
        self.upload_counter += 1
        storage_url = f"documents/{user_id}/{document_id}/{filename}"
        self.storage_uploaded.append(storage_url)
        return storage_url

    def delete_storage_object(self, storage_url: str) -> None:
        self.storage_deleted.append(storage_url)

    def delete_vector_store(self, document_id: str) -> None:
        self.vector_deleted.append(document_id)


class FakeVectorCollection:
    def __init__(self, *, count: int, query_result: dict):
        self._count = count
        self._query_result = query_result
        self.query_call_count = 0

    def count(self) -> int:
        return self._count

    def query(self, *, query_embeddings, n_results: int, include: list[str]) -> dict:
        self.query_call_count += 1
        return self._query_result


class FakeVectorStore:
    def __init__(
        self,
        *,
        count: int = 4,
        head_documents: list[str] | None = None,
        head_metadatas: list[dict | None] | None = None,
        head_ids: list[str | None] | None = None,
        query_documents: list[str | None] | None = None,
        query_metadatas: list[dict | None] | None = None,
        query_ids: list[str | None] | None = None,
    ):
        self.embeddings = SimpleNamespace(embed_query=lambda question: [0.1, 0.2, 0.3])
        self.get_call_count = 0
        self._head_result = {
            "documents": head_documents or [],
            "metadatas": head_metadatas or [],
            "ids": head_ids or [],
        }
        self._collection = FakeVectorCollection(
            count=count,
            query_result={
                "documents": [query_documents or []],
                "metadatas": [query_metadatas or []],
                "ids": [query_ids or []],
            },
        )

    def get(self, *, limit: int, include: list[str]) -> dict:
        self.get_call_count += 1
        return self._head_result


class AppIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = _build_test_app()
        self.state = InMemoryAppState()
        self.postgrest_client = FakePostgrestClient(self.state)
        self.exit_stack = ExitStack()

        for target in (
            "backend.services.persistence.documents_repository.get_postgrest_client",
            "backend.services.persistence.conversations_repository.get_postgrest_client",
            "backend.services.persistence.messages_repository.get_postgrest_client",
        ):
            self.exit_stack.enter_context(patch(target, return_value=self.postgrest_client))
        self.exit_stack.enter_context(
            patch(
                "backend.services.persistence.conversations_repository.uuid",
                new=SimpleNamespace(uuid4=lambda: f"convo-{self.state.conversation_counter + 1}"),
            )
        )
        self.exit_stack.enter_context(
            patch(
                "backend.services.persistence.messages_repository.uuid",
                new=SimpleNamespace(uuid4=lambda: f"msg-{self.state.message_counter + 1}"),
            )
        )

        self.exit_stack.enter_context(
            patch(
                "backend.routers.chat.answer_question",
                return_value=AnswerDecision(
                    answer="Document summary answer",
                    intent="summary",
                    retrieval_mode="head",
                    answer_status="answered",
                    citations=[],
                ),
            )
        )

        self.exit_stack.enter_context(
            patch("backend.services.document_lifecycle.extract_text_from_pdf", return_value="alpha beta")
        )
        self.exit_stack.enter_context(
            patch("backend.services.document_lifecycle.chunk_text", return_value=["alpha", "beta"])
        )
        self.exit_stack.enter_context(
            patch("backend.services.document_lifecycle.build_vector_store", return_value=2)
        )
        self.exit_stack.enter_context(
            patch(
                "backend.services.document_lifecycle.upload_file_to_storage",
                side_effect=self.state.upload_storage_object,
            )
        )
        self.exit_stack.enter_context(
            patch(
                "backend.services.document_lifecycle.uuid",
                new=SimpleNamespace(uuid4=lambda: f"doc-upload-{self.state.upload_counter + 1}"),
            )
        )
        self.exit_stack.enter_context(
            patch(
                "backend.services.document_lifecycle.delete_storage_object",
                side_effect=self.state.delete_storage_object,
            )
        )
        self.exit_stack.enter_context(
            patch(
                "backend.services.document_lifecycle.delete_vector_store",
                side_effect=self.state.delete_vector_store,
            )
        )

    def tearDown(self):
        self.exit_stack.close()
        self.app.dependency_overrides.clear()

    async def test_auth_ownership_constraints_are_enforced_at_http_boundary(self):
        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path="/conversations?document_id=doc-b",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(
            json.loads(response_body),
            {"detail": "You are not authorized to access this document."},
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path="/conversations/convo-b/messages",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(
            json.loads(response_body),
            {"detail": "You are not authorized to access this conversation."},
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="DELETE",
            path="/documents/doc-b",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 403)
        self.assertEqual(
            json.loads(response_body),
            {"detail": "You are not authorized to access this document."},
        )

    async def test_upload_delete_consistency_is_visible_across_documents_and_conversations(self):
        boundary = "boundary123"
        body = _multipart_request_body(
            field_name="file",
            filename="report.pdf",
            content_type="application/pdf",
            payload=b"%PDF-1.4",
            boundary=boundary,
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/upload",
            body=body,
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode("utf-8")),
                (b"content-length", str(len(body)).encode("utf-8")),
            ],
        )
        self.assertEqual(status, 200)
        upload_payload = json.loads(response_body)
        document_id = upload_payload["document_id"]
        self.assertIn(document_id, self.state.documents)
        self.assertIn(f"documents/user-a/{document_id}/report.pdf", self.state.storage_uploaded)

        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/conversations",
            body=json.dumps({"document_id": document_id}).encode("utf-8"),
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", b"application/json"),
            ],
        )
        self.assertEqual(status, 200)
        conversation_id = json.loads(response_body)["conversation_id"]

        status, _, response_body = await _request_asgi(
            self.app,
            method="DELETE",
            path=f"/documents/{document_id}",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(response_body),
            {
                "deleted": True,
                "lifecycle_status": "deleted",
                "cleanup_status": "completed",
            },
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path="/documents",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 200)
        self.assertNotIn(
            document_id,
            [document["id"] for document in json.loads(response_body)["documents"]],
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path=f"/conversations/{conversation_id}/messages",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(response_body), {"detail": "Conversation not found."})
        self.assertIn(document_id, self.state.vector_deleted)

    async def test_chat_persistence_and_retrieval_work_across_requests(self):
        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/conversations",
            body=json.dumps({"document_id": "doc-a"}).encode("utf-8"),
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", b"application/json"),
            ],
        )
        self.assertEqual(status, 200)
        conversation_id = json.loads(response_body)["conversation_id"]

        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/chat",
            body=json.dumps(
                {
                    "document_id": "doc-a",
                    "conversation_id": conversation_id,
                    "message": "Summarize the document",
                }
            ).encode("utf-8"),
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", b"application/json"),
            ],
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(response_body),
            {
                "answer": "Document summary answer",
                "intent": "summary",
                "retrieval_mode": "head",
                "answer_status": "answered",
                "citations": [],
            },
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path="/conversations?document_id=doc-a",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            [conversation["id"] for conversation in json.loads(response_body)["conversations"]],
            [conversation_id],
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path=f"/conversations/{conversation_id}/messages",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(response_body)["messages"],
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "Summarize the document",
                    "created_at": "2026-06-11T12:01:00Z",
                },
                {
                    "id": "msg-2",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": "Document summary answer",
                    "created_at": "2026-06-11T12:02:00Z",
                },
            ],
        )

    async def test_chat_returns_structured_502_when_rag_pipeline_fails(self):
        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/conversations",
            body=json.dumps({"document_id": "doc-a"}).encode("utf-8"),
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", b"application/json"),
            ],
        )
        self.assertEqual(status, 200)
        conversation_id = json.loads(response_body)["conversation_id"]

        with patch(
            "backend.routers.chat.answer_question",
            side_effect=RuntimeError("model unavailable"),
        ):
            status, _, response_body = await _request_asgi(
                self.app,
                method="POST",
                path="/chat",
                body=json.dumps(
                    {
                        "document_id": "doc-a",
                        "conversation_id": conversation_id,
                        "message": "Summarize the document",
                    }
                ).encode("utf-8"),
                headers=[
                    (b"x-test-user", b"user-a"),
                    (b"content-type", b"application/json"),
                ],
            )

        self.assertEqual(status, 502)
        self.assertEqual(
            json.loads(response_body),
            {"detail": "model unavailable"},
        )

        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path=f"/conversations/{conversation_id}/messages",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(response_body)["messages"],
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "Summarize the document",
                    "created_at": "2026-06-11T12:01:00Z",
                }
            ],
        )


class ChatPipelineIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = _build_test_app()
        self.state = InMemoryAppState()
        self.postgrest_client = FakePostgrestClient(self.state)
        self.exit_stack = ExitStack()

        for target in (
            "backend.services.persistence.documents_repository.get_postgrest_client",
            "backend.services.persistence.conversations_repository.get_postgrest_client",
            "backend.services.persistence.messages_repository.get_postgrest_client",
        ):
            self.exit_stack.enter_context(patch(target, return_value=self.postgrest_client))

        self.exit_stack.enter_context(
            patch(
                "backend.services.persistence.conversations_repository.uuid",
                new=SimpleNamespace(uuid4=lambda: f"convo-{self.state.conversation_counter + 1}"),
            )
        )
        self.exit_stack.enter_context(
            patch(
                "backend.services.persistence.messages_repository.uuid",
                new=SimpleNamespace(uuid4=lambda: f"msg-{self.state.message_counter + 1}"),
            )
        )

    def tearDown(self):
        self.exit_stack.close()
        self.app.dependency_overrides.clear()

    async def _create_conversation(self, document_id: str = "doc-a") -> str:
        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/conversations",
            body=json.dumps({"document_id": document_id}).encode("utf-8"),
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", b"application/json"),
            ],
        )
        self.assertEqual(status, 200)
        return json.loads(response_body)["conversation_id"]

    async def _chat(self, conversation_id: str, message: str) -> tuple[int, dict]:
        status, _, response_body = await _request_asgi(
            self.app,
            method="POST",
            path="/chat",
            body=json.dumps(
                {
                    "document_id": "doc-a",
                    "conversation_id": conversation_id,
                    "message": message,
                }
            ).encode("utf-8"),
            headers=[
                (b"x-test-user", b"user-a"),
                (b"content-type", b"application/json"),
            ],
        )
        return status, json.loads(response_body)

    async def _messages(self, conversation_id: str) -> list[dict]:
        status, _, response_body = await _request_asgi(
            self.app,
            method="GET",
            path=f"/conversations/{conversation_id}/messages",
            headers=[(b"x-test-user", b"user-a")],
        )
        self.assertEqual(status, 200)
        return json.loads(response_body)["messages"]

    async def test_chat_runs_real_grounded_answer_path_and_persists_citations(self):
        conversation_id = await self._create_conversation()
        vectordb = FakeVectorStore(
            count=6,
            query_documents=["The refund window is 30 days from the purchase date."],
            query_metadatas=[{"chunk_id": "doc-a:chunk:0"}],
            query_ids=[None],
        )
        llm = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content='{"answer": "The refund window is 30 days."}')
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            status, payload = await self._chat(conversation_id, "What is the refund window?")

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "answer": "The refund window is 30 days.",
                "intent": "qa",
                "retrieval_mode": "semantic",
                "answer_status": "answered",
                "citations": [
                    {
                        "chunk_id": "doc-a:chunk:0",
                        "excerpt": "The refund window is 30 days from the purchase date.",
                    }
                ],
            },
        )
        self.assertEqual(
            await self._messages(conversation_id),
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "What is the refund window?",
                    "created_at": "2026-06-11T12:01:00Z",
                },
                {
                    "id": "msg-2",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": "The refund window is 30 days.",
                    "created_at": "2026-06-11T12:02:00Z",
                },
            ],
        )
        self.assertEqual(vectordb.get_call_count, 0)
        self.assertEqual(vectordb._collection.query_call_count, 1)

    async def test_chat_runs_real_summary_head_retrieval_path_and_persists_citations(self):
        conversation_id = await self._create_conversation()
        vectordb = FakeVectorStore(
            count=6,
            head_documents=["This handbook explains the benefits policy and time-off rules."],
            head_metadatas=[{"chunk_id": "doc-a:chunk:0"}],
            head_ids=[None],
        )
        llm = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content='{"answer": "The handbook covers benefits policy and time-off rules."}'
            )
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            status, payload = await self._chat(conversation_id, "Summarize this document.")

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "answer": "The handbook covers benefits policy and time-off rules.",
                "intent": "summary",
                "retrieval_mode": "head",
                "answer_status": "answered",
                "citations": [
                    {
                        "chunk_id": "doc-a:chunk:0",
                        "excerpt": "This handbook explains the benefits policy and time-off rules.",
                    }
                ],
            },
        )
        self.assertEqual(
            await self._messages(conversation_id),
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "Summarize this document.",
                    "created_at": "2026-06-11T12:01:00Z",
                },
                {
                    "id": "msg-2",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": "The handbook covers benefits policy and time-off rules.",
                    "created_at": "2026-06-11T12:02:00Z",
                },
            ],
        )
        self.assertEqual(vectordb.get_call_count, 1)
        self.assertEqual(vectordb._collection.query_call_count, 0)

    async def test_chat_returns_deterministic_fallback_when_retrieval_evidence_is_too_weak(self):
        conversation_id = await self._create_conversation()
        vectordb = FakeVectorStore(
            query_documents=["The onboarding checklist covers payroll setup and laptop pickup."],
            query_metadatas=[{"chunk_id": "doc-a:chunk:3"}],
            query_ids=[None],
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
        ):
            status, payload = await self._chat(conversation_id, "What is the refund window?")

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "intent": "qa",
                "retrieval_mode": "semantic",
                "answer_status": "insufficient_context",
                "citations": [],
            },
        )
        chat_openai_mock.assert_not_called()
        self.assertEqual(
            await self._messages(conversation_id),
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "What is the refund window?",
                    "created_at": "2026-06-11T12:01:00Z",
                },
                {
                    "id": "msg-2",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": INSUFFICIENT_CONTEXT_ANSWER,
                    "created_at": "2026-06-11T12:02:00Z",
                },
            ],
        )

    async def test_chat_rejects_ungrounded_model_answer_before_persisting_response_citations(self):
        conversation_id = await self._create_conversation()
        vectordb = FakeVectorStore(
            query_documents=["The refund window is 30 days from the purchase date."],
            query_metadatas=[{"chunk_id": "doc-a:chunk:0"}],
            query_ids=[None],
        )
        llm = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(
                content='{"answer": "The refund window is 45 days and includes free returns."}'
            )
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            status, payload = await self._chat(conversation_id, "What is the refund window?")

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "intent": "qa",
                "retrieval_mode": "semantic",
                "answer_status": "insufficient_context",
                "citations": [],
            },
        )
        self.assertEqual(
            await self._messages(conversation_id),
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "What is the refund window?",
                    "created_at": "2026-06-11T12:01:00Z",
                },
                {
                    "id": "msg-2",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": INSUFFICIENT_CONTEXT_ANSWER,
                    "created_at": "2026-06-11T12:02:00Z",
                },
            ],
        )

    async def test_chat_falls_back_when_model_repeatedly_breaks_structured_output_contract(self):
        conversation_id = await self._create_conversation()
        vectordb = FakeVectorStore(
            query_documents=["The refund window is 30 days from the purchase date."],
            query_metadatas=[{"chunk_id": "doc-a:chunk:0"}],
            query_ids=[None],
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="The refund window is 30 days."),
            SimpleNamespace(content='{"answer": ""}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            status, payload = await self._chat(conversation_id, "What is the refund window?")

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "intent": "qa",
                "retrieval_mode": "semantic",
                "answer_status": "insufficient_context",
                "citations": [],
            },
        )
        self.assertEqual(
            await self._messages(conversation_id),
            [
                {
                    "id": "msg-1",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "What is the refund window?",
                    "created_at": "2026-06-11T12:01:00Z",
                },
                {
                    "id": "msg-2",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": INSUFFICIENT_CONTEXT_ANSWER,
                    "created_at": "2026-06-11T12:02:00Z",
                },
            ],
        )
        self.assertEqual(llm.invoke.call_count, 2)


if __name__ == "__main__":
    unittest.main()
