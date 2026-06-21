import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend.models.schemas import ChatRequest, ConversationCreateRequest
from backend.routers.chat import chat
from backend.routers.conversations import (
    create_conversation_endpoint,
    get_conversation_messages,
    get_user_conversations,
)
from backend.routers.documents import delete_user_document


class AuthorizationRoutesTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_list_conversations_allows_missing_document_filter(self):
        with (
            patch(
                "backend.routers.conversations.require_user_document",
                side_effect=HTTPException(status_code=404, detail="Document not found."),
            ),
            patch(
                "backend.routers.conversations.list_user_conversations",
                return_value=[],
            ) as list_user_conversations_mock,
        ):
            response = await get_user_conversations(document_id="doc-missing", user_id="user-a")

        self.assertEqual(response.conversations, [])
        list_user_conversations_mock.assert_called_once_with(
            user_id="user-a",
            document_id="doc-missing",
        )

    async def test_list_conversations_rejects_cross_user_document_filter(self):
        with patch(
            "backend.routers.conversations.require_user_document",
            side_effect=HTTPException(
                status_code=403,
                detail="You are not authorized to access this document.",
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await get_user_conversations(document_id="doc-b", user_id="user-a")

        self.assertEqual(exc.exception.status_code, 403)
        self.assertEqual(
            exc.exception.detail,
            "You are not authorized to access this document.",
        )

    async def test_create_conversation_rejects_cross_user_document(self):
        with patch(
            "backend.routers.conversations.require_user_document",
            side_effect=HTTPException(
                status_code=403,
                detail="You are not authorized to access this document.",
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await create_conversation_endpoint(
                    ConversationCreateRequest(document_id="doc-b"),
                    user_id="user-a",
                )

        self.assertEqual(exc.exception.status_code, 403)
        self.assertEqual(
            exc.exception.detail,
            "You are not authorized to access this document.",
        )

    async def test_get_conversation_messages_rejects_cross_user_conversation(self):
        with patch(
            "backend.routers.conversations.require_user_conversation",
            side_effect=HTTPException(
                status_code=403,
                detail="You are not authorized to access this conversation.",
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await get_conversation_messages(conversation_id="convo-b", user_id="user-a")

        self.assertEqual(exc.exception.status_code, 403)
        self.assertEqual(
            exc.exception.detail,
            "You are not authorized to access this conversation.",
        )

    async def test_chat_rejects_cross_user_conversation(self):
        with patch(
            "backend.routers.chat.require_user_conversation",
            side_effect=HTTPException(
                status_code=403,
                detail="You are not authorized to access this conversation.",
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await chat(
                    ChatRequest(
                        document_id="doc-b",
                        conversation_id="convo-b",
                        message="What is in the PDF?",
                    ),
                    user_id="user-a",
                )

        self.assertEqual(exc.exception.status_code, 403)
        self.assertEqual(
            exc.exception.detail,
            "You are not authorized to access this conversation.",
        )

    async def test_chat_rejects_mismatched_owned_document(self):
        with (
            patch(
                "backend.routers.chat.require_user_conversation",
                return_value={"id": "convo-a", "user_id": "user-a", "document_id": "doc-a"},
            ),
            patch(
                "backend.routers.chat.require_user_document",
                return_value={"id": "doc-other", "user_id": "user-a"},
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await chat(
                    ChatRequest(
                        document_id="doc-other",
                        conversation_id="convo-a",
                        message="What is in the PDF?",
                    ),
                    user_id="user-a",
                )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(
            exc.exception.detail,
            "Conversation does not belong to the provided document.",
        )

    async def test_delete_document_rejects_cross_user_document(self):
        with patch(
            "backend.routers.documents.require_user_document",
            side_effect=HTTPException(
                status_code=403,
                detail="You are not authorized to access this document.",
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await delete_user_document(document_id="doc-b", user_id="user-a")

        self.assertEqual(exc.exception.status_code, 403)
        self.assertEqual(
            exc.exception.detail,
            "You are not authorized to access this document.",
        )

    async def test_delete_document_returns_structured_lifecycle_failure(self):
        failed_result = Mock(status="failed")
        failed_result.to_http_exception.return_value = HTTPException(
            status_code=502,
            detail={
                "message": "Failed to delete document from Supabase Storage",
                "lifecycle_status": "failed",
                "failure_stage": "storage",
                "reason_code": "storage_delete_failed",
                "cleanup_status": "partial",
            },
        )

        with (
            patch(
                "backend.routers.documents.require_user_document",
                return_value={"id": "doc-a", "user_id": "user-a", "storage_url": "pdfs/user-a/doc-a/file.pdf"},
            ),
            patch(
                "backend.routers.documents.delete_document_lifecycle",
                return_value=failed_result,
            ),
        ):
            with self.assertRaises(HTTPException) as exc:
                await delete_user_document(document_id="doc-a", user_id="user-a")

        self.assertEqual(exc.exception.status_code, 502)
        self.assertEqual(
            exc.exception.detail,
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
