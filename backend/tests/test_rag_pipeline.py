import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.services.rag_pipeline import (
    AnswerCitation,
    INSUFFICIENT_CONTEXT_ANSWER,
    _build_generation_prompt,
    _route_intent,
    _select_retrieval_policy,
    answer_question,
)


class RagPipelineTestCase(unittest.TestCase):
    def _build_vector_store(
        self,
        *,
        count: int = 4,
        docs=None,
        head_documents=None,
        head_metadatas=None,
        head_ids=None,
        query_documents=None,
        query_metadatas=None,
        query_ids=None,
    ):
        vectordb = Mock()
        vectordb._collection.count.return_value = count
        vectordb.similarity_search.return_value = docs or []
        vectordb.get.return_value = {
            "documents": head_documents or [],
            "metadatas": head_metadatas or [],
            "ids": head_ids or [],
        }
        query_documents = query_documents if query_documents is not None else [
            getattr(doc, "page_content", None) for doc in (docs or [])
        ]
        query_metadatas = query_metadatas if query_metadatas is not None else [
            getattr(doc, "metadata", None) for doc in (docs or [])
        ]
        query_ids = query_ids if query_ids is not None else [None for _ in query_documents]
        vectordb._collection.query.return_value = {
            "documents": [query_documents],
            "metadatas": [query_metadatas],
            "ids": [query_ids],
        }
        vectordb.embeddings = Mock()
        vectordb.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        return vectordb

    @staticmethod
    def _logged_event(logger_mock) -> dict:
        return json.loads(logger_mock.info.call_args.args[0])

    def test_answer_question_uses_llm_for_grounded_retrieval(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund window is 30 days."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm) as chat_openai_mock,
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, "The refund window is 30 days.")
        self.assertEqual(answer.intent, "qa")
        self.assertEqual(answer.retrieval_mode, "semantic")
        self.assertEqual(answer.answer_status, "answered")
        self.assertEqual(
            answer.citations,
            [
                AnswerCitation(
                    chunk_id="doc-1:chunk:0",
                    excerpt="The refund window is 30 days from the purchase date.",
                )
            ],
        )
        vectordb._collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=4,
            include=["documents", "metadatas"],
        )
        self.assertEqual(chat_openai_mock.call_count, 2)
        self.assertEqual(llm.invoke.call_count, 2)

    def test_answer_question_retries_when_model_breaks_json_contract(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="qa"),
            SimpleNamespace(content="The refund window is 30 days."),
            SimpleNamespace(content='{"answer": "The refund window is 30 days."}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, "The refund window is 30 days.")
        self.assertEqual(answer.answer_status, "answered")
        self.assertEqual(llm.invoke.call_count, 3)

    def test_answer_question_falls_back_after_repeated_json_contract_failures(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="qa"),
            SimpleNamespace(content="The refund window is 30 days."),
            SimpleNamespace(content='{"answer": ""}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])
        self.assertEqual(llm.invoke.call_count, 3)

    def test_answer_question_rejects_ungrounded_generated_answer(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund window is 45 days and includes free returns."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])

    def test_answer_question_accepts_grounded_paraphrase(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund period is 30 days."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, "The refund period is 30 days.")
        self.assertEqual(answer.answer_status, "answered")

    def test_answer_question_accepts_brief_grounded_numeric_answer(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "It lasts 30 days."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, "It lasts 30 days.")
        self.assertEqual(answer.answer_status, "answered")

    def test_qa_questions_route_to_semantic_retrieval_policy(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="qa")

        with patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm):
            policy = _select_retrieval_policy("What is the refund window?", total_chunks=12)

        self.assertEqual(policy.intent, "qa")
        self.assertEqual(policy.mode, "semantic")
        self.assertEqual(policy.limit, 4)
        self.assertTrue(policy.enforce_quality_gate)

    def test_summary_questions_route_to_head_retrieval_policy(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="summary")

        with patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm):
            policy = _select_retrieval_policy("What is this document about?", total_chunks=12)

        self.assertEqual(policy.intent, "summary")
        self.assertEqual(policy.mode, "head")
        self.assertEqual(policy.limit, 8)
        self.assertFalse(policy.enforce_quality_gate)

    def test_route_intent_treats_main_points_as_summary(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="summary")

        with patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm):
            self.assertEqual(_route_intent("What are the main points of this PDF?"), "summary")

    def test_generation_prompt_is_thin_and_harness_owned(self):
        prompt = _build_generation_prompt("What is the refund window?", "Refunds are allowed for 30 days.")

        self.assertIn(
            "You answer questions using only the provided document excerpts.",
            prompt,
        )
        self.assertIn("Do not invent facts", prompt)
        self.assertNotIn("I couldn't find enough information", prompt)
        self.assertNotIn("Avoid markdown formatting", prompt)
        self.assertNotIn("Use short paragraphs or simple bullets", prompt)

    def test_answer_question_returns_deterministic_fallback_for_low_evidence_retrieval(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="Payroll setup and laptop pickup.",
                    metadata={"chunk_id": "doc-1:chunk:3"},
                )
            ]
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.intent, "qa")
        self.assertEqual(answer.retrieval_mode, "semantic")
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])
        chat_openai_mock.assert_called_once()

    def test_answer_question_returns_deterministic_fallback_when_retrieval_is_empty(self):
        vectordb = self._build_vector_store(docs=[])

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.intent, "qa")
        self.assertEqual(answer.retrieval_mode, "semantic")
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])
        chat_openai_mock.assert_called_once()

    def test_answer_question_returns_fallback_when_semantic_retrieval_has_no_chunk_ids(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={},
                )
            ]
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])
        chat_openai_mock.assert_called_once()

    def test_answer_question_uses_legacy_query_ids_for_semantic_citations(self):
        vectordb = self._build_vector_store(
            docs=[],
            query_documents=["The refund window is 30 days from the purchase date."],
            query_metadatas=[{}],
            query_ids=["legacy-chunk-7"],
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund window is 30 days."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(
            answer.citations,
            [
                AnswerCitation(
                    chunk_id="legacy-chunk-7",
                    excerpt="The refund window is 30 days from the purchase date.",
                )
            ],
        )

    def test_summary_questions_still_use_head_context(self):
        vectordb = self._build_vector_store(
            docs=[],
            head_documents=["This handbook explains the benefits policy and time-off rules."],
            head_metadatas=[{"chunk_id": "doc-1:chunk:0"}],
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="summary"),
            SimpleNamespace(content='{"answer": "It explains benefits and time off."}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "Summarize this document.")

        self.assertEqual(answer.answer, "It explains benefits and time off.")
        self.assertEqual(answer.intent, "summary")
        self.assertEqual(answer.retrieval_mode, "head")
        self.assertEqual(answer.answer_status, "answered")
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].chunk_id, "doc-1:chunk:0")
        vectordb.get.assert_called_once_with(limit=4, include=["documents", "metadatas"])

    def test_summary_questions_return_fallback_when_head_context_has_no_chunk_ids(self):
        vectordb = self._build_vector_store(
            docs=[],
            head_documents=["This handbook explains the benefits policy and time-off rules."],
            head_metadatas=[{}],
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="summary")

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm) as chat_openai_mock,
        ):
            answer = answer_question("doc-1", "Summarize this document.")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.intent, "summary")
        self.assertEqual(answer.retrieval_mode, "head")
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])
        chat_openai_mock.assert_called_once()

    def test_summary_questions_use_legacy_get_ids_for_citations(self):
        vectordb = self._build_vector_store(
            docs=[],
            head_documents=["This handbook explains the benefits policy and time-off rules."],
            head_metadatas=[{}],
            head_ids=["legacy-head-2"],
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="summary"),
            SimpleNamespace(content='{"answer": "It explains benefits and time off."}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "Summarize this document.")

        self.assertEqual(answer.answer_status, "answered")
        self.assertEqual(answer.citations[0].chunk_id, "legacy-head-2")

    def test_summary_answer_must_be_grounded_in_head_context(self):
        vectordb = self._build_vector_store(
            docs=[],
            head_documents=["This handbook explains the benefits policy and time-off rules."],
            head_metadatas=[{"chunk_id": "doc-1:chunk:0"}],
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="summary"),
            SimpleNamespace(content='{"answer": "It explains benefits, time-off rules, and stock option grants."}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "Summarize this document.")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])

    def test_summary_paraphrase_can_pass_grounding_validation(self):
        vectordb = self._build_vector_store(
            docs=[],
            head_documents=["This handbook explains the benefits policy and time-off rules."],
            head_metadatas=[{"chunk_id": "doc-1:chunk:0"}],
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="summary"),
            SimpleNamespace(content='{"answer": "The handbook covers benefits policy and time-off rules."}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "Summarize this document.")

        self.assertEqual(answer.answer, "The handbook covers benefits policy and time-off rules.")
        self.assertEqual(answer.answer_status, "answered")

    def test_answer_question_normalizes_markdown_formatting_in_model_output(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content=(
                '{"answer": "**Refund window:** 30 days\\n\\n```text\\n'
                'From the purchase date.\\n```"}'
            )
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(
            answer.answer,
            "Refund window: 30 days\n\nFrom the purchase date.",
        )
        self.assertEqual(answer.answer_status, "answered")

    def test_answer_question_preserves_code_formatting_when_requested(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="Use the command pip install app to install the package.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "```bash\\npip install app\\n```"}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
        ):
            answer = answer_question("doc-1", "What install command?")

        self.assertEqual(answer.answer, "```bash\npip install app\n```")
        self.assertEqual(answer.answer_status, "answered")

    def test_qa_questions_do_not_fall_back_to_head_context(self):
        vectordb = self._build_vector_store(
            docs=[],
            head_documents=["The refund window is 30 days from the purchase date."],
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
        ):
            answer = answer_question("doc-1", "What is the refund window?")

        self.assertEqual(answer.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(answer.intent, "qa")
        self.assertEqual(answer.retrieval_mode, "semantic")
        self.assertEqual(answer.answer_status, "insufficient_context")
        self.assertEqual(answer.citations, [])
        vectordb._collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=4,
            include=["documents", "metadatas"],
        )
        vectordb.get.assert_not_called()
        chat_openai_mock.assert_called_once()

    def test_answer_question_logs_retrieval_quality_metrics_for_answered_decision(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund window is 30 days."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertEqual(event["event"], "answer_policy_decision")
        self.assertEqual(event["document_id"], "doc-1")
        self.assertEqual(event["answer_status"], "answered")
        self.assertIsNone(event["fallback_reason_code"])
        self.assertEqual(event["retrieval_mode"], "semantic")
        self.assertEqual(event["total_chunk_count"], 4)
        self.assertTrue(event["chunk_count_available"])
        self.assertEqual(event["retrieved_document_count"], 1)
        self.assertEqual(event["citation_count"], 1)
        self.assertEqual(event["missing_citation_count"], 0)
        self.assertEqual(event["citation_completeness_ratio"], 1.0)
        self.assertEqual(event["question_term_count"], 2)
        self.assertEqual(event["overlap_term_count"], 2)
        self.assertEqual(event["required_term_overlap"], 2)
        self.assertTrue(event["has_sufficient_context"])
        self.assertEqual(event["structured_output_retry_count"], 0)
        self.assertTrue(event["answer_grounded"])

    def test_answer_question_logs_explicit_reason_when_quality_gate_fails(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="Payroll setup and laptop pickup.",
                    metadata={"chunk_id": "doc-1:chunk:3"},
                )
            ]
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertEqual(event["answer_status"], "insufficient_context")
        self.assertEqual(event["fallback_reason_code"], "retrieval_quality_gate_failed")
        self.assertTrue(event["quality_gate_applied"])
        self.assertFalse(event["has_sufficient_context"])
        self.assertEqual(event["overlap_term_count"], 0)
        self.assertEqual(event["structured_output_retry_count"], 0)
        self.assertIsNone(event["answer_grounded"])
        chat_openai_mock.assert_called_once()

    def test_answer_question_logs_citation_completeness_when_chunk_ids_are_missing(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={},
                )
            ]
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertEqual(event["answer_status"], "insufficient_context")
        self.assertEqual(event["fallback_reason_code"], "empty_context")
        self.assertEqual(event["retrieved_document_count"], 1)
        self.assertEqual(event["citation_count"], 0)
        self.assertEqual(event["missing_citation_count"], 1)
        self.assertEqual(event["citation_completeness_ratio"], 0.0)
        self.assertEqual(event["retrieved_context_char_count"], 0)
        chat_openai_mock.assert_called_once()

    def test_answer_question_logs_structured_output_failure_reason(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="qa"),
            SimpleNamespace(content="The refund window is 30 days."),
            SimpleNamespace(content='{"answer": ""}'),
        ]

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertEqual(event["answer_status"], "insufficient_context")
        self.assertEqual(event["fallback_reason_code"], "structured_output_invalid")
        self.assertEqual(event["structured_output_retry_count"], 2)
        self.assertIsNone(event["answer_grounded"])

    def test_answer_question_logs_ungrounded_answer_failure_reason(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund window is 45 days and includes free returns."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertEqual(event["answer_status"], "insufficient_context")
        self.assertEqual(event["fallback_reason_code"], "answer_not_grounded")
        self.assertFalse(event["answer_grounded"])
        self.assertEqual(event["structured_output_retry_count"], 0)

    def test_answer_question_logs_unknown_chunk_count_when_count_lookup_fails(self):
        vectordb = self._build_vector_store(
            docs=[
                SimpleNamespace(
                    page_content="The refund window is 30 days from the purchase date.",
                    metadata={"chunk_id": "doc-1:chunk:0"},
                )
            ]
        )
        vectordb._collection.count.side_effect = RuntimeError("count unavailable")
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='{"answer": "The refund window is 30 days."}'
        )

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI", return_value=llm),
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertIsNone(event["total_chunk_count"])
        self.assertFalse(event["chunk_count_available"])

    def test_answer_question_logs_null_citation_completeness_when_nothing_is_retrieved(self):
        vectordb = self._build_vector_store(docs=[])

        with (
            patch("backend.services.rag_pipeline.get_vector_store", return_value=vectordb),
            patch("backend.services.rag_pipeline.ChatOpenAI") as chat_openai_mock,
            patch("backend.services.rag_pipeline.logger") as logger_mock,
        ):
            answer_question("doc-1", "What is the refund window?")

        event = self._logged_event(logger_mock)
        self.assertEqual(event["answer_status"], "insufficient_context")
        self.assertEqual(event["fallback_reason_code"], "empty_context")
        self.assertEqual(event["retrieved_document_count"], 0)
        self.assertEqual(event["citation_count"], 0)
        self.assertEqual(event["missing_citation_count"], 0)
        self.assertIsNone(event["citation_completeness_ratio"])
        chat_openai_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
