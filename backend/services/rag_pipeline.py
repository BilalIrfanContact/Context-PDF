import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .vector_store import get_vector_store


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You answer questions using only the provided document excerpts. "
    "Do not invent facts that are not supported by the excerpts."
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "I couldn't find enough information in the document to answer that question."
)
_STRUCTURED_OUTPUT_RETRY_LIMIT = 2
_STRUCTURED_OUTPUT_INSTRUCTION = (
    'Return only valid JSON with this exact shape: {"answer": string}. '
    "Do not include markdown, code fences, or any extra keys."
)

_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_GROUNDING_STOPWORDS = _QUESTION_STOPWORDS | {
    "about",
    "all",
    "also",
    "answer",
    "based",
    "can",
    "document",
    "enough",
    "enrollment",
    "exception",
    "final",
    "found",
    "helpful",
    "here",
    "information",
    "into",
    "its",
    "more",
    "not",
    "only",
    "should",
    "than",
    "their",
    "them",
    "there",
    "these",
    "they",
    "through",
    "under",
    "using",
    "user",
    "your",
}
Intent = Literal["summary", "qa"]
RouteIntent = Literal["summary", "qa", "off_topic"]
RetrievalMode = Literal["head", "semantic"]
FallbackReasonCode = Literal[
    "retrieval_quality_gate_failed",
    "empty_context",
    "structured_output_invalid",
    "answer_not_grounded",
]


@dataclass(frozen=True)
class RetrievalPolicy:
    intent: Intent
    mode: RetrievalMode
    limit: int
    enforce_quality_gate: bool


@dataclass(frozen=True)
class AnswerDecision:
    answer: str
    intent: Intent
    retrieval_mode: RetrievalMode
    answer_status: Literal["answered", "insufficient_context"]
    citations: list["AnswerCitation"]


@dataclass(frozen=True)
class AnswerCitation:
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    citations: list[AnswerCitation]
    retrieved_document_count: int


@dataclass(frozen=True)
class StructuredAnswerResult:
    answer: str | None
    invalid_attempt_count: int


class LlmAnswerPayload(BaseModel):
    answer: str


def _build_generation_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{_STRUCTURED_OUTPUT_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "JSON Response:"
    )


def _build_retry_prompt(question: str, context: str, invalid_response: str, error: str) -> str:
    return (
        f"{_build_generation_prompt(question, context)}\n\n"
        "Your previous response did not match the required JSON contract.\n"
        f"Validation error: {error}\n"
        f"Previous response:\n{invalid_response}\n\n"
        'Reply again with only valid JSON matching exactly {"answer": string}.'
    )


def _coerce_response_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()

    return str(content).strip()


def _parse_structured_answer(response_text: str, question: str) -> LlmAnswerPayload:
    payload = LlmAnswerPayload.model_validate_json(response_text)
    normalized_answer = _normalize_answer_text(payload.answer, question)
    if not normalized_answer:
        raise ValueError("answer must not be empty")
    return LlmAnswerPayload(answer=normalized_answer)


def _question_requests_literal_formatting(question: str) -> bool:
    q = question.lower()
    formatting_triggers = (
        "code",
        "command",
        "commands",
        "snippet",
        "script",
        "json",
        "yaml",
        "sql",
        "regex",
        "markdown",
        "example output",
    )
    return any(trigger in q for trigger in formatting_triggers)


def _answer_looks_like_code(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return False

    stripped = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", stripped)
    stripped = stripped.replace("```", "")

    code_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "SELECT ",
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "npm ",
        "pip ",
        "python ",
        "curl ",
        "{",
        "[",
    )
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return any(
        any(line.startswith(marker) for marker in code_markers)
        for line in lines
    )


def _normalize_answer_text(answer: str, question: str) -> str:
    text = answer.replace("\r\n", "\n").strip()
    if not text:
        return ""

    if _question_requests_literal_formatting(question) or _answer_looks_like_code(text):
        return text

    text = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(^|[\s(])(\*|_)([^*_]+?)\2(?=[\s).,!?]|$)", r"\1\3", text)

    normalized_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            normalized_lines.append("")
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"^[-*+]\s+", "- ", line)
        normalized_lines.append(line)

    normalized_text = "\n".join(normalized_lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
    return normalized_text.strip()


def _generate_structured_answer(llm, question: str, context: str) -> StructuredAnswerResult:
    prompt = _build_generation_prompt(question, context)
    invalid_attempt_count = 0

    for attempt in range(_STRUCTURED_OUTPUT_RETRY_LIMIT):
        response = llm.invoke(prompt)
        response_text = _coerce_response_text(response)

        try:
            return StructuredAnswerResult(
                answer=_parse_structured_answer(response_text, question).answer,
                invalid_attempt_count=invalid_attempt_count,
            )
        except (ValidationError, ValueError) as exc:
            invalid_attempt_count += 1
            if attempt == _STRUCTURED_OUTPUT_RETRY_LIMIT - 1:
                return StructuredAnswerResult(
                    answer=None,
                    invalid_attempt_count=invalid_attempt_count,
                )
            prompt = _build_retry_prompt(question, context, response_text, str(exc))

    return StructuredAnswerResult(answer=None, invalid_attempt_count=invalid_attempt_count)


def _format_texts(texts: Iterable[str]) -> str:
    return "\n\n".join(text for text in texts if text).strip()


def _extract_question_terms(text: str) -> set[str]:
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2 and term not in _QUESTION_STOPWORDS
    }
    return terms


def _has_sufficient_context(question: str, context: str) -> bool:
    if not context:
        return False
    return len(context.strip()) >= 50


def _retrieval_overlap_metrics(question: str, context: str) -> tuple[int, int, bool]:
    if not context:
        question_terms = _extract_question_terms(question)
        required_overlap = 0 if not question_terms else (
            1 if len(question_terms) == 1 else min(2, len(question_terms))
        )
        return 0, required_overlap, False

    question_terms = _extract_question_terms(question)
    if not question_terms:
        return 0, 0, True

    context_terms = set(re.findall(r"[a-z0-9]+", context.lower()))
    overlap_count = len(question_terms & context_terms)
    required_overlap = 1 if len(question_terms) == 1 else min(2, len(question_terms))
    return overlap_count, required_overlap, overlap_count >= required_overlap


def _extract_grounding_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2 and term not in _GROUNDING_STOPWORDS
    }


def _extract_numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.lower()))


def _split_answer_segments(answer: str) -> list[str]:
    normalized = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", answer)
    normalized = normalized.replace("```", "")
    normalized = re.sub(r"`([^`]+)`", r"\1", normalized)
    lines = [line.strip(" -") for line in normalized.splitlines()]
    segments = []
    for line in lines:
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            segment = part.strip()
            if segment:
                segments.append(segment)
    return segments or [answer.strip()]


def _is_segment_grounded(segment: str, evidence_terms: set[str], evidence_numbers: set[str]) -> bool:
    segment_terms = _extract_grounding_terms(segment)
    segment_numbers = _extract_numeric_tokens(segment)

    if segment_numbers and not segment_numbers.issubset(evidence_numbers):
        return False

    if not segment_terms:
        return bool(segment_numbers) or not segment.strip()

    overlap = segment_terms & evidence_terms
    unsupported_terms = segment_terms - evidence_terms
    if not overlap:
        return False

    if segment_numbers:
        return len(unsupported_terms) <= len(overlap)

    required_overlap = 1 if len(segment_terms) <= 3 else 2
    overlap_ratio = len(overlap) / len(segment_terms)
    return len(overlap) >= required_overlap and overlap_ratio >= 0.7


def _is_answer_grounded(answer: str, citations: Sequence[AnswerCitation]) -> bool:
    evidence_text = _format_texts(citation.excerpt for citation in citations)
    if not evidence_text:
        return False

    evidence_terms = _extract_grounding_terms(evidence_text)
    evidence_numbers = _extract_numeric_tokens(evidence_text)
    segments = _split_answer_segments(answer)
    if not segments:
        return False

    return all(
        _is_segment_grounded(segment, evidence_terms, evidence_numbers)
        for segment in segments
    )

def _route_intent(question: str) -> RouteIntent:
    prompt = (
        "Classify the user's question for a document Q&A app.\n"
        "Return exactly one label: summary, qa, or off_topic.\n"
        "summary = asking what the uploaded document is about.\n"
        "qa = asking for specific information from the uploaded document.\n"
        "off_topic = not about the uploaded document.\n\n"
        f"Question: {question}\n"
        "Label:"
    )

    try:
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-nano")
        llm = ChatOpenAI(model=model, temperature=0)
        response_text = _coerce_response_text(llm.invoke(prompt)).lower()
    except Exception:
        return "qa"

    match = re.search(r"\b(summary|qa|off_topic)\b", response_text)
    if match:
        return match.group(1)  # type: ignore[return-value]
    return "qa"


def _select_retrieval_policy(question: str, total_chunks: int) -> RetrievalPolicy:
    limit = min(8, max(1, total_chunks))
    routed_intent = _route_intent(question)
    if routed_intent == "summary":
        return RetrievalPolicy(
            intent="summary",
            mode="head",
            limit=limit,
            enforce_quality_gate=False,
        )

    return RetrievalPolicy(
        intent="qa",
        mode="semantic",
        limit=min(4, max(1, total_chunks)),
        enforce_quality_gate=True,
    )


def _head_context(vectordb, limit: int = 6) -> RetrievedContext:
    result = vectordb.get(limit=limit, include=["documents", "metadatas"])
    documents: Sequence[str] = result.get("documents") or []
    metadatas: Sequence[dict | None] = result.get("metadatas") or []
    ids: Sequence[str | None] = result.get("ids") or []
    citations = []
    cited_documents = []
    for index, document in enumerate(documents):
        if not document:
            continue
        citation = _citation_from_metadata(metadatas, ids, index, document)
        if citation is None:
            continue
        citations.append(citation)
        cited_documents.append(document)
    return RetrievedContext(
        text=_format_texts(cited_documents),
        citations=citations,
        retrieved_document_count=len([document for document in documents if document]),
    )


def _citation_from_metadata(
    metadatas: Sequence[dict | None],
    ids: Sequence[str | None],
    index: int,
    document_text: str,
) -> AnswerCitation | None:
    metadata = metadatas[index] if index < len(metadatas) else None
    chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) else None
    if not chunk_id and index < len(ids):
        chunk_id = ids[index]
    if not chunk_id:
        return None
    return AnswerCitation(chunk_id=chunk_id, excerpt=document_text)


def _cited_context_from_query_result(result: dict) -> RetrievedContext:
    documents_groups: Sequence[Sequence[str | None]] = result.get("documents") or []
    metadatas_groups: Sequence[Sequence[dict | None]] = result.get("metadatas") or []
    ids_groups: Sequence[Sequence[str | None]] = result.get("ids") or []

    documents = documents_groups[0] if documents_groups else []
    metadatas = metadatas_groups[0] if metadatas_groups else []
    ids = ids_groups[0] if ids_groups else []

    citations = []
    cited_texts = []
    for index, document in enumerate(documents):
        if not document:
            continue
        citation = _citation_from_metadata(metadatas, ids, index, document)
        if citation is None:
            continue
        citations.append(citation)
        cited_texts.append(document)
    return RetrievedContext(
        text=_format_texts(cited_texts),
        citations=citations,
        retrieved_document_count=len([document for document in documents if document]),
    )


def _semantic_context(vectordb, question: str, limit: int) -> RetrievedContext:
    embedding_function = getattr(vectordb, "embeddings", None)
    if embedding_function is None:
        raise ValueError("Vector store is missing an embedding function.")
    query_embedding = embedding_function.embed_query(question)
    result = vectordb._collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas"],
    )
    return _cited_context_from_query_result(result)


def _citation_from_doc(doc) -> AnswerCitation | None:
    metadata = getattr(doc, "metadata", None)
    chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) else None
    page_content = getattr(doc, "page_content", "")
    if not chunk_id or not page_content:
        return None
    return AnswerCitation(chunk_id=chunk_id, excerpt=page_content)


def _retrieve_context(vectordb, question: str, policy: RetrievalPolicy) -> RetrievedContext:
    if policy.mode == "head":
        return _head_context(vectordb, limit=policy.limit)
    return _semantic_context(vectordb, question, policy.limit)


def _citation_completeness_ratio(context: RetrievedContext) -> float | None:
    if context.retrieved_document_count == 0:
        return None
    return len(context.citations) / context.retrieved_document_count


def _emit_answer_policy_telemetry(
    *,
    document_id: str,
    question: str,
    total_chunks: int | None,
    policy: RetrievalPolicy,
    context: RetrievedContext,
    answer_status: Literal["answered", "insufficient_context"],
    fallback_reason_code: FallbackReasonCode | None,
    structured_output_retry_count: int,
    answer_grounded: bool | None,
) -> None:
    overlap_term_count, required_term_overlap, has_sufficient_context = _retrieval_overlap_metrics(
        question,
        context.text,
    )
    question_term_count = len(_extract_question_terms(question))
    citation_count = len(context.citations)
    citation_completeness_ratio = _citation_completeness_ratio(context)
    event = {
        "event": "answer_policy_decision",
        "document_id": document_id,
        "intent": policy.intent,
        "retrieval_mode": policy.mode,
        "answer_status": answer_status,
        "fallback_reason_code": fallback_reason_code,
        "quality_gate_applied": policy.enforce_quality_gate,
        "total_chunk_count": total_chunks,
        "chunk_count_available": total_chunks is not None,
        "retrieved_document_count": context.retrieved_document_count,
        "retrieved_context_char_count": len(context.text),
        "question_term_count": question_term_count,
        "overlap_term_count": overlap_term_count,
        "required_term_overlap": required_term_overlap,
        "has_sufficient_context": has_sufficient_context,
        "citation_count": citation_count,
        "missing_citation_count": context.retrieved_document_count - citation_count,
        "citation_completeness_ratio": citation_completeness_ratio,
        "structured_output_retry_count": structured_output_retry_count,
        "answer_grounded": answer_grounded,
    }
    logger.info(
        json.dumps(event, sort_keys=True),
    )


def _insufficient_context_decision(policy: RetrievalPolicy) -> AnswerDecision:
    return AnswerDecision(
        answer=INSUFFICIENT_CONTEXT_ANSWER,
        intent=policy.intent,
        retrieval_mode=policy.mode,
        answer_status="insufficient_context",
        citations=[],
    )


def answer_question(document_id: str, question: str) -> AnswerDecision:
    vectordb = get_vector_store(document_id=document_id)
    total_chunks: int | None = None
    try:
        total = vectordb._collection.count()
        total_chunks = total
    except Exception:
        total = 4

    policy = _select_retrieval_policy(question, total)
    context = _retrieve_context(vectordb, question, policy)

    if not context.text:
        decision = _insufficient_context_decision(policy)
        _emit_answer_policy_telemetry(
            document_id=document_id,
            question=question,
            total_chunks=total_chunks,
            policy=policy,
            context=context,
            answer_status=decision.answer_status,
            fallback_reason_code="empty_context",
            structured_output_retry_count=0,
            answer_grounded=None,
        )
        return decision

    if policy.enforce_quality_gate and not _has_sufficient_context(question, context.text):
        decision = _insufficient_context_decision(policy)
        _emit_answer_policy_telemetry(
            document_id=document_id,
            question=question,
            total_chunks=total_chunks,
            policy=policy,
            context=context,
            answer_status=decision.answer_status,
            fallback_reason_code="retrieval_quality_gate_failed",
            structured_output_retry_count=0,
            answer_grounded=None,
        )
        return decision

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-nano")
    llm = ChatOpenAI(model=model, temperature=0)
    structured_answer = _generate_structured_answer(llm, question, context.text)
    if structured_answer.answer is None:
        decision = _insufficient_context_decision(policy)
        _emit_answer_policy_telemetry(
            document_id=document_id,
            question=question,
            total_chunks=total_chunks,
            policy=policy,
            context=context,
            answer_status=decision.answer_status,
            fallback_reason_code="structured_output_invalid",
            structured_output_retry_count=structured_answer.invalid_attempt_count,
            answer_grounded=None,
        )
        return decision
    answer_grounded = _is_answer_grounded(structured_answer.answer, context.citations)
    if not answer_grounded:
        decision = _insufficient_context_decision(policy)
        _emit_answer_policy_telemetry(
            document_id=document_id,
            question=question,
            total_chunks=total_chunks,
            policy=policy,
            context=context,
            answer_status=decision.answer_status,
            fallback_reason_code="answer_not_grounded",
            structured_output_retry_count=structured_answer.invalid_attempt_count,
            answer_grounded=answer_grounded,
        )
        return decision

    decision = AnswerDecision(
        answer=structured_answer.answer,
        intent=policy.intent,
        retrieval_mode=policy.mode,
        answer_status="answered",
        citations=context.citations,
    )
    _emit_answer_policy_telemetry(
        document_id=document_id,
        question=question,
        total_chunks=total_chunks,
        policy=policy,
        context=context,
        answer_status=decision.answer_status,
        fallback_reason_code=None,
        structured_output_retry_count=structured_answer.invalid_attempt_count,
        answer_grounded=answer_grounded,
    )
    return decision
