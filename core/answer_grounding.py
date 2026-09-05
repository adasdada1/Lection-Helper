import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.technical_terms import identifier_support_source


INSUFFICIENT_ANSWER = (
    "В предоставленных фрагментах лекции нет достаточной информации "
    "для надёжного ответа на этот вопрос."
)
GROUNDING_FAILED_ANSWER = (
    "Подходящий материал найден, но сформированный ответ не прошёл "
    "проверку по транскрипту. Попробуйте переформулировать вопрос."
)
PROVIDER_FAILED_ANSWER = (
    "Сервис генерации ответа временно недоступен. Попробуйте повторить вопрос."
)
NON_LECTURE_ANSWER = (
    "Я отвечаю на вопросы по загруженным лекциям. "
    "Задайте вопрос по материалу курса."
)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_STOP_WORDS = {
    "а", "без", "был", "была", "были", "быть", "в", "во", "для", "до",
    "его", "ее", "если", "же", "за", "и", "из", "или", "их", "к", "как",
    "ко", "на", "над", "не", "но", "о", "об", "от", "по", "под", "при",
    "с", "со", "так", "такой", "то", "у", "что", "это", "этот", "эта",
    "эти", "этого", "этой", "является",
}
_RUSSIAN_SUFFIXES = (
    "иями", "ями", "ами", "ого", "ему", "ыми", "ими", "ение", "ания",
    "овать", "ывает", "ивает", "ывать", "ивать", "ется", "ются", "ать", "ять",
    "ить", "еть", "ает", "яет", "ует", "ют", "ят", "ит", "ет",
    "ого", "ему", "ому", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий",
    "ой", "ей", "ую", "юю", "ам", "ям", "ах", "ях", "ов", "ев", "ом",
    "ем", "а", "я", "ы", "и", "у", "ю", "е", "о",
)
_EQUIVALENT_TERMS = (
    frozenset({"опис", "описан", "описыва", "представ"}),
    frozenset({"принадлеж", "относ"}),
    frozenset({"отдельн", "конкретн"}),
    frozenset({"созда", "создав"}),
    frozenset({"хран", "содерж"}),
)
_NEGATION_MARKERS = ("не", "нет", "нельзя", "невозможно", "никогда")
_STRONG_MARKERS = (
    "все", "всегда", "любой", "любые", "каждый", "обязательно", "никогда",
    "исключительно", "только",
)
_CAUSAL_MARKERS = (
    "потому что", "поэтому", "следовательно", "из-за", "приводит к",
    "благодаря этому",
)
_QUALIFIER_GROUPS = (
    ("может", "могут", "возможно"),
    ("обычно", "как правило"),
    ("часто", "иногда", "редко"),
    ("большинство", "у большинства", "некоторые"),
    ("например", "в частности", "в этом случае", "в данном случае"),
    ("конкретный", "конкретная", "конкретное", "данный", "данная"),
)


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    quote: str


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: list[EvidenceRef] = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[AnswerClaim] = Field(min_length=1)


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: Literal["entailed", "partial", "unsupported"]
    reason: str = ""
    unsupported_parts: list[str] = Field(default_factory=list)
    missing_qualifiers: list[str] = Field(default_factory=list)
    term_substitutions: list[str] = Field(default_factory=list)


class ClaimValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ClaimVerdict]


def answer_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "evidence"],
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["source_id", "quote"],
                                "properties": {
                                    "source_id": {"type": "string", "minLength": 1},
                                    "quote": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def claim_validation_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "claim_id", "verdict", "reason", "unsupported_parts",
                        "missing_qualifiers", "term_substitutions",
                    ],
                    "properties": {
                        "claim_id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["entailed", "partial", "unsupported"],
                        },
                        "reason": {"type": "string"},
                        "unsupported_parts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "missing_qualifiers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "term_substitutions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def _extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def _is_atomic(text: str) -> bool:
    if not text or "\n" in text:
        return False
    if text.startswith(("#", "- ", "* ")):
        return False
    parts = re.split(r"(?<=[!?])\s+|(?<=\.)\s+(?=[А-ЯA-Z])", text)
    return len([part for part in parts if part.strip()]) == 1


def _empty_result(
    answer: str = INSUFFICIENT_ANSWER,
    valid: bool = False,
    rejected_claims: list[dict] | None = None,
    failure_reason: str = "no_evidence",
) -> dict:
    return {
        "answer": answer,
        "source_ids": [],
        "quotes": {},
        "claims": [],
        "rejected_claims": rejected_claims or [],
        "valid": valid,
        "failure_reason": failure_reason,
    }


def _result_from_claims(
    claims: list[dict],
    rejected_claims: list[dict] | None = None,
) -> dict:
    if not claims:
        return _empty_result()

    source_ids = []
    quotes: dict[str, list[str]] = {}
    for claim in claims:
        for evidence in claim["evidence"]:
            source_id = evidence["source_id"]
            quote = evidence["quote"]
            if source_id not in source_ids:
                source_ids.append(source_id)
            source_quotes = quotes.setdefault(source_id, [])
            if quote not in source_quotes:
                source_quotes.append(quote)

    return {
        "answer": " ".join(claim["text"] for claim in claims),
        "source_ids": source_ids,
        "quotes": quotes,
        "claims": claims,
        "rejected_claims": rejected_claims or [],
        "valid": True,
        "failure_reason": None,
    }


def validate_grounded_answer(raw: str, evidence_sources: dict[str, dict]) -> dict:
    try:
        payload = json.loads(_extract_json(raw))
        answer = GeneratedAnswer.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return _empty_result(
            GROUNDING_FAILED_ANSWER,
            failure_reason="invalid_model_response",
        )

    accepted = []
    rejected = []

    for claim in answer.claims:
        text = claim.text.strip()
        if not _is_atomic(text):
            rejected.append({
                "claim": claim.model_dump(),
                "verdict": {"verdict": "structural", "reason": "non_atomic_claim"},
            })
            continue

        claim_evidence = []
        evidence_valid = True
        for evidence in claim.evidence:
            source = evidence_sources.get(evidence.source_id)
            quote = evidence.quote.strip()
            if not source:
                rejected.append({
                    "claim": claim.model_dump(),
                    "verdict": {"verdict": "structural", "reason": "unknown_source"},
                })
                evidence_valid = False
                break
            if not quote or quote not in source["text"]:
                rejected.append({
                    "claim": claim.model_dump(),
                    "verdict": {"verdict": "structural", "reason": "exact_quote_mismatch"},
                })
                evidence_valid = False
                break
            claim_evidence.append({
                "source_id": evidence.source_id,
                "quote": quote,
            })

        if not evidence_valid or not claim_evidence:
            continue

        accepted.append({
            "claim_id": f"C{len(accepted) + 1}",
            "text": text,
            "evidence": claim_evidence,
        })

    if not accepted:
        return _empty_result(
            GROUNDING_FAILED_ANSWER,
            rejected_claims=rejected,
            failure_reason="claims_failed_structural_validation",
        )

    return _result_from_claims(accepted, rejected_claims=rejected)


def apply_semantic_validation(grounded: dict, raw: str) -> dict:
    claims = grounded.get("claims", [])
    if not grounded.get("valid") or not claims:
        return grounded

    try:
        payload = json.loads(_extract_json(raw))
        validation = ClaimValidation.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return _empty_result(
            GROUNDING_FAILED_ANSWER,
            failure_reason="invalid_validation_response",
        )

    expected_ids = {claim["claim_id"] for claim in claims}
    verdicts = {}
    for result in validation.results:
        if result.claim_id not in expected_ids or result.claim_id in verdicts:
            return _empty_result(
                GROUNDING_FAILED_ANSWER,
                failure_reason="invalid_validation_claim_ids",
            )
        verdicts[result.claim_id] = result.verdict

    if set(verdicts) != expected_ids:
        return _empty_result(
            GROUNDING_FAILED_ANSWER,
            failure_reason="incomplete_validation_response",
        )

    accepted = []
    rejected = []
    results_by_id = {
        result.claim_id: result for result in validation.results
    }
    for claim in claims:
        result = results_by_id[claim["claim_id"]]
        if result.verdict == "entailed":
            accepted.append(claim)
        else:
            rejected.append({
                "claim": claim,
                "verdict": result.model_dump(),
            })

    if not accepted:
        return _empty_result(
            GROUNDING_FAILED_ANSWER,
            rejected_claims=rejected,
            failure_reason="semantic_validation_rejected_all",
        )

    grounded_result = _result_from_claims(accepted)
    grounded_result["rejected_claims"] = rejected
    return grounded_result


def _normalize_text(text: str) -> str:
    words = _WORD_RE.findall(text.casefold().replace("ё", "е"))
    return " ".join(words)


def _stem_word(word: str) -> str:
    normalized = word.casefold().replace("ё", "е")
    if len(normalized) < 4 or not re.fullmatch(r"[а-я]+", normalized):
        return normalized
    for suffix in _RUSSIAN_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 3:
            return normalized[:-len(suffix)]
    return normalized


def _content_terms(text: str) -> set[str]:
    return {
        _stem_word(word)
        for word in _WORD_RE.findall(text.casefold().replace("ё", "е"))
        if word not in _STOP_WORDS and len(word) > 2
    }


def _contains_marker(text: str, marker: str) -> bool:
    normalized = f" {_normalize_text(text)} "
    return f" {_normalize_text(marker)} " in normalized


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


def _term_is_covered(term: str, evidence_terms: set[str]) -> bool:
    if term in evidence_terms:
        return True
    for group in _EQUIVALENT_TERMS:
        if term in group and group.intersection(evidence_terms):
            return True
    return False


def _special_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(text)
        if "_" in token or re.search(r"[A-Za-z]", token)
    }


def _claim_risk(
    claim: dict,
    question: str = "",
) -> tuple[str, list[str]]:
    text = claim["text"]
    evidence_text = " ".join(
        evidence["quote"] for evidence in claim["evidence"]
    )
    normalized_text = _normalize_text(text)
    normalized_quotes = [
        _normalize_text(evidence["quote"])
        for evidence in claim["evidence"]
    ]
    if normalized_text in normalized_quotes:
        return "safe", []

    reasons = []
    claim_numbers = set(_NUMBER_RE.findall(text))
    evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
    if claim_numbers - evidence_numbers:
        reasons.append("new_number")

    claim_special = _special_tokens(text)
    evidence_special = _special_tokens(evidence_text)
    identifier_support = {
        identifier: identifier_support_source(
            identifier,
            evidence_text,
            question,
        )
        for identifier in claim_special - evidence_special
    }
    unsupported_identifiers = {
        identifier
        for identifier, source in identifier_support.items()
        if source is None
    }
    question_identifiers = {
        identifier
        for identifier, source in identifier_support.items()
        if source == "question"
    }
    if unsupported_identifiers:
        reasons.append("new_identifier")

    if reasons:
        return "blocked", reasons

    claim_negation = _contains_any(text, _NEGATION_MARKERS)
    evidence_negation = _contains_any(evidence_text, _NEGATION_MARKERS)
    if claim_negation != evidence_negation:
        reasons.append("negation_changed")

    if question_identifiers:
        reasons.append("identifier_from_question")

    for marker in _STRONG_MARKERS:
        if _contains_marker(text, marker) and not _contains_marker(evidence_text, marker):
            reasons.append("stronger_scope")
            break

    for marker in _CAUSAL_MARKERS:
        if _contains_marker(text, marker) and not _contains_marker(evidence_text, marker):
            reasons.append("new_causal_relation")
            break

    for group in _QUALIFIER_GROUPS:
        if _contains_any(evidence_text, group) and not _contains_any(text, group):
            reasons.append("qualifier_dropped")
            break

    if _contains_marker(text, "например") and not _contains_marker(
        evidence_text, "например"
    ):
        reasons.append("example_generalization")

    if len(claim["evidence"]) > 1:
        reasons.append("multiple_evidence_fragments")

    claim_terms = _content_terms(text)
    evidence_terms = _content_terms(evidence_text)
    covered_terms = {
        term for term in claim_terms
        if _term_is_covered(term, evidence_terms)
    }
    novel_terms = claim_terms - covered_terms
    if claim_terms and len(covered_terms) / len(claim_terms) < 0.5:
        reasons.append("low_lexical_overlap")
    if len(novel_terms) >= 2 and reasons:
        reasons.append("several_new_terms")

    return ("review", reasons) if reasons else ("safe", [])


def build_semantic_review_plan(
    grounded: dict,
    question: str = "",
) -> dict:
    if not grounded.get("valid") or not grounded.get("claims"):
        return {
            "safe_claim_ids": [],
            "review": _empty_result(valid=True),
            "blocked_claims": [],
            "risk_reasons": {},
        }

    safe_claim_ids = []
    review_claims = []
    blocked_claims = []
    risk_reasons = {}
    for claim in grounded["claims"]:
        level, reasons = _claim_risk(claim, question=question)
        if reasons:
            risk_reasons[claim["claim_id"]] = reasons
        if level == "safe":
            safe_claim_ids.append(claim["claim_id"])
        elif level == "review":
            review_claims.append(claim)
        else:
            blocked_claims.append(claim)

    review = (
        _result_from_claims(review_claims)
        if review_claims
        else _empty_result(valid=True)
    )
    return {
        "safe_claim_ids": safe_claim_ids,
        "review": review,
        "blocked_claims": blocked_claims,
        "risk_reasons": risk_reasons,
    }


def merge_semantic_review(
    grounded: dict,
    plan: dict,
    reviewed: dict,
) -> dict:
    accepted_ids = set(plan["safe_claim_ids"])
    if reviewed.get("valid"):
        accepted_ids.update(
            claim["claim_id"] for claim in reviewed.get("claims", [])
        )
    accepted = [
        claim for claim in grounded.get("claims", [])
        if claim["claim_id"] in accepted_ids
    ]
    rejected = list(grounded.get("rejected_claims", []))
    rejected.extend([
        {
            "claim": claim,
            "verdict": {
                "verdict": "blocked_by_rules",
                "reason": ",".join(plan["risk_reasons"].get(claim["claim_id"], [])),
            },
        }
        for claim in plan["blocked_claims"]
    ])
    rejected.extend(reviewed.get("rejected_claims", []))
    reviewed_ids = {
        item.get("claim", {}).get("claim_id")
        for item in reviewed.get("rejected_claims", [])
    }
    if not reviewed.get("valid"):
        for claim in plan["review"].get("claims", []):
            if claim["claim_id"] not in reviewed_ids:
                rejected.append({
                    "claim": claim,
                    "verdict": {
                        "verdict": "validator_failed",
                        "reason": reviewed.get("failure_reason") or "validator_unavailable",
                    },
                })
    if not accepted:
        return _empty_result(
            GROUNDING_FAILED_ANSWER,
            rejected_claims=rejected,
            failure_reason="all_claims_rejected",
        )
    result = _result_from_claims(accepted)
    result["rejected_claims"] = rejected
    return result
