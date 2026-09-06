import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.technical_terms import identifier_support_source


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_NEGATION_MARKERS = ("не", "нет", "нельзя", "невозможно", "никогда")
_STRONG_MARKERS = (
    "все", "всегда", "любой", "любые", "каждый", "обязательно", "никогда",
    "исключительно", "только",
)
_CAUSAL_MARKERS = (
    "потому что", "поэтому", "следовательно", "из-за", "приводит к",
    "благодаря этому",
)
_STOP_WORDS = {
    "а", "без", "был", "была", "были", "быть", "в", "во", "для", "до",
    "его", "ее", "если", "же", "за", "и", "из", "или", "их", "к", "как",
    "ко", "на", "над", "не", "но", "о", "об", "от", "по", "под", "при",
    "с", "со", "так", "такой", "то", "у", "что", "это", "этот", "эта",
    "эти", "этого", "этой", "является",
}


class SemanticSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    basis: Literal["source", "synthesis", "inference"]
    covers: list[str] = Field(min_length=1)


class CoverageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    status: Literal["covered", "not_in_sources"]
    section_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class SemanticAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[SemanticSection] = Field(min_length=1)
    coverage: list[CoverageItem] = Field(min_length=1)


class SectionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    verdict: Literal[
        "supported", "reasonable_inference", "unsupported", "contradiction"
    ]
    reason: str = ""


class SemanticValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[SectionVerdict]


def build_question_requirements(question: str) -> list[dict]:
    normalized = " ".join(question.casefold().replace("ё", "е").split())
    descriptions = []
    asks_why = bool(re.search(r"\bпочему\b|\bпо какой причине\b", normalized))
    asks_mechanism = bool(re.search(
        r"\bкак\b[^.!?;]*\b(?:работ\w*|устро\w*|выполня\w*|происход\w*|запрещ\w*)\b",
        normalized,
    ))

    if (
        re.search(r"\bперечисл\w*\b", normalized)
        or re.search(r"\bназов\w*\b.*\bвсе\b", normalized)
        or re.search(r"\bвсе\s+(?:четыре|три|два|основн\w*|принцип\w*)\b", normalized)
        or re.search(r"\bкакие\b.*\b(?:правил\w*|этап\w*|принцип\w*)\b", normalized)
    ):
        descriptions.append("Назвать все запрошенные элементы без пропусков.")

    if (
        "что это" in normalized
        or "что такое" in normalized
        or "за что" in normalized
        or re.search(r"\bсущност\w*\b", normalized)
        or re.search(r"\bознача\w*\b", normalized)
        or (
            re.search(r"\b(?:объясн\w*|объясня\w*|поясн\w*)\b", normalized)
            and not (asks_why or asks_mechanism)
        )
    ):
        descriptions.append(
            "Объяснить сущность или назначение каждого запрошенного понятия. "
            "Если требуется объяснить каждый элемент перечня, одних названий недостаточно."
        )

    if asks_why:
        descriptions.append(
            "Объяснить причину: показать связь между условием и результатом, "
            "а не только назвать термин. Сохранить существенные оговорки."
        )
    if asks_mechanism:
        descriptions.append(
            "Объяснить механизм или последовательность действий из вопроса "
            "и результат этих действий."
        )

    if re.search(r"\bкогда\b", normalized) or re.search(
        r"\bпримен\w*\b", normalized
    ):
        descriptions.append(
            "Объяснить, когда применяется каждый запрошенный вариант отдельно. "
            "В сравнении раскрыть применение обеих сторон, а не только одной."
        )

    if re.search(r"\bзачем\b", normalized) or re.search(r"\bдля чего\b", normalized):
        descriptions.append("Объяснить практическую цель или пользу.")

    if (
        re.search(r"\bотлич\w*\b", normalized)
        or re.search(r"\bразниц\w*\b", normalized)
        or re.search(r"\bсравн\w*\b", normalized)
    ):
        descriptions.append("Прямо показать различия между запрошенными понятиями.")

    if re.search(
        r"\b(?:привед\w*|покаж\w*|дай|дайте|добав\w*)\b[^.!?;]*\bпример\w*\b"
        r"|\bна примере\b|\bс\s+(?:(?:коротк\w*|прост\w*|одним)\s+)?пример\w*\b",
        normalized,
    ):
        descriptions.append("Привести пример из предоставленного материала.")

    if not descriptions:
        descriptions.append("Дать прямой и достаточный ответ на вопрос студента.")

    return [
        {"requirement_id": f"R{index}", "description": description}
        for index, description in enumerate(descriptions, start=1)
    ]


def select_answer_reasoning(requirements: list[dict]) -> str:
    return "medium" if len(requirements) > 1 else "low"


def response_language_matches(question: str, sections: list[dict]) -> bool:
    """Не выпускать английский ответ на явно русский вопрос."""
    if not re.search(r"[А-Яа-яЁё]", question):
        return True

    text = " ".join(str(section.get("text", "")) for section in sections)
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return False
    cyrillic_count = sum(bool(re.fullmatch(r"[А-Яа-яЁё]", char)) for char in letters)
    return cyrillic_count / len(letters) >= 0.3


def build_response_style(question: str, requirements: list[dict]) -> dict:
    """Задать глубину объяснения без нормы, которую нужно добирать словами."""
    normalized = " ".join(question.casefold().replace("ё", "е").split())
    brief = re.search(
        r"\b(?:кратк\w*|коротк\w*|сжато|лаконичн\w*|покороче)\b",
        normalized,
    )
    detailed = re.search(
        r"\b(?:подробн\w*|развернут\w*|детальн\w*|максимально полно)\b",
        normalized,
    )

    style = {
        "mode": "brief" if brief else "detailed" if detailed else "standard",
        "stop_when": (
            "Дан прямой ответ, раскрыты запрошенные части, необходимое объяснение "
            "и существенные оговорки. Не добирай слова и не повторяй итог."
        ),
        "coverage": (
            "Сохрани все требования и раскрой запрошенные аспекты для каждого объекта."
            if len(requirements) > 1 else
            "Объём определяет смысл вопроса, а не количество найденных фрагментов."
        ),
    }
    if brief:
        style.update({
            "examples": "Пример нужен, если он запрошен или без него непонятен ответ.",
            "structure": (
                "Сократи формулировки, но не причины, шаги и запрошенные элементы. "
                "Без вступления и повторного итога."
            ),
        })
    elif detailed:
        style.update({
            "examples": "Разбери полезные примеры из материала, не повторяющие друг друга.",
            "structure": (
                "Последовательно раскрой причины, шаги и условия по теме вопроса. "
                "Каждый абзац должен добавлять объяснение, а не пересказывать предыдущий."
            ),
        })
    else:
        style.update({
            "examples": (
                "Для абстрактного понятия используй короткий поясняющий пример, "
                "если он нужен; не добавляй пример к простому названию факта. "
                "Если запрошены несколько примеров, приведи их."
            ),
            "structure": (
                "Прямой ответ, затем необходимое пояснение. Для факта достаточно "
                "названия с полезным уточнением; для почему/как нужно объяснение, "
                "для сравнения или перечня - все запрошенные стороны."
            ),
        })
    return style


def semantic_answer_json_schema(requirements: list[dict]) -> dict:
    requirement_ids = [item["requirement_id"] for item in requirements]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["sections", "coverage"],
        "properties": {
            "sections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "section_id", "text", "source_ids", "basis", "covers",
                    ],
                    "properties": {
                        "section_id": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "minLength": 1},
                        "source_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "pattern": "^T[1-9][0-9]*$"},
                        },
                        "basis": {
                            "type": "string",
                            "enum": ["source", "synthesis", "inference"],
                        },
                        "covers": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": requirement_ids},
                        },
                    },
                },
            },
            "coverage": {
                "type": "array",
                "minItems": len(requirement_ids),
                "maxItems": len(requirement_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "requirement_id", "status", "section_ids", "reason",
                    ],
                    "properties": {
                        "requirement_id": {
                            "type": "string",
                            "enum": requirement_ids,
                        },
                        "status": {
                            "type": "string",
                            "enum": ["covered", "not_in_sources"],
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def semantic_validation_json_schema() -> dict:
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
                    "required": ["section_id", "verdict", "reason"],
                    "properties": {
                        "section_id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "supported", "reasonable_inference",
                                "unsupported", "contradiction",
                            ],
                        },
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def validate_semantic_answer(
    raw: str,
    sources: dict[str, dict],
    requirements: list[dict],
) -> dict:
    try:
        parsed = SemanticAnswer.model_validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return _empty_result("invalid_model_response")

    requirement_ids = {item["requirement_id"] for item in requirements}
    section_ids = [section.section_id for section in parsed.sections]
    if len(section_ids) != len(set(section_ids)):
        return _empty_result("duplicate_section_ids")

    coverage_ids = [item.requirement_id for item in parsed.coverage]
    if len(coverage_ids) != len(set(coverage_ids)) or set(coverage_ids) != requirement_ids:
        return _empty_result("invalid_coverage_requirements")

    known_sections = set(section_ids)
    known_sources = set(sources)
    for section in parsed.sections:
        if not set(section.source_ids).issubset(known_sources):
            return _empty_result("unknown_source")
        if not set(section.covers).issubset(requirement_ids):
            return _empty_result("unknown_requirement")

    for item in parsed.coverage:
        referenced = set(item.section_ids)
        if not referenced.issubset(known_sections):
            return _empty_result("unknown_coverage_section")
        if item.status == "covered" and not referenced:
            return _empty_result("empty_covered_requirement")
        if item.status == "not_in_sources" and referenced:
            return _empty_result("unsupported_coverage_reference")
        for section_id in referenced:
            section = next(
                section for section in parsed.sections
                if section.section_id == section_id
            )
            if item.requirement_id not in section.covers:
                return _empty_result("inconsistent_coverage_mapping")

    sections = [section.model_dump() for section in parsed.sections]
    coverage = [item.model_dump() for item in parsed.coverage]
    return _result_from_sections(sections, coverage, requirements)


def build_section_review_plan(
    grounded: dict,
    sources: dict[str, dict],
    question: str,
) -> dict:
    safe_section_ids = []
    review_sections = []
    blocked_sections = []
    risk_reasons = {}

    for section in grounded.get("sections", []):
        level, reasons = _section_risk(section, sources, question)
        if reasons:
            risk_reasons[section["section_id"]] = reasons
        if level == "safe":
            safe_section_ids.append(section["section_id"])
        elif level == "review":
            review_sections.append(_review_payload(section, sources))
        else:
            blocked_sections.append({
                "section": section,
                "reasons": reasons,
            })

    return {
        "safe_section_ids": safe_section_ids,
        "review_sections": review_sections,
        "blocked_sections": blocked_sections,
        "risk_reasons": risk_reasons,
    }


def apply_section_validation(review_sections: list[dict], raw: str) -> dict:
    expected_ids = {section["section_id"] for section in review_sections}
    try:
        parsed = SemanticValidation.model_validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return {
            "valid": False,
            "accepted_section_ids": [],
            "rejected_sections": [
                {"section": section, "verdict": "validator_failed"}
                for section in review_sections
            ],
            "failure_reason": "invalid_validation_response",
        }

    result_ids = [item.section_id for item in parsed.results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != expected_ids:
        return {
            "valid": False,
            "accepted_section_ids": [],
            "rejected_sections": [
                {"section": section, "verdict": "validator_failed"}
                for section in review_sections
            ],
            "failure_reason": "invalid_validation_section_ids",
        }

    accepted = []
    rejected = []
    review_by_id = {section["section_id"]: section for section in review_sections}
    for item in parsed.results:
        if item.verdict in {"supported", "reasonable_inference"}:
            accepted.append(item.section_id)
        else:
            rejected.append({
                "section": review_by_id[item.section_id],
                "verdict": item.model_dump(),
            })

    return {
        "valid": True,
        "accepted_section_ids": accepted,
        "rejected_sections": rejected,
        "failure_reason": None,
    }


def merge_section_review(
    grounded: dict,
    review_plan: dict,
    validation: dict,
    requirements: list[dict],
) -> dict:
    accepted_ids = set(review_plan["safe_section_ids"])
    accepted_ids.update(validation.get("accepted_section_ids", []))
    sections = [
        section for section in grounded.get("sections", [])
        if section["section_id"] in accepted_ids
    ]
    rejected = list(review_plan["blocked_sections"])
    rejected.extend(validation.get("rejected_sections", []))
    coverage = _coverage_for_sections(
        sections,
        grounded.get("coverage", []),
        requirements,
    )
    result = _result_from_sections(sections, coverage, requirements)
    result["rejected_sections"] = rejected
    if not validation.get("valid", True):
        result["validator_failure"] = validation.get("failure_reason")
        result["lost_requirement_ids"] = list(dict.fromkeys(
            requirement_id
            for section in review_plan["review_sections"]
            for requirement_id in next(
                original["covers"]
                for original in grounded.get("sections", [])
                if original["section_id"] == section["section_id"]
            )
        ))
    return result


def combine_semantic_results(
    base: dict,
    repair: dict,
    requirements: list[dict],
) -> dict:
    sections = []
    base_sections = base.get("sections", []) if base.get("valid") else []
    repair_sections = repair.get("sections", []) if repair.get("valid") else []
    for section in base_sections + repair_sections:
        item = dict(section)
        item["section_id"] = f"S{len(sections) + 1}"
        sections.append(item)
    coverage = _coverage_for_sections(sections, [], requirements)
    result = _result_from_sections(sections, coverage, requirements)
    result["rejected_sections"] = (
        base.get("rejected_sections", []) + repair.get("rejected_sections", [])
    )
    return result


def _section_risk(
    section: dict,
    sources: dict[str, dict],
    question: str,
) -> tuple[str, list[str]]:
    text = section["text"]
    source_text = " ".join(
        sources[source_id]["text"] for source_id in section["source_ids"]
    )
    normalized_context = " ".join(dict.fromkeys(
        sources[source_id].get("normalized_context", "")
        for source_id in section["source_ids"]
        if sources[source_id].get("normalized_context")
    ))
    identifier_context = f"{source_text} {normalized_context}".strip()
    source_anchor = _semantic_anchor(text, source_text)
    reasons = []

    text_numbers = set(_NUMBER_RE.findall(text))
    source_numbers = set(_NUMBER_RE.findall(source_text))
    if text_numbers - source_numbers:
        return "blocked", ["new_number"]

    text_identifiers = _special_tokens(text)
    source_identifiers = _special_tokens(source_text)
    unsupported_identifiers = {
        identifier for identifier in text_identifiers - source_identifiers
        if identifier_support_source(identifier, identifier_context, question) is None
    }
    if unsupported_identifiers:
        reasons.append("new_identifier")

    if _has_any(text, _NEGATION_MARKERS) != _has_any(source_anchor, _NEGATION_MARKERS):
        reasons.append("negation_changed")

    for marker in _STRONG_MARKERS:
        if _has_marker(text, marker) and not _has_marker(identifier_context, marker):
            reasons.append("stronger_scope")
            break

    for marker in _CAUSAL_MARKERS:
        if _has_marker(text, marker) and not _has_marker(source_anchor, marker):
            reasons.append("new_causal_relation")
            break

    if section["basis"] == "inference":
        reasons.append("declared_inference")

    text_terms = _content_terms(text)
    source_terms = _content_terms(source_text)
    overlap = len(text_terms & source_terms) / len(text_terms) if text_terms else 1.0
    if overlap < 0.2:
        reasons.append("low_semantic_anchor")

    return ("review", reasons) if reasons else ("safe", [])


def _review_payload(section: dict, sources: dict[str, dict]) -> dict:
    return {
        "section_id": section["section_id"],
        "text": section["text"],
        "basis": section["basis"],
        "sources": [
            {"source_id": source_id, "text": sources[source_id]["text"]}
            for source_id in section["source_ids"]
        ],
    }


def _coverage_for_sections(
    sections: list[dict],
    previous_coverage: list[dict],
    requirements: list[dict],
) -> list[dict]:
    previous = {
        item["requirement_id"]: item for item in previous_coverage
    }
    coverage = []
    for requirement in requirements:
        requirement_id = requirement["requirement_id"]
        section_ids = [
            section["section_id"] for section in sections
            if requirement_id in section.get("covers", [])
        ]
        if section_ids:
            coverage.append({
                "requirement_id": requirement_id,
                "status": "covered",
                "section_ids": section_ids,
                "reason": "",
            })
        else:
            coverage.append({
                "requirement_id": requirement_id,
                "status": "not_in_sources",
                "section_ids": [],
                "reason": previous.get(requirement_id, {}).get("reason", ""),
            })
    return coverage


def _result_from_sections(
    sections: list[dict],
    coverage: list[dict],
    requirements: list[dict],
) -> dict:
    source_ids = []
    for section in sections:
        for source_id in section.get("source_ids", []):
            if source_id not in source_ids:
                source_ids.append(source_id)
    uncovered = [
        item["requirement_id"] for item in coverage
        if item["status"] != "covered"
    ]
    return {
        "valid": bool(sections),
        "answer": "\n\n".join(section["text"].strip() for section in sections),
        "sections": sections,
        "coverage": coverage,
        "source_ids": source_ids,
        "uncovered_requirement_ids": uncovered,
        "requirements": requirements,
        "failure_reason": None if sections else "no_supported_sections",
        "rejected_sections": [],
    }


def _empty_result(reason: str) -> dict:
    return {
        "valid": False,
        "answer": "",
        "sections": [],
        "coverage": [],
        "source_ids": [],
        "uncovered_requirement_ids": [],
        "requirements": [],
        "failure_reason": reason,
        "rejected_sections": [],
    }


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold().replace("ё", "е")))


def _has_marker(text: str, marker: str) -> bool:
    return f" {_normalize(marker)} " in f" {_normalize(text)} "


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(_has_marker(text, marker) for marker in markers)


def _content_terms(text: str) -> set[str]:
    return {
        word for word in _WORD_RE.findall(text.casefold().replace("ё", "е"))
        if word not in _STOP_WORDS and len(word) > 2
    }


def _special_tokens(text: str) -> set[str]:
    return {
        token.casefold() for token in _WORD_RE.findall(text)
        if "_" in token or re.search(r"[A-Za-z]", token)
    }


def _semantic_anchor(text: str, source_text: str) -> str:
    source_units = [
        unit.strip() for unit in re.split(r"(?<=[.!?])\s+", source_text)
        if unit.strip()
    ]
    text_units = [
        unit.strip() for unit in re.split(r"(?<=[.!?])\s+", text)
        if unit.strip()
    ]
    if not source_units or not text_units:
        return source_text

    anchors = []
    for unit in text_units:
        terms = _content_terms(unit)
        anchor = max(
            source_units,
            key=lambda candidate: len(terms & _content_terms(candidate)),
        )
        if anchor not in anchors:
            anchors.append(anchor)
    return " ".join(anchors)
