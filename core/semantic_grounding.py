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
    replaces: list[str] = Field(default_factory=list)


class CoverageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    status: Literal["covered", "missing_in_answer", "not_in_sources", "blocked_by_validation"]
    section_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    missing_aspects: list[str] = Field(default_factory=list)


class SemanticAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[SemanticSection] = Field(min_length=1)
    coverage: list[CoverageItem] = Field(min_length=1)


class SemanticRepair(SemanticAnswer):
    sections: list[SemanticSection] = Field(min_length=0)


class SectionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    verdict: Literal[
        "supported", "reasonable_inference", "partial", "unsupported", "contradiction"
    ]
    reason: str = ""
    problematic_text: str = ""
    repair_instruction: str = ""
    supported_sentences: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class DuplicateSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    duplicate_of: str


class SemanticValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[SectionVerdict]
    coverage: list[CoverageItem] = Field(default_factory=list)
    duplicates: list[DuplicateSection] = Field(default_factory=list)


def _comparison_subjects(question: str) -> list[str]:
    """Выделить явную пару без словаря тем и отдельного LLM-запроса."""
    text = re.sub(r"\s+", " ", question.strip())
    tail = r"(?=\s+и\s+(?:когда|как|зачем|в каких)\b|[?;:.!]|$)"
    patterns = (
        r"\bчем\s+(.+?)\s+отличается\s+от\s+(.+?)" + tail,
        r"\bразница\s+между\s+(.+?)\s+и\s+(.+?)" + tail,
        r"\bсравни(?:те)?\s+(.+?)\s+и\s+(.+?)" + tail,
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            subjects = [part.strip(" ,") for part in match.groups()]
            if all(subjects) and all(len(part) <= 120 for part in subjects):
                return subjects
    return []


def requires_coverage_review(question: str, requirements: list[dict]) -> bool:
    return len(requirements) > 1 or bool(re.search(
        r"\b(?:перечисл\w*|кажд\w*|несколько|сравн\w*|отлич\w*|разниц\w*)\b"
        r"|\bи\s+(?:как|почему|когда|зачем|что|чем|где|сколько)\b"
        r"|\b(?:назов\w*|какие|расскажи|объясни|что такое)\b[^.!?]*\b(?:и|или|все|всех)\b"
        r"|\bназов\w*\s+(?:\d+|два|две|три|четыре|пять|основн\w*)\b"
        r"|\bвсе\s+(?:\w+\s+)?принцип\w*",
        question.casefold(),
    )) or question.count("?") > 1


def build_question_requirements(question: str) -> list[dict]:
    normalized = " ".join(question.casefold().replace("ё", "е").split())
    descriptions = []
    subjects = _comparison_subjects(question)
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
        if subjects:
            descriptions.extend(
                f"Объяснить условия применения варианта «{subject}»: "
                "когда его выбирать, а не только как он объявляется."
                for subject in subjects
            )
        else:
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
                            "enum": ["covered", "missing_in_answer", "not_in_sources"],
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
    schema = SemanticValidation.model_json_schema()
    schema["required"] = ["results", "coverage", "duplicates"]
    return schema


def semantic_repair_json_schema(requirements: list[dict], accepted: list[dict]) -> dict:
    schema = semantic_answer_json_schema(requirements)
    schema["properties"]["sections"]["minItems"] = 0
    section = schema["properties"]["sections"]["items"]
    section["required"].append("replaces")
    allowed_ids = [item["section_id"] for item in accepted]
    section["properties"]["replaces"] = {
        "type": "array", "uniqueItems": True,
        "items": {"type": "string", **({"enum": allowed_ids} if allowed_ids else {})},
        **({} if allowed_ids else {"maxItems": 0}),
    }
    return schema


def validate_semantic_answer(
    raw: str,
    sources: dict[str, dict],
    requirements: list[dict],
    accepted_sections: list[dict] | None = None,
) -> dict:
    try:
        response_type = SemanticRepair if accepted_sections is not None else SemanticAnswer
        parsed = response_type.model_validate_json(raw)
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
    accepted_ids = {item["section_id"] for item in accepted_sections or []}
    replacement_ids = [target for section in parsed.sections for target in section.replaces]
    if known_sections & accepted_ids:
        return _empty_result("repair_section_id_collision")
    if not set(replacement_ids).issubset(accepted_ids):
        return _empty_result("unknown_replacement_target")
    if len(replacement_ids) != len(set(replacement_ids)):
        return _empty_result("duplicate_replacement_target")
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
        if item.status != "covered" and referenced:
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
    result = _result_from_sections(sections, coverage, requirements)
    if accepted_sections is not None and not sections:
        # A schema-valid no-op is safer than forcing the repair to invent a section.
        result.update(valid=True, failure_reason=None)
    return result


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


def apply_section_validation(
    review_sections: list[dict],
    raw: str,
    requirements: list[dict] | None = None,
    all_sections: list[dict] | None = None,
    allowed_source_ids: set[str] | None = None,
) -> dict:
    expected_ids = {section["section_id"] for section in review_sections}
    all_sections = all_sections if all_sections is not None else review_sections
    known_ids = {section["section_id"] for section in all_sections}

    def failed(reason: str) -> dict:
        return {
            "valid": False,
            "accepted_section_ids": [],
            "rejected_sections": [
                {"section": section, "verdict": "validator_failed", "reason": reason}
                for section in review_sections
            ],
            "failure_reason": reason,
            "coverage_required": bool(requirements),
        }

    try:
        parsed = SemanticValidation.model_validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return failed("invalid_validation_response")
    result_ids = [item.section_id for item in parsed.results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != expected_ids:
        return failed("invalid_validation_section_ids")
    if allowed_source_ids is not None:
        for item in parsed.results:
            if not set(item.source_ids).issubset(allowed_source_ids):
                return failed("unknown_validation_source")
            if item.verdict in {"supported", "reasonable_inference", "partial"} and not item.source_ids:
                return failed("missing_validation_source")

    if requirements:
        required_ids = {item["requirement_id"] for item in requirements}
        coverage_ids = [item.requirement_id for item in parsed.coverage]
        if len(coverage_ids) != len(set(coverage_ids)) or set(coverage_ids) != required_ids:
            return failed("invalid_validation_coverage_ids")
        for item in parsed.coverage:
            if not set(item.section_ids).issubset(known_ids):
                return failed("unknown_validation_coverage_section")
            if item.status == "covered" and (not item.section_ids or item.missing_aspects):
                return failed("inconsistent_validation_coverage")
            if item.status != "covered" and item.section_ids:
                return failed("inconsistent_validation_coverage")

    positions = {item["section_id"]: index for index, item in enumerate(all_sections)}
    duplicate_ids = [item.section_id for item in parsed.duplicates]
    if len(duplicate_ids) != len(set(duplicate_ids)):
        return failed("duplicate_validation_duplicate_ids")
    for item in parsed.duplicates:
        if item.section_id not in known_ids or item.duplicate_of not in known_ids:
            return failed("unknown_duplicate_section")
        if positions[item.duplicate_of] >= positions[item.section_id]:
            return failed("invalid_duplicate_order")

    accepted = []
    rejected = []
    retained_text = {}
    review_by_id = {section["section_id"]: section for section in review_sections}
    for item in parsed.results:
        if item.verdict in {"supported", "reasonable_inference"}:
            accepted.append(item.section_id)
        else:
            if item.verdict == "partial" and item.supported_sentences:
                # Only a verbatim prefix of whole, self-contained sentences can survive.
                units = [unit.strip() for unit in re.split(
                    r"(?<=[.!?])\s+|\n+", review_by_id[item.section_id]["text"],
                ) if unit.strip()]
                prefix = item.supported_sentences
                if len(prefix) < len(units) and units[:len(prefix)] == prefix:
                    accepted.append(item.section_id)
                    retained_text[item.section_id] = " ".join(prefix)
            rejected.append({
                "section": review_by_id[item.section_id],
                "verdict": item.model_dump(),
            })
    return {
        "valid": True,
        "accepted_section_ids": accepted,
        "retained_text": retained_text,
        "retained_sources": {
            item.section_id: item.source_ids for item in parsed.results
            if item.section_id in accepted and item.source_ids
        },
        "rejected_sections": rejected,
        "failure_reason": None,
        "coverage": [item.model_dump() for item in parsed.coverage],
        "coverage_required": bool(requirements),
        "duplicates": [item.model_dump() for item in parsed.duplicates],
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
        {**section, "text": validation.get("retained_text", {}).get(
            section["section_id"], section["text"],
        ), "source_ids": validation.get("retained_sources", {}).get(
            section["section_id"], section["source_ids"],
        )} for section in grounded.get("sections", [])
        if section["section_id"] in accepted_ids
    ]
    duplicate_map = {}
    by_id = {section["section_id"]: section for section in sections}
    positions = {section["section_id"]: index for index, section in enumerate(grounded["sections"])}
    for pair in sorted(validation.get("duplicates", []), key=lambda item: positions[item["section_id"]]):
        duplicate_id, keeper_id = pair["section_id"], pair["duplicate_of"]
        while keeper_id in duplicate_map:
            keeper_id = duplicate_map[keeper_id]
        if duplicate_id in by_id and keeper_id in by_id:
            if {duplicate_id, keeper_id} & set(validation.get("retained_text", {})):
                continue
            duplicate_map[duplicate_id] = keeper_id
    sections = [item for item in sections if item["section_id"] not in duplicate_map]
    rejected = list(review_plan["blocked_sections"])
    rejected.extend(validation.get("rejected_sections", []))
    if validation.get("coverage_required"):
        previous_coverage = validation.get("coverage", [])
        if not validation.get("valid"):
            previous_coverage = [{
                "requirement_id": item["requirement_id"],
                "status": "missing_in_answer", "section_ids": [],
                "reason": "Полнота не подтверждена: проверка не завершилась корректно.",
            } for item in requirements]
    else:
        previous_coverage = grounded.get("coverage", [])
    previous_coverage = [{
        **item,
        "section_ids": list(dict.fromkeys(
            duplicate_map.get(section_id, section_id)
            for section_id in item.get("section_ids", [])
        )),
    } for item in previous_coverage]
    coverage = _coverage_for_sections(sections, previous_coverage, requirements)
    result = _result_from_sections(sections, coverage, requirements)
    result["rejected_sections"] = rejected
    result["removed_duplicate_ids"] = list(duplicate_map)
    result["duplicate_replacements"] = duplicate_map
    result["coverage_checked"] = bool(
        validation.get("coverage_required") and validation.get("valid")
    )
    if not sections and validation.get("failure_reason"):
        result["failure_reason"] = validation["failure_reason"]
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
    """Собрать черновик адресной замены; он ещё должен пройти общую проверку."""
    sections = []
    base_sections = base.get("sections", []) if base.get("valid") else []
    repair_sections = repair.get("sections", []) if repair.get("valid") else []
    replacements = {
        target: section for section in repair_sections for target in section.get("replaces", [])
    }
    inserted_ids = set()
    for section in base_sections:
        item = replacements.get(section["section_id"], section)
        if item["section_id"] not in inserted_ids:
            sections.append(dict(item))
            inserted_ids.add(item["section_id"])
    for section in repair_sections:
        if section["section_id"] not in inserted_ids:
            sections.append(dict(section))
            inserted_ids.add(section["section_id"])
    coverage = _coverage_for_sections(sections, [], requirements)
    result = _result_from_sections(sections, coverage, requirements)
    result["rejected_sections"] = (
        base.get("rejected_sections", []) + repair.get("rejected_sections", [])
    )
    return result


def commit_semantic_repair(base: dict, patch: dict, checked: dict, requirements: list[dict]) -> dict:
    """Неудачная замена не удаляет ранее принятый текст."""
    base_sections = base.get("sections", []) if base.get("valid") else []
    base_by_id = {item["section_id"]: item for item in base_sections}
    checked_by_id = {item["section_id"]: item for item in checked.get("sections", [])}
    covered_ids = {
        item["requirement_id"] for item in checked.get("coverage", [])
        if item["status"] == "covered"
    }
    accepted_changes = []
    rollback_ids = set()
    for change in patch.get("sections", []):
        targets = change.get("replaces", [])
        obligations = {
            requirement_id for target in targets
            for requirement_id in base_by_id[target].get("covers", [])
        }
        section_id = change["section_id"]
        if section_id in checked_by_id and (
            not targets or (checked.get("coverage_checked") and obligations.issubset(covered_ids))
        ):
            accepted_changes.append(checked_by_id[section_id])
        elif targets:
            rollback_ids.add(section_id)

    accepted_patch = {"valid": bool(accepted_changes), "sections": accepted_changes}
    merged = combine_semantic_results(base, accepted_patch, requirements)
    if checked.get("coverage_checked"):
        present_ids = {section["section_id"] for section in merged["sections"]}
        removed = {
            duplicate_id for duplicate_id, keeper_id in checked.get("duplicate_replacements", {}).items()
            if keeper_id in present_ids
        }
        merged["sections"] = [s for s in merged["sections"] if s["section_id"] not in removed]
    coverage = _coverage_for_sections(merged["sections"], checked.get("coverage", []), requirements)
    # Only an unchanged, previously covered section can restore coverage after rollback.
    base_coverage = _coverage_for_sections(merged["sections"], base.get("coverage", []), requirements)
    final_ids = {s["section_id"] for s in merged["sections"]}
    unchanged_ids = {s["section_id"] for s in base_sections if s["section_id"] in final_ids}
    for index, item in enumerate(coverage):
        prior = base_coverage[index]
        if item["status"] != "covered" and prior["status"] == "covered" and (
            set(prior["section_ids"]).issubset(unchanged_ids)
        ):
            coverage[index] = prior
    result = _result_from_sections(merged["sections"], coverage, requirements)
    result["coverage_checked"] = checked.get("coverage_checked", False)
    result["rejected_sections"] = base.get("rejected_sections", []) + checked.get("rejected_sections", [])
    result["rolled_back_replacement_ids"] = sorted(rollback_ids)
    if not result["valid"]:
        result["failure_reason"] = checked.get("failure_reason") or base.get("failure_reason")
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
        "source_ids": list(section["source_ids"]),
        "covers": list(section.get("covers", [])),
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
    surviving_ids = {section["section_id"] for section in sections}
    coverage = []
    for requirement in requirements:
        requirement_id = requirement["requirement_id"]
        prior = previous.get(requirement_id)
        if prior is not None:
            item = dict(prior)
            referenced = set(item.get("section_ids", []))
            if item["status"] == "covered" and (
                not referenced or not referenced.issubset(surviving_ids)
            ):
                item.update(
                    status="blocked_by_validation", section_ids=[],
                    reason="Часть ответа, необходимая для требования, не прошла проверку.",
                )
            coverage.append(item)
            continue
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
                "status": "missing_in_answer",
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
