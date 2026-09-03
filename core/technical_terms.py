import json
import re
from functools import lru_cache
from pathlib import Path


_TERMS_PATH = Path(__file__).resolve().parent.parent / "technical_terms.json"


def _normalize(text: str) -> str:
    return " ".join(
        text.casefold().replace("ё", "е").replace("@", "").split()
    )


def _contains(text: str, value: str) -> bool:
    normalized_text = _normalize(text)
    normalized_value = _normalize(value)
    if not normalized_value:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_value).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, normalized_text, flags=re.UNICODE) is not None


@lru_cache(maxsize=1)
def load_technical_terms() -> dict[str, tuple[str, ...]]:
    if not _TERMS_PATH.exists():
        return {}
    with _TERMS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("technical_terms.json должен содержать объект")

    terms = {}
    for canonical, aliases in payload.items():
        if not isinstance(canonical, str) or not isinstance(aliases, list):
            raise ValueError("Некорректная запись в technical_terms.json")
        values = [canonical, *(alias for alias in aliases if isinstance(alias, str))]
        terms[canonical] = tuple(dict.fromkeys(values))
    return terms


def canonical_identifier(value: str) -> str | None:
    normalized = _normalize(value)
    for canonical, aliases in load_technical_terms().items():
        if any(normalized == _normalize(alias) for alias in aliases):
            return canonical.casefold()
    return None


def identifier_support_source(
    identifier: str,
    evidence_text: str,
    question: str = "",
) -> str | None:
    canonical = canonical_identifier(identifier)
    if canonical is None:
        if _contains(evidence_text, identifier):
            return "evidence"
        if _contains(question, identifier):
            return "question"
        return None

    aliases = next(
        aliases
        for name, aliases in load_technical_terms().items()
        if name.casefold() == canonical
    )
    if any(_contains(evidence_text, alias) for alias in aliases):
        return "evidence"
    if any(_contains(question, alias) for alias in aliases):
        return "question"
    return None


def relevant_term_guidance(text: str) -> str:
    lines = []
    for canonical, aliases in load_technical_terms().items():
        matched = [alias for alias in aliases if _contains(text, alias)]
        if not matched:
            continue
        displayed = ", ".join(dict.fromkeys(matched))
        lines.append(f"{displayed} = {canonical}")
    return "\n".join(lines)
