import json
import unittest

from core.semantic_grounding import (
    apply_section_validation,
    build_question_requirements,
    build_response_style,
    build_section_review_plan,
    merge_section_review,
    response_language_matches,
    select_answer_reasoning,
    semantic_answer_json_schema,
    validate_semantic_answer,
)


class SemanticGroundingTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            "T1": {
                "text": (
                    "Метод класса получает ссылку на сам класс. "
                    "Его используют как альтернативный конструктор."
                ),
                "source": {"filename": "lecture.mp3"},
            },
        }

    def test_composite_question_builds_explicit_requirements(self):
        requirements = build_question_requirements(
            "Сравни все четыре принципа: за что отвечает каждый и когда применяется?"
        )

        descriptions = " ".join(item["description"] for item in requirements)
        self.assertIn("все запрошенные элементы", descriptions)
        self.assertIn("сущность или назначение", descriptions)
        self.assertIn("когда применяется", descriptions)
        self.assertIn("различия", descriptions)
        self.assertEqual(select_answer_reasoning(requirements), "medium")

    def test_simple_question_uses_low_reasoning(self):
        requirements = build_question_requirements("Что такое метод класса?")

        self.assertEqual(len(requirements), 1)
        self.assertEqual(select_answer_reasoning(requirements), "low")

    def test_russian_question_rejects_english_answer(self):
        self.assertFalse(response_language_matches(
            "Как в лекциях объясняется полиморфизм?",
            [{"text": "Polymorphism means one interface with many implementations."}],
        ))

    def test_russian_question_accepts_russian_answer_with_identifiers(self):
        self.assertTrue(response_language_matches(
            "Как в лекциях объясняется полиморфизм?",
            [{"text": "Метод send() имеет разные реализации у разных типов."}],
        ))

    def test_brief_style_keeps_composite_coverage_with_small_budget(self):
        question = (
            "Сравни все четыре принципа: для каждого кратко объясни, "
            "за что он отвечает, когда применяется и чем отличается."
        )
        requirements = build_question_requirements(question)

        style = build_response_style(question, requirements)

        self.assertEqual(style["mode"], "brief")
        self.assertEqual(len(requirements), 4)
        self.assertEqual(style["target_words"], 200)
        self.assertEqual(style["upper_word_budget"], 280)

    def test_standard_style_is_shorter_than_detailed_style(self):
        standard_question = (
            "Чем метод класса отличается от статического метода "
            "и когда использовать каждый?"
        )
        detailed_question = (
            "Расскажи подробно, что такое инкапсуляция, зачем она нужна "
            "и когда применяется."
        )
        standard_requirements = build_question_requirements(standard_question)
        detailed_requirements = build_question_requirements(detailed_question)

        standard = build_response_style(standard_question, standard_requirements)
        detailed = build_response_style(detailed_question, detailed_requirements)

        self.assertEqual(standard["mode"], "standard")
        self.assertEqual(detailed["mode"], "detailed")
        self.assertLess(
            standard["upper_word_budget"],
            detailed["upper_word_budget"],
        )

    def test_natural_paraphrase_needs_no_exact_quote(self):
        requirements = build_question_requirements("Что такое метод класса?")
        raw = json.dumps({
            "sections": [{
                "section_id": "S1",
                "text": "Метод класса работает с самим классом.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R1"],
            }],
            "coverage": [{
                "requirement_id": "R1",
                "status": "covered",
                "section_ids": ["S1"],
                "reason": "",
            }],
        }, ensure_ascii=False)

        grounded = validate_semantic_answer(raw, self.sources, requirements)
        plan = build_section_review_plan(
            grounded,
            self.sources,
            "Что такое метод класса?",
        )

        self.assertTrue(grounded["valid"])
        self.assertEqual(grounded["answer"], "Метод класса работает с самим классом.")
        self.assertEqual(plan["safe_section_ids"], ["S1"])
        self.assertNotIn("quote", semantic_answer_json_schema(requirements)["properties"])

    def test_new_number_is_blocked_locally(self):
        requirements = build_question_requirements("Что такое метод класса?")
        raw = json.dumps({
            "sections": [{
                "section_id": "S1",
                "text": "Метод класса создаёт 10 объектов.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R1"],
            }],
            "coverage": [{
                "requirement_id": "R1",
                "status": "covered",
                "section_ids": ["S1"],
                "reason": "",
            }],
        }, ensure_ascii=False)

        grounded = validate_semantic_answer(raw, self.sources, requirements)
        plan = build_section_review_plan(grounded, self.sources, "Что такое метод класса?")

        self.assertEqual(plan["safe_section_ids"], [])
        self.assertEqual(plan["review_sections"], [])
        self.assertEqual(plan["blocked_sections"][0]["reasons"], ["new_number"])

    def test_identifier_from_normalized_summary_is_allowed(self):
        requirements = build_question_requirements("Что такое метод класса?")
        sources = {
            "T1": {
                "text": "Метод класса получает ссылку на сам класс.",
                "normalized_context": "Декоратор DomainPolicy связан с методом класса.",
                "source": {"filename": "lecture.mp3"},
            },
        }
        raw = json.dumps({
            "sections": [{
                "section_id": "S1",
                "text": "Метод DomainPolicy получает ссылку на сам класс.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R1"],
            }],
            "coverage": [{
                "requirement_id": "R1",
                "status": "covered",
                "section_ids": ["S1"],
                "reason": "",
            }],
        }, ensure_ascii=False)

        grounded = validate_semantic_answer(raw, sources, requirements)
        plan = build_section_review_plan(
            grounded,
            sources,
            "Что такое метод класса?",
        )

        self.assertEqual(plan["safe_section_ids"], ["S1"])
        self.assertEqual(plan["blocked_sections"], [])

    def test_unknown_identifier_is_reviewed_instead_of_destroying_section(self):
        requirements = build_question_requirements("Что такое метод класса?")
        raw = json.dumps({
            "sections": [{
                "section_id": "S1",
                "text": "Метод NovelAPI получает ссылку на сам класс.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R1"],
            }],
            "coverage": [{
                "requirement_id": "R1",
                "status": "covered",
                "section_ids": ["S1"],
                "reason": "",
            }],
        }, ensure_ascii=False)

        grounded = validate_semantic_answer(raw, self.sources, requirements)
        plan = build_section_review_plan(
            grounded,
            self.sources,
            "Что такое метод класса?",
        )

        self.assertEqual(plan["safe_section_ids"], [])
        self.assertEqual(len(plan["review_sections"]), 1)
        self.assertEqual(plan["blocked_sections"], [])
        self.assertIn("new_identifier", plan["risk_reasons"]["S1"])

    def test_reasonable_inference_is_kept_and_unsupported_is_removed(self):
        requirements = build_question_requirements(
            "Что такое метод класса и когда он применяется?"
        )
        raw = json.dumps({
            "sections": [
                {
                    "section_id": "S1",
                    "text": "Метод класса работает с самим классом.",
                    "source_ids": ["T1"],
                    "basis": "source",
                    "covers": ["R1"],
                },
                {
                    "section_id": "S2",
                    "text": "Он уместен для создания объектов альтернативным способом.",
                    "source_ids": ["T1"],
                    "basis": "inference",
                    "covers": ["R2"],
                },
            ],
            "coverage": [
                {
                    "requirement_id": "R1",
                    "status": "covered",
                    "section_ids": ["S1"],
                    "reason": "",
                },
                {
                    "requirement_id": "R2",
                    "status": "covered",
                    "section_ids": ["S2"],
                    "reason": "",
                },
            ],
        }, ensure_ascii=False)
        grounded = validate_semantic_answer(raw, self.sources, requirements)
        plan = build_section_review_plan(
            grounded,
            self.sources,
            "Что такое метод класса и когда он применяется?",
        )
        validation = apply_section_validation(
            plan["review_sections"],
            json.dumps({
                "results": [{
                    "section_id": "S2",
                    "verdict": "reasonable_inference",
                    "reason": "Следует из назначения альтернативного конструктора.",
                }],
            }, ensure_ascii=False),
        )
        merged = merge_section_review(
            grounded,
            plan,
            validation,
            requirements,
        )

        self.assertTrue(merged["valid"])
        self.assertEqual(merged["uncovered_requirement_ids"], [])
        self.assertIn("альтернативным способом", merged["answer"])

    def test_validator_failure_marks_overlapping_coverage_as_lost(self):
        requirements = build_question_requirements("Что такое метод класса?")
        raw = json.dumps({
            "sections": [
                {
                    "section_id": "S1",
                    "text": "Метод класса работает с самим классом.",
                    "source_ids": ["T1"],
                    "basis": "source",
                    "covers": ["R1"],
                },
                {
                    "section_id": "S2",
                    "text": "Из этого следует связь с самим классом.",
                    "source_ids": ["T1"],
                    "basis": "inference",
                    "covers": ["R1"],
                },
            ],
            "coverage": [{
                "requirement_id": "R1",
                "status": "covered",
                "section_ids": ["S1", "S2"],
                "reason": "",
            }],
        }, ensure_ascii=False)
        grounded = validate_semantic_answer(raw, self.sources, requirements)
        plan = build_section_review_plan(
            grounded,
            self.sources,
            "Что такое метод класса?",
        )
        validation = apply_section_validation(plan["review_sections"], "")

        merged = merge_section_review(
            grounded,
            plan,
            validation,
            requirements,
        )

        self.assertEqual(merged["uncovered_requirement_ids"], [])
        self.assertEqual(merged["lost_requirement_ids"], ["R1"])


if __name__ == "__main__":
    unittest.main()
