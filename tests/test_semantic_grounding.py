import json
import unittest

from core.semantic_grounding import (
    apply_section_validation,
    build_question_requirements,
    build_response_style,
    build_section_review_plan,
    combine_semantic_results,
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

    def test_brief_style_keeps_composite_coverage_without_word_quota(self):
        question = (
            "Сравни все четыре принципа: для каждого кратко объясни, "
            "за что он отвечает, когда применяется и чем отличается."
        )
        requirements = build_question_requirements(question)

        style = build_response_style(question, requirements)

        self.assertEqual(style["mode"], "brief")
        self.assertEqual(len(requirements), 4)
        self.assertNotIn("target_words", style)
        self.assertNotIn("upper_word_budget", style)
        self.assertIn("не причины, шаги", style["structure"])
        self.assertIn("каждого объекта", style["coverage"])

    def test_style_changes_explanation_depth_not_word_targets(self):
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
        for style in (standard, detailed):
            self.assertNotIn("target_words", style)
            self.assertNotIn("upper_word_budget", style)
            self.assertIn("stop_when", style)
        self.assertIn("Прямой ответ", standard["structure"])
        self.assertIn("причины, шаги и условия", detailed["structure"])

    def test_name_question_has_no_extra_example_or_length_requirement(self):
        question = "От какого типа наследуется словарь в примере?"
        requirements = build_question_requirements(question)
        style = build_response_style(question, requirements)

        self.assertEqual(len(requirements), 1)
        self.assertNotIn("Привести пример", requirements[0]["description"])
        self.assertNotIn("target_words", style)
        self.assertIn("полезным уточнением", style["structure"])

    def test_explain_each_is_separate_from_naming_items(self):
        names = build_question_requirements("Перечисли четыре принципа ООП.")
        explained = build_question_requirements(
            "Перечисли четыре принципа ООП и объясни каждый."
        )

        self.assertEqual(len(names), 1)
        self.assertEqual(len(explained), 2)
        self.assertIn("каждого запрошенного понятия", explained[1]["description"])
        self.assertIn("одних названий недостаточно", explained[1]["description"])

    def test_why_and_how_require_explanation_even_when_brief(self):
        cases = [
            ("Кратко: почему super() не просто вызов родителя?", "причину"),
            ("Кратко: как словарь запрещает перезапись?", "последовательность"),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                requirements = build_question_requirements(question)
                self.assertIn(expected, requirements[0]["description"])
                self.assertEqual(build_response_style(question, requirements)["mode"], "brief")

    def test_comparison_with_usage_covers_both_sides(self):
        requirements = build_question_requirements(
            "Сравни наследование и композицию: чем отличаются и когда использовать?"
        )
        descriptions = " ".join(item["description"] for item in requirements)

        self.assertEqual(len(requirements), 2)
        self.assertIn("обеих сторон", descriptions)
        self.assertIn("различия", descriptions)

    def test_explicit_examples_are_not_lost(self):
        for question in (
            "Объясни полиморфизм с примером.",
            "Приведи два примера полиморфизма.",
            "Поясни наследование на примере.",
        ):
            with self.subTest(question=question):
                descriptions = " ".join(
                    item["description"] for item in build_question_requirements(question)
                )
                self.assertIn("Привести пример", descriptions)

    def test_merge_never_reintroduces_sections_from_invalid_result(self):
        requirements = build_question_requirements("Что такое метод класса?")
        english = {
            "section_id": "S1", "text": "A class method receives the class.",
            "source_ids": ["T2"], "basis": "source", "covers": ["R1"],
        }
        russian = {
            "section_id": "S1", "text": "Метод класса работает с самим классом.",
            "source_ids": ["T1"], "basis": "source", "covers": ["R1"],
        }
        for base, repair in (
            ({"valid": False, "sections": [english]}, {"valid": True, "sections": [russian]}),
            ({"valid": True, "sections": [russian]}, {"valid": False, "sections": [english]}),
        ):
            with self.subTest(base_valid=base["valid"]):
                merged = combine_semantic_results(base, repair, requirements)
                self.assertEqual(merged["answer"], russian["text"])
                self.assertEqual(merged["source_ids"], ["T1"])
                self.assertEqual(len(merged["sections"]), 1)

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
