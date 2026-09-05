import json
import unittest

from core.answer_grounding import (
    GROUNDING_FAILED_ANSWER,
    answer_json_schema,
    apply_semantic_validation,
    build_semantic_review_plan,
    merge_semantic_review,
    validate_grounded_answer,
)


class GroundedAnswerValidationTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            "T1": {
                "text": "Класс можно представить как описание нового типа объектов.",
                "source": {"filename": "lecture.mp3"},
            },
            "T2": {
                "text": "После определения класса на его основе можно создавать экземпляры.",
                "source": {"filename": "lecture.mp3"},
            },
            "T3": {
                "text": "Конкретный экземпляр этого класса будет представлять определённую книгу.",
                "source": {"filename": "lecture.mp3"},
            },
        }

    def validate(self, payload):
        return validate_grounded_answer(
            json.dumps(payload, ensure_ascii=False),
            self.sources,
        )

    def test_accepts_supported_claim_with_exact_quote(self):
        result = self.validate({
            "claims": [{
                "text": "Класс описывает новый тип объектов.",
                "evidence": [{
                    "source_id": "T1",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        })

        self.assertTrue(result["valid"])
        self.assertEqual(result["answer"], "Класс описывает новый тип объектов.")
        self.assertEqual(result["source_ids"], ["T1"])
        self.assertEqual(
            result["quotes"]["T1"],
            ["Класс можно представить как описание нового типа объектов."],
        )

    def test_drops_claim_with_changed_quote(self):
        result = self.validate({
            "claims": [
                {
                    "text": "Класс описывает новый тип объектов.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": "Класс является шаблоном нового типа объектов.",
                    }],
                },
                {
                    "text": "На основе класса можно создавать экземпляры.",
                    "evidence": [{
                        "source_id": "T2",
                        "quote": "на его основе можно создавать экземпляры",
                    }],
                },
            ],
        })

        self.assertTrue(result["valid"])
        self.assertEqual(
            result["answer"],
            "На основе класса можно создавать экземпляры.",
        )
        self.assertEqual(result["source_ids"], ["T2"])

    def test_rejects_unknown_source(self):
        result = self.validate({
            "claims": [{
                "text": "Класс описывает новый тип объектов.",
                "evidence": [{
                    "source_id": "T9",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        })

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)
        self.assertEqual(
            result["failure_reason"],
            "claims_failed_structural_validation",
        )
        self.assertEqual(result["source_ids"], [])

    def test_rejects_multiple_sentences_in_one_claim(self):
        result = self.validate({
            "claims": [{
                "text": "Класс описывает тип объектов. Экземпляр создаётся на его основе.",
                "evidence": [
                    {
                        "source_id": "T1",
                        "quote": "Класс можно представить как описание нового типа объектов.",
                    },
                    {
                        "source_id": "T2",
                        "quote": "на его основе можно создавать экземпляры",
                    },
                ],
            }],
        })

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)

    def test_rejects_generator_insufficient_status(self):
        result = self.validate({"status": "insufficient", "claims": []})

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)
        self.assertEqual(result["failure_reason"], "invalid_model_response")
        self.assertEqual(result["source_ids"], [])

    def test_rejects_invalid_json(self):
        result = validate_grounded_answer("не json", self.sources)

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)
        self.assertEqual(result["failure_reason"], "invalid_model_response")

    def test_semantic_validation_keeps_only_entailed_claims(self):
        grounded = self.validate({
            "claims": [
                {
                    "text": "Класс описывает новый тип объектов.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": "Класс можно представить как описание нового типа объектов.",
                    }],
                },
                {
                    "text": "Все экземпляры обязательно хранят разные данные.",
                    "evidence": [{
                        "source_id": "T2",
                        "quote": "на его основе можно создавать экземпляры",
                    }],
                },
            ],
        })
        verdicts = json.dumps({
            "results": [
                {"claim_id": "C1", "verdict": "entailed"},
                {"claim_id": "C2", "verdict": "partial"},
            ],
        })

        result = apply_semantic_validation(grounded, verdicts)

        self.assertTrue(result["valid"])
        self.assertEqual(result["answer"], "Класс описывает новый тип объектов.")
        self.assertEqual(result["source_ids"], ["T1"])

    def test_semantic_validation_rejects_generalization(self):
        grounded = self.validate({
            "claims": [{
                "text": "У всех экземпляров значения атрибутов различаются.",
                "evidence": [{
                    "source_id": "T2",
                    "quote": "на его основе можно создавать экземпляры",
                }],
            }],
        })
        verdicts = json.dumps({
            "results": [{
                "claim_id": "C1",
                "verdict": "partial",
                "reason": "Конкретный материал не подтверждает общее правило.",
                "unsupported_parts": ["у всех экземпляров"],
            }],
        }, ensure_ascii=False)

        result = apply_semantic_validation(grounded, verdicts)

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)
        self.assertEqual(result["source_ids"], [])

    def test_semantic_validation_rejects_invalid_response(self):
        grounded = self.validate({
            "claims": [{
                "text": "Класс описывает новый тип объектов.",
                "evidence": [{
                    "source_id": "T1",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        })

        result = apply_semantic_validation(grounded, "не json")

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)

    def test_semantic_validation_rejects_unknown_claim_id(self):
        grounded = self.validate({
            "claims": [{
                "text": "Класс описывает новый тип объектов.",
                "evidence": [{
                    "source_id": "T1",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        })
        verdicts = json.dumps({
            "results": [{"claim_id": "C9", "verdict": "entailed"}],
        })

        result = apply_semantic_validation(grounded, verdicts)

        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], GROUNDING_FAILED_ANSWER)

    def test_risk_gate_accepts_close_paraphrase_without_llm(self):
        grounded = self.validate({
            "claims": [{
                "text": "Класс описывает новый тип объектов.",
                "evidence": [{
                    "source_id": "T1",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        })

        plan = build_semantic_review_plan(grounded)

        self.assertEqual(plan["safe_claim_ids"], ["C1"])
        self.assertEqual(plan["review"]["claims"], [])
        self.assertEqual(plan["blocked_claims"], [])

    def test_risk_gate_does_not_review_new_terms_without_other_risk(self):
        sources = {
            "T1": {
                "text": (
                    "Если объект отвечает за собственное корректное состояние, "
                    "мы используем идею инкапсуляции."
                ),
                "source": {"filename": "lecture.mp3"},
            },
        }
        grounded = validate_grounded_answer(
            json.dumps({
                "claims": [{
                    "text": (
                        "Инкапсуляция уместна, когда объект самостоятельно "
                        "отвечает за собственное корректное состояние."
                    ),
                    "evidence": [{
                        "source_id": "T1",
                        "quote": sources["T1"]["text"],
                    }],
                }],
            }, ensure_ascii=False),
            sources,
        )

        plan = build_semantic_review_plan(grounded)

        self.assertEqual(plan["safe_claim_ids"], ["C1"])
        self.assertEqual(plan["review"]["claims"], [])
        self.assertNotIn("C1", plan["risk_reasons"])

    def test_grounded_answer_has_no_fixed_claim_count_limit(self):
        payload = {
            "claims": [
                {
                    "text": "Класс можно представить как описание нового типа объектов.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": "Класс можно представить как описание нового типа объектов.",
                    }],
                }
                for _ in range(12)
            ],
        }

        grounded = self.validate(payload)

        self.assertTrue(grounded["valid"])
        self.assertEqual(len(grounded["claims"]), 12)
        self.assertNotIn("maxItems", answer_json_schema()["properties"]["claims"])

    def test_risk_gate_accepts_known_asr_identifier_alias(self):
        sources = {
            "T1": {
                "text": (
                    "STR отвечает на вопрос, как показать объект пользователю. "
                    "REPR отвечает на вопрос, как полезнее представить этот "
                    "объект разработчику."
                ),
                "source": {"filename": "lecture.mp3"},
            },
        }
        grounded = validate_grounded_answer(
            json.dumps({
                "claims": [{
                    "text": (
                        "Метод __str__ предназначен для пользователя, а "
                        "__repr__ — для разработчика."
                    ),
                    "evidence": [{
                        "source_id": "T1",
                        "quote": sources["T1"]["text"],
                    }],
                }],
            }, ensure_ascii=False),
            sources,
        )

        plan = build_semantic_review_plan(
            grounded,
            question="Чем __str__ отличается от __repr__?",
        )

        self.assertEqual(plan["blocked_claims"], [])
        self.assertNotIn("new_identifier", plan["risk_reasons"].get("C1", []))

    def test_identifier_from_question_is_not_treated_as_new_fact(self):
        sources = {
            "T1": {
                "text": "Этот метод возвращает элемент по переданному ключу.",
                "source": {"filename": "lecture.mp3"},
            },
        }
        grounded = validate_grounded_answer(
            json.dumps({
                "claims": [{
                    "text": "Метод __getitem__ возвращает элемент по ключу.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": sources["T1"]["text"],
                    }],
                }],
            }, ensure_ascii=False),
            sources,
        )

        plan = build_semantic_review_plan(
            grounded,
            question="Что делает __getitem__?",
        )

        self.assertEqual(plan["blocked_claims"], [])
        self.assertNotIn("new_identifier", plan["risk_reasons"].get("C1", []))
        self.assertIn(
            "identifier_from_question",
            plan["risk_reasons"].get("C1", []),
        )

    def test_risk_gate_sends_stronger_scope_to_llm(self):
        grounded = self.validate({
            "claims": [{
                "text": "Все экземпляры обязательно хранят разные данные.",
                "evidence": [{
                    "source_id": "T2",
                    "quote": "на его основе можно создавать экземпляры",
                }],
            }],
        })

        plan = build_semantic_review_plan(grounded)

        self.assertEqual(plan["safe_claim_ids"], [])
        self.assertEqual(
            [claim["claim_id"] for claim in plan["review"]["claims"]],
            ["C1"],
        )
        self.assertIn("stronger_scope", plan["risk_reasons"]["C1"])

    def test_risk_gate_sends_example_generalization_to_llm(self):
        grounded = self.validate({
            "claims": [{
                "text": (
                    "Конкретный экземпляр класса представляет определённый "
                    "объект этого типа, например конкретную книгу."
                ),
                "evidence": [{
                    "source_id": "T3",
                    "quote": "Конкретный экземпляр этого класса будет представлять определённую книгу.",
                }],
            }],
        })

        plan = build_semantic_review_plan(grounded)

        self.assertEqual(
            [claim["claim_id"] for claim in plan["review"]["claims"]],
            ["C1"],
        )
        self.assertIn("example_generalization", plan["risk_reasons"]["C1"])
        self.assertIn("several_new_terms", plan["risk_reasons"]["C1"])

    def test_risk_gate_blocks_new_number_without_llm(self):
        grounded = self.validate({
            "claims": [{
                "text": "На основе класса можно создать 10 экземпляров.",
                "evidence": [{
                    "source_id": "T2",
                    "quote": "на его основе можно создавать экземпляры",
                }],
            }],
        })

        plan = build_semantic_review_plan(grounded)

        self.assertEqual(plan["review"]["claims"], [])
        self.assertEqual(
            [claim["claim_id"] for claim in plan["blocked_claims"]],
            ["C1"],
        )
        self.assertIn("new_number", plan["risk_reasons"]["C1"])

    def test_merge_keeps_safe_and_entailed_claims_in_original_order(self):
        grounded = self.validate({
            "claims": [
                {
                    "text": "Все экземпляры обязательно хранят разные данные.",
                    "evidence": [{
                        "source_id": "T2",
                        "quote": "на его основе можно создавать экземпляры",
                    }],
                },
                {
                    "text": "Класс описывает новый тип объектов.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": "Класс можно представить как описание нового типа объектов.",
                    }],
                },
            ],
        })
        plan = build_semantic_review_plan(grounded)
        verdicts = json.dumps({
            "results": [{"claim_id": "C1", "verdict": "entailed"}],
        })
        reviewed = apply_semantic_validation(plan["review"], verdicts)

        result = merge_semantic_review(grounded, plan, reviewed)

        self.assertEqual(result["claims"][0]["claim_id"], "C1")
        self.assertEqual(result["claims"][1]["claim_id"], "C2")

    def test_merge_keeps_safe_claim_when_llm_validation_fails(self):
        grounded = self.validate({
            "claims": [
                {
                    "text": "Класс описывает новый тип объектов.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": "Класс можно представить как описание нового типа объектов.",
                    }],
                },
                {
                    "text": "Все экземпляры обязательно хранят разные данные.",
                    "evidence": [{
                        "source_id": "T2",
                        "quote": "на его основе можно создавать экземпляры",
                    }],
                },
            ],
        })
        plan = build_semantic_review_plan(grounded)
        reviewed = apply_semantic_validation(plan["review"], "не json")

        result = merge_semantic_review(grounded, plan, reviewed)

        self.assertTrue(result["valid"])
        self.assertEqual(result["answer"], "Класс описывает новый тип объектов.")


if __name__ == "__main__":
    unittest.main()
