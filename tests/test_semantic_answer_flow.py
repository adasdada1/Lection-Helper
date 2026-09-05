import asyncio
import json
import unittest
from unittest.mock import patch

import core.rag as rag


class SemanticAnswerFlowTests(unittest.TestCase):
    def setUp(self):
        self.chat = {
            "chat_id": "chat-1",
            "course_id": "course-1",
            "title": "Чат",
        }
        self.evidence = {
            "T1": {
                "text": (
                    "Метод класса получает ссылку на сам класс. "
                    "Его используют как альтернативный конструктор."
                ),
                "source": {
                    "evidence_id": "T1",
                    "filename": "lecture.mp3",
                    "content_kind": "transcript",
                    "text_preview": "",
                },
            },
        }

    def response(self, sections, coverage):
        return {
            "answer": json.dumps({
                "sections": sections,
                "coverage": coverage,
            }, ensure_ascii=False),
            "model": "deepseek-v4-flash",
            "usage": {"prompt_tokens": 100, "completion_tokens": 100},
            "estimated_cost_upper_usd": 0.000176,
        }

    def test_semantic_context_contains_summary_and_transcript_without_quotes(self):
        matches = [
            {
                "id": "t1",
                "text": "Транскрипт подробно объясняет назначение метода класса.",
                "metadata": {
                    "content_kind": "transcript",
                    "filename": "lecture.mp3",
                    "doc_id": "doc-1",
                    "chunk_index": 1,
                },
            },
            {
                "id": "s1",
                "text": "ИИ-конспект нормализует объяснение метода класса.",
                "metadata": {
                    "content_kind": "summary",
                    "filename": "lecture.mp3",
                    "doc_id": "doc-1",
                    "chunk_index": 1,
                },
            },
        ]
        requirements = rag.build_question_requirements("Что такое метод класса?")
        with (
            patch.object(rag.chat_memory, "get_summary", return_value=None),
            patch.object(rag.chat_memory, "get_recent_messages", return_value=[]),
            patch("core.chunk_times.get_times", return_value={}),
        ):
            messages, evidence = rag._build_messages(
                "Что такое метод класса?",
                matches,
                "chat-1",
                semantic_mode=True,
                requirements=requirements,
            )

        self.assertIn("своими словами", messages[0]["content"])
        self.assertIn("Отвечай на языке последнего вопроса", messages[0]["content"])
        self.assertIn(
            "весь пользовательский текст в sections должен быть на русском",
            messages[0]["content"],
        )
        self.assertIn("нормализованный учебный материал", messages[-1]["content"])
        self.assertIn("ИИ-конспект нормализует", messages[-1]["content"])
        self.assertIn("Транскрипт подробно", messages[-1]["content"])
        self.assertIn('"mode": "standard"', messages[-1]["content"])
        self.assertIn('"upper_word_budget": 220', messages[-1]["content"])
        self.assertNotIn("В поле quote", messages[-1]["content"])
        self.assertEqual(list(evidence), ["T1"])
        self.assertIn(
            "ИИ-конспект нормализует",
            evidence["T1"]["normalized_context"],
        )

    def run_ask(self, generated, repair=None, validation=None):
        with (
            patch.object(rag, "GROUNDING_MODE", "semantic"),
            patch.object(rag.chat_memory, "get_chat", return_value=self.chat),
            patch.object(rag.chat_memory, "add_message"),
            patch.object(rag.chat_memory, "maybe_summarize"),
            patch.object(rag.chat_memory, "get_message_count", return_value=1),
            patch.object(
                rag,
                "_load_short_scoped_transcript",
                return_value=[{"id": "c1"}],
            ),
            patch.object(
                rag,
                "_build_messages",
                return_value=([{"role": "user", "content": "q"}], self.evidence),
            ),
            patch.object(
                rag,
                "call_llm_for_semantic_answer",
                return_value=generated,
            ) as answer_call,
            patch.object(
                rag,
                "call_llm_for_semantic_validation",
                return_value=validation,
            ) as validator_call,
            patch.object(
                rag,
                "call_llm_for_semantic_repair",
                return_value=repair,
            ) as repair_call,
        ):
            result = asyncio.run(
                rag.ask(
                    "Что такое метод класса и когда он применяется?",
                    "chat-1",
                    scope_doc_id="doc-1",
                )
            )
        return result, answer_call, validator_call, repair_call

    def test_complete_semantic_answer_is_returned_naturally(self):
        generated = self.response(
            [
                {
                    "section_id": "S1",
                    "text": "Метод класса получает ссылку на сам класс.",
                    "source_ids": ["T1"],
                    "basis": "source",
                    "covers": ["R1"],
                },
                {
                    "section_id": "S2",
                    "text": "Метод класса используют как альтернативный конструктор.",
                    "source_ids": ["T1"],
                    "basis": "source",
                    "covers": ["R2"],
                },
            ],
            [
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
        )

        result, answer_call, validator_call, repair_call = self.run_ask(generated)

        self.assertEqual(result["grounding_status"], "supported")
        self.assertIn("сам класс", result["answer"])
        self.assertIn("альтернативный конструктор", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(answer_call.call_args.args[2], "medium")
        validator_call.assert_not_called()
        repair_call.assert_not_called()

    def test_validator_removal_triggers_targeted_repair(self):
        generated = self.response(
            [
                {
                    "section_id": "S1",
                    "text": "Метод класса получает ссылку на сам класс.",
                    "source_ids": ["T1"],
                    "basis": "source",
                    "covers": ["R1"],
                },
                {
                    "section_id": "S2",
                    "text": "Поэтому он подходит для создания объектов.",
                    "source_ids": ["T1"],
                    "basis": "inference",
                    "covers": ["R2"],
                },
            ],
            [
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
        )
        validation = {
            "answer": json.dumps({
                "results": [{
                    "section_id": "S2",
                    "verdict": "unsupported",
                    "reason": "Слишком широкая формулировка.",
                }],
            }, ensure_ascii=False),
            "model": "deepseek-v4-flash",
        }
        repair = self.response(
            [{
                "section_id": "S1",
                "text": "Метод класса используют как альтернативный конструктор.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R2"],
            }],
            [{
                "requirement_id": "R2",
                "status": "covered",
                "section_ids": ["S1"],
                "reason": "",
            }],
        )

        result, _, validator_call, repair_call = self.run_ask(
            generated,
            repair,
            validation,
        )

        self.assertEqual(result["grounding_status"], "supported_after_retry")
        self.assertIn("альтернативный конструктор", result["answer"])
        self.assertEqual(validator_call.call_count, 1)
        self.assertEqual(repair_call.call_count, 1)
        repair_messages = repair_call.call_args.args[0]
        self.assertIn(
            "Отвечай на языке последнего вопроса",
            repair_messages[0]["content"],
        )

    def test_missing_requirement_gets_one_targeted_repair(self):
        generated = self.response(
            [{
                "section_id": "S1",
                "text": "Метод класса получает ссылку на сам класс.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R1"],
            }],
            [
                {
                    "requirement_id": "R1",
                    "status": "covered",
                    "section_ids": ["S1"],
                    "reason": "",
                },
                {
                    "requirement_id": "R2",
                    "status": "not_in_sources",
                    "section_ids": [],
                    "reason": "Не найдено.",
                },
            ],
        )
        repair = self.response(
            [{
                "section_id": "S1",
                "text": "Метод класса используют как альтернативный конструктор.",
                "source_ids": ["T1"],
                "basis": "source",
                "covers": ["R2"],
            }],
            [{
                "requirement_id": "R2",
                "status": "covered",
                "section_ids": ["S1"],
                "reason": "",
            }],
        )

        result, _, validator_call, repair_call = self.run_ask(generated, repair)

        self.assertEqual(result["grounding_status"], "supported_after_retry")
        self.assertIn("сам класс", result["answer"])
        self.assertIn("альтернативный конструктор", result["answer"])
        self.assertEqual(repair_call.call_count, 1)
        missing_payload = repair_call.call_args.args[0][1]["content"]
        self.assertIn('"requirement_id": "R2"', missing_payload)
        validator_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
