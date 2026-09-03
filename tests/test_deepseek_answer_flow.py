import asyncio
import json
import unittest
from unittest.mock import patch

import core.rag as rag
from core.deepseek_client import DeepSeekCompletionError


class DeepSeekAnswerFlowTests(unittest.TestCase):
    def setUp(self):
        self.chat = {
            "chat_id": "chat-1",
            "course_id": "course-1",
            "title": "Чат",
        }
        self.evidence = {
            "T1": {
                "text": "Класс можно представить как описание нового типа объектов.",
                "source": {
                    "evidence_id": "T1",
                    "filename": "lecture.mp3",
                    "content_kind": "transcript",
                },
            },
        }
        self.valid_answer = json.dumps({
            "claims": [{
                "text": "Класс можно представить как описание нового типа объектов.",
                "evidence": [{
                    "source_id": "T1",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        }, ensure_ascii=False)

    def ask(self, answers):
        with (
            patch.object(rag.chat_memory, "get_chat", return_value=self.chat),
            patch.object(rag.chat_memory, "add_message"),
            patch.object(rag.chat_memory, "maybe_summarize"),
            patch.object(rag.chat_memory, "get_message_count", return_value=1),
            patch.object(rag, "_load_short_scoped_transcript", return_value=[{"id": "c1"}]),
            patch.object(
                rag,
                "_build_messages",
                return_value=([{"role": "user", "content": "q"}], self.evidence),
            ),
            patch.object(rag, "call_llm_for_answer", side_effect=answers) as generate,
            patch.object(rag, "get_embedder") as embedder,
            patch.object(rag, "_retrieve_candidates") as retrieve,
        ):
            result = asyncio.run(
                rag.ask("Что такое класс?", "chat-1", scope_doc_id="doc-1")
            )
        return result, generate, embedder, retrieve

    def test_short_selected_lecture_skips_retrieval_models(self):
        result, generate, embedder, retrieve = self.ask([{
            "answer": self.valid_answer,
            "model": "deepseek-v4-flash",
        }])

        self.assertEqual(result["grounding_status"], "supported")
        self.assertEqual(generate.call_count, 1)
        embedder.assert_not_called()
        retrieve.assert_not_called()

    def test_failed_first_answer_gets_one_targeted_retry(self):
        invalid_answer = json.dumps({
            "claims": [{
                "text": "Класс описывает все объекты программы.",
                "evidence": [{
                    "source_id": "T9",
                    "quote": "Несуществующая цитата",
                }],
            }],
        }, ensure_ascii=False)
        result, generate, _, _ = self.ask([
            {
                "answer": invalid_answer,
                "model": "deepseek-v4-flash",
            },
            {
                "answer": self.valid_answer,
                "model": "deepseek-v4-flash",
            },
        ])

        self.assertEqual(result["grounding_status"], "supported_after_retry")
        self.assertEqual(generate.call_count, 2)
        retry_messages = generate.call_args_list[1].args[0]
        self.assertIn("rejected_claims", retry_messages[1]["content"])
        self.assertIn("unknown_source", retry_messages[1]["content"])

    def test_provider_error_is_not_reported_as_missing_evidence(self):
        result, generate, _, _ = self.ask([RuntimeError("provider down")])

        self.assertEqual(result["grounding_status"], "provider_error")
        self.assertEqual(result["grounding_reason"], "provider down")
        self.assertEqual(generate.call_count, 1)

    def test_paid_failed_completion_keeps_usage_telemetry(self):
        failure = DeepSeekCompletionError(
            "finish_reason=length",
            {
                "answer": "",
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 3000, "completion_tokens": 4096},
                "estimated_cost_upper_usd": 0.006731,
            },
        )
        result, generate, _, _ = self.ask([failure])

        self.assertEqual(result["grounding_status"], "provider_error")
        self.assertEqual(result["model"], "deepseek-v4-flash")
        self.assertEqual(result["deepseek_usage"]["calls"], 1)
        self.assertEqual(result["deepseek_usage"]["completion_tokens"], 4096)
        self.assertEqual(
            result["deepseek_usage"]["details"][0]["operation"],
            "failed_answer_generation",
        )
        self.assertEqual(generate.call_count, 1)

    def test_paid_failed_validator_keeps_usage_telemetry(self):
        risky_answer = json.dumps({
            "claims": [{
                "text": "Все классы обязательно описывают только один объект.",
                "evidence": [{
                    "source_id": "T1",
                    "quote": "Класс можно представить как описание нового типа объектов.",
                }],
            }],
        }, ensure_ascii=False)
        failure = DeepSeekCompletionError(
            "validator finish_reason=length",
            {
                "answer": "",
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 1200, "completion_tokens": 4096},
                "estimated_cost_upper_usd": 0.005934,
            },
        )

        with patch.object(rag, "_run_semantic_validation", side_effect=failure):
            grounded = rag._ground_generated_answer(
                risky_answer,
                self.evidence,
                "Что такое класс?",
            )

        self.assertEqual(len(grounded["deepseek_calls"]), 1)
        self.assertEqual(
            grounded["deepseek_calls"][0]["operation"],
            "failed_semantic_validation",
        )
        self.assertEqual(
            grounded["deepseek_calls"][0]["completion_tokens"],
            4096,
        )

    def test_invalid_output_after_retry_has_separate_status(self):
        result, generate, _, _ = self.ask([
            {"answer": "not json", "model": "deepseek-v4-flash"},
            {"answer": "still not json", "model": "deepseek-v4-flash"},
        ])

        self.assertEqual(result["grounding_status"], "invalid_output")
        self.assertEqual(result["grounding_reason"], "invalid_model_response")
        self.assertEqual(generate.call_count, 2)

    def test_valid_claim_survives_when_another_claim_is_rejected(self):
        partial_answer = json.dumps({
            "claims": [
                {
                    "text": "Класс можно представить как описание нового типа объектов.",
                    "evidence": [{
                        "source_id": "T1",
                        "quote": "Класс можно представить как описание нового типа объектов.",
                    }],
                },
                {
                    "text": "Все классы создают десять объектов.",
                    "evidence": [{
                        "source_id": "T9",
                        "quote": "Несуществующая цитата",
                    }],
                },
            ],
        }, ensure_ascii=False)
        result, generate, _, _ = self.ask([{
            "answer": partial_answer,
            "model": "deepseek-v4-flash",
        }])

        self.assertEqual(result["grounding_status"], "supported_partial")
        self.assertEqual(result["answer"], self.evidence["T1"]["text"])
        self.assertEqual(generate.call_count, 1)

    def test_short_lecture_loader_requires_both_limits(self):
        chunks = [
            {"id": "c1", "text": "a" * 100, "metadata": {}},
            {"id": "c2", "text": "b" * 100, "metadata": {}},
        ]
        with (
            patch.object(rag, "list_chunks", return_value=chunks),
            patch.object(rag, "SHORT_LECTURE_MAX_CHUNKS", 2),
            patch.object(rag, "SHORT_LECTURE_MAX_CHARS", 200),
        ):
            self.assertEqual(rag._load_short_scoped_transcript("doc-1"), chunks)

        with (
            patch.object(rag, "list_chunks", return_value=chunks),
            patch.object(rag, "SHORT_LECTURE_MAX_CHUNKS", 1),
            patch.object(rag, "SHORT_LECTURE_MAX_CHARS", 200),
        ):
            self.assertIsNone(rag._load_short_scoped_transcript("doc-1"))

        with (
            patch.object(rag, "list_chunks", return_value=chunks),
            patch.object(rag, "SHORT_LECTURE_MAX_CHUNKS", 2),
            patch.object(rag, "SHORT_LECTURE_MAX_CHARS", 199),
        ):
            self.assertIsNone(rag._load_short_scoped_transcript("doc-1"))


if __name__ == "__main__":
    unittest.main()
