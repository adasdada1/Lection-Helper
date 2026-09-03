import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import core.rag as rag
import main


class LightweightChatTests(unittest.TestCase):
    def test_comparison_question_requires_direct_difference(self):
        guidance = rag._question_guidance(
            "Чем класс отличается от экземпляра объясни мне?"
        )

        self.assertIn("Первый тезис", guidance)
        self.assertIn("главное различие", guidance)
        self.assertIn("Не заменяй", guidance)

    def test_regular_question_has_no_extra_comparison_guidance(self):
        self.assertEqual(rag._question_guidance("Что такое класс?"), "")

    def test_lexical_candidates_use_only_transcripts(self):
        expected = [{"id": "chunk-1"}]
        with patch.object(rag, "lexical_search", return_value=expected) as search:
            result = rag._retrieve_lexical_candidates(
                "класс экземпляр",
                limit=7,
                course_id="course-1",
                scope_doc_id="doc-1",
            )

        self.assertEqual(result, expected)
        search.assert_called_once_with(
            "класс экземпляр",
            top_k=7,
            content_kind="transcript",
            course_id="course-1",
            doc_id="doc-1",
        )

    def test_ask_skips_embedder_and_reranker_in_lightweight_mode(self):
        chat = {"chat_id": "chat-1", "course_id": "course-1", "title": "Чат"}
        with (
            patch.object(rag.chat_memory, "get_chat", return_value=chat),
            patch.object(rag.chat_memory, "add_message"),
            patch.object(rag.chat_memory, "maybe_summarize"),
            patch.object(rag.chat_memory, "get_message_count", return_value=1),
            patch.object(rag, "_build_messages", return_value=([{"role": "user", "content": "q"}], {})),
            patch.object(rag, "_retrieve_lexical_candidates", return_value=[]) as lexical,
            patch.object(rag, "_retrieve_candidates") as dense,
            patch.object(rag, "get_embedder") as embedder,
        ):
            result = asyncio.run(
                rag.ask("вопрос", "chat-1", use_local_models=False)
            )

        self.assertEqual(result["chat_id"], "chat-1")
        lexical.assert_called_once()
        dense.assert_not_called()
        embedder.assert_not_called()

    def test_chat_endpoint_uses_lightweight_mode_for_active_job(self):
        original_jobs = main.job_manager.jobs
        main.job_manager.jobs = {
            "job-1": {"status": "processing", "filename": "lecture.mp3"},
        }
        response = {
            "answer": "Ответ",
            "model": "model",
            "chat_id": "chat-1",
            "sources": [],
        }
        try:
            with (
                patch("core.chat_memory.get_chat", return_value={"title": "Чат"}),
                patch("core.rag.ask", new=AsyncMock(return_value=response)) as ask,
            ):
                result = asyncio.run(
                    main.chat(main.ChatRequest(message="Вопрос", chat_id="chat-1"))
                )
        finally:
            main.job_manager.jobs = original_jobs

        self.assertEqual(result["answer"], "Ответ")
        ask.assert_awaited_once_with(
            "Вопрос",
            chat_id="chat-1",
            scope_doc_id=None,
            use_local_models=False,
        )


if __name__ == "__main__":
    unittest.main()
