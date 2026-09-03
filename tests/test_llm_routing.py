import unittest
from unittest.mock import Mock, patch

from core import llm


class OpenRouterRoutingTests(unittest.TestCase):
    def response(self, content: str):
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": content}}],
        }
        return response

    def test_notes_use_minimax_then_only_quality_fallbacks(self):
        with (
            patch.object(llm, "OPENROUTER_API_KEY", "test-key"),
            patch.object(
                llm.requests,
                "post",
                side_effect=[self.response(""), self.response("готово")],
            ) as post,
            patch.object(llm.time, "sleep"),
        ):
            result = llm.call_llm([{"role": "user", "content": "lecture"}])

        models = [call.kwargs["json"]["model"] for call in post.call_args_list]
        self.assertEqual(models, [llm.PRIMARY_MODEL, llm.FALLBACK_MODELS[0]])
        self.assertEqual(
            [llm.PRIMARY_MODEL] + llm.FALLBACK_MODELS,
            [
                "minimax/minimax-m3:free",
                "z-ai/glm-5.2:free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
            ],
        )
        self.assertEqual(result["answer"], "готово")

    def test_history_summary_uses_only_utility_pool(self):
        with (
            patch.object(llm, "OPENROUTER_API_KEY", "test-key"),
            patch.object(llm.requests, "post", return_value=self.response("сводка")) as post,
        ):
            llm.call_llm_for_summarization([
                {"role": "user", "content": "dialog"},
            ])

        self.assertEqual(post.call_args.kwargs["json"]["model"], llm.UTILITY_MODELS[0])
        self.assertEqual(
            llm.UTILITY_MODELS,
            [
                "google/gemma-4-31b-it:free",
                "google/gemma-4-26b-a4b-it:free",
                "poolside/laguna-s-2.1:free",
            ],
        )

    def test_title_falls_back_only_inside_utility_pool(self):
        with (
            patch.object(llm, "OPENROUTER_API_KEY", "test-key"),
            patch.object(
                llm.requests,
                "post",
                side_effect=[
                    self.response(""),
                    self.response(""),
                    self.response("Заголовок"),
                ],
            ) as post,
            patch.object(llm.time, "sleep"),
        ):
            result = llm.call_llm_for_title([
                {"role": "user", "content": "dialog"},
            ])

        models = [call.kwargs["json"]["model"] for call in post.call_args_list]
        self.assertEqual(models, llm.UTILITY_MODELS)
        self.assertEqual(result["answer"], "Заголовок")


if __name__ == "__main__":
    unittest.main()
