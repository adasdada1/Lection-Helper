import unittest
from unittest.mock import Mock, patch

from core import deepseek_client, llm


class ValidationLlmTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
            "additionalProperties": False,
        }

    def json_response(self, content='{"items": []}'):
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "model": "deepseek-v4-flash",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": content},
            }],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        }
        return response

    def test_answer_uses_direct_deepseek_thinking_json_output(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                return_value=self.json_response(),
            ) as post,
        ):
            result = llm.call_llm_for_answer(
                [{"role": "user", "content": "{}"}],
                self.schema,
            )

        self.assertEqual(post.call_args.args[0], deepseek_client.DEEPSEEK_API_URL)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], deepseek_client.DEEPSEEK_MODEL)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(
            payload["reasoning_effort"],
            llm.DEEPSEEK_REASONING_EFFORT,
        )
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", payload)
        self.assertIn("grounded_lecture_answer", payload["messages"][0]["content"])
        self.assertIn('"additionalProperties": false', payload["messages"][0]["content"])
        self.assertEqual(result["answer"], '{"items": []}')
        self.assertAlmostEqual(result["estimated_cost_upper_usd"], 0.000572)

    def test_validation_uses_smaller_output_limit(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                return_value=self.json_response(),
            ) as post,
        ):
            llm.call_llm_for_validation(
                [{"role": "user", "content": "{}"}],
                self.schema,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], llm.DEEPSEEK_VALIDATION_MAX_TOKENS)
        self.assertEqual(
            payload["reasoning_effort"],
            llm.DEEPSEEK_VALIDATION_REASONING_EFFORT,
        )

    def test_invalid_json_output_is_rejected(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                return_value=self.json_response("not json"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "невалидный JSON") as raised:
                llm.call_llm_for_answer(
                    [{"role": "user", "content": "{}"}],
                    self.schema,
                )
        self.assertEqual(raised.exception.result["usage"]["prompt_tokens"], 1000)
        self.assertAlmostEqual(
            raised.exception.result["estimated_cost_upper_usd"],
            0.000572,
        )


if __name__ == "__main__":
    unittest.main()
