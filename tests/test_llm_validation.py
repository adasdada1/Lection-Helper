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

    def json_response(
        self,
        content='{"items": []}',
        finish_reason="stop",
        completion_tokens=100,
    ):
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "model": "deepseek-v4-flash",
            "choices": [{
                "finish_reason": finish_reason,
                "message": {"content": content},
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": completion_tokens,
            },
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

    def test_semantic_answer_accepts_adaptive_low_reasoning(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                return_value=self.json_response(),
            ) as post,
        ):
            llm.call_llm_for_semantic_answer(
                [{"role": "user", "content": "{}"}],
                self.schema,
                "low",
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertIn(
            "semantic_lecture_answer",
            payload["messages"][0]["content"],
        )

    def test_semantic_repair_disables_thinking(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                return_value=self.json_response(),
            ) as post,
        ):
            llm.call_llm_for_semantic_repair(
                [{"role": "user", "content": "{}"}],
                self.schema,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["max_tokens"], llm.DEEPSEEK_REPAIR_MAX_TOKENS)

    def test_semantic_validator_falls_back_without_thinking_after_length(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                side_effect=[
                    self.json_response(
                        "",
                        finish_reason="length",
                        completion_tokens=8191,
                    ),
                    self.json_response('{"results": []}'),
                ],
            ) as post,
        ):
            result = llm.call_llm_for_semantic_validation(
                [{"role": "user", "content": "{}"}],
                self.schema,
            )

        fallback_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(fallback_payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", fallback_payload)
        self.assertEqual(
            fallback_payload["max_tokens"],
            llm.DEEPSEEK_REPAIR_MAX_TOKENS,
        )
        self.assertEqual(result["generation_mode"], "no_thinking_after_length")
        self.assertEqual(
            result["prior_results"][0]["usage"]["completion_tokens"],
            8191,
        )

    def test_answer_falls_back_without_thinking_after_length(self):
        with (
            patch.object(deepseek_client, "DEEPSEEK_API_KEY", "test-key"),
            patch.object(
                deepseek_client.requests,
                "post",
                side_effect=[
                    self.json_response(
                        "",
                        finish_reason="length",
                        completion_tokens=4096,
                    ),
                    self.json_response(),
                ],
            ) as post,
        ):
            result = llm.call_llm_for_answer(
                [{"role": "user", "content": "{}"}],
                self.schema,
            )

        first_payload = post.call_args_list[0].kwargs["json"]
        fallback_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["thinking"], {"type": "enabled"})
        self.assertEqual(
            first_payload["max_tokens"],
            llm.DEEPSEEK_ANSWER_MAX_TOKENS,
        )
        self.assertEqual(fallback_payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", fallback_payload)
        self.assertEqual(
            fallback_payload["max_tokens"],
            llm.DEEPSEEK_ANSWER_FALLBACK_MAX_TOKENS,
        )
        self.assertEqual(result["generation_mode"], "no_thinking_after_length")
        self.assertEqual(
            result["prior_results"][0]["usage"]["completion_tokens"],
            4096,
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
