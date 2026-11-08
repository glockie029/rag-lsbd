import json
import os
import unittest
from unittest.mock import patch

import httpx

from chat_api import complete_chat


class ChatApiTests(unittest.TestCase):
    def request(self, env, *, status=200, data=None):
        seen = {}
        original_client = httpx.Client

        def handler(request):
            seen["url"] = str(request.url)
            seen["authorization"] = request.headers.get("Authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(status, json=data)

        def make_client(**kwargs):
            seen["trust_env"] = kwargs["trust_env"]
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        payload = {"messages": [{"role": "user", "content": "检查接口"}], "stream": False, "max_tokens": 64, "temperature": 0.0}
        with patch.dict(os.environ, env, clear=True), patch("chat_api.httpx.Client", side_effect=make_client):
            answer = complete_chat(payload)
        self.assertNotIn("model", payload)  # Caller data is not mutated.
        return answer, seen

    def test_remote_protocol_and_auth(self):
        answer, seen = self.request(
            {"OPENAI_BASE_URL": "https://compatible.example/v1/", "OPENAI_MODEL": "example-model", "OPENAI_API_KEY": "test-only-value"},
            data={"choices": [{"message": {"content": "连接成功"}}]},
        )
        self.assertEqual(answer, "连接成功")
        self.assertEqual(seen["url"], "https://compatible.example/v1/chat/completions")
        self.assertEqual(seen["authorization"], "Bearer test-only-value")
        self.assertTrue(seen["trust_env"])
        self.assertEqual(seen["body"]["model"], "example-model")
        self.assertEqual(seen["body"]["messages"][0]["content"], "检查接口")
        self.assertNotIn("chat_template_kwargs", seen["body"])

    def test_local_no_auth_no_proxy(self):
        answer, seen = self.request(
            {"OPENAI_BASE_URL": "http://127.0.0.1:8081/v1", "OPENAI_MODEL": "local-example"},
            data={"choices": [{"message": {"content": "local response"}}]},
        )
        self.assertEqual(answer, "local response")
        self.assertIsNone(seen["authorization"])
        self.assertFalse(seen["trust_env"])

    def test_missing_model(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "OPENAI_MODEL"):
            complete_chat({"messages": []})

    def test_missing_official_api_key(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "example"}, clear=True), self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            complete_chat({"messages": []})

    def test_http_errors(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.request({"OPENAI_MODEL": "example", "OPENAI_API_KEY": "test-only-value"}, status=401, data={"error": {"message": "Unauthorized"}})

    def test_malformed_response(self):
        with self.assertRaisesRegex(ValueError, "choices"):
            self.request({"OPENAI_MODEL": "example", "OPENAI_API_KEY": "test-only-value"}, data={"choices": []})

    def test_non_text_response(self):
        with self.assertRaisesRegex(ValueError, "text response"):
            self.request({"OPENAI_MODEL": "example", "OPENAI_API_KEY": "test-only-value"}, data={"choices": [{"message": {"content": None}}]})


if __name__ == "__main__":
    unittest.main()
