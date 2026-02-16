from io import BytesIO
from unittest.mock import Mock, patch

import requests
from PIL import Image
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from . import services
from .models import ChatMessage


class ChatSendViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="pw12345pass")
        self.client.force_login(self.user)
        self.url = reverse("chat_send")

    def test_text_prompt_uses_moonshot_tool_calling(self):
        with patch("chat.views.call_moonshot_with_tools", return_value="Tool-backed answer") as mock_tools:
            response = self.client.post(self.url, {"message": "Explain MCP vs API tools"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["assistant_message"], "Tool-backed answer")
        self.assertIn("session_id", payload)
        self.assertEqual(payload["sources"], [])

        self.assertEqual(ChatMessage.objects.filter(role="user").count(), 1)
        self.assertEqual(ChatMessage.objects.filter(role="assistant").count(), 1)
        mock_tools.assert_called_once()
        self.assertTrue(mock_tools.call_args.kwargs["enable_web_search"])
        self.assertEqual(mock_tools.call_args.kwargs["model_override"], "moonshot-v1-auto")

    def test_search_model_override_is_respected(self):
        with patch.dict("os.environ", {"MOONSHOT_ENABLE_WEB_SEARCH": "true", "MOONSHOT_SEARCH_MODEL": "moonshot-v1-auto"}):
            with patch("chat.views.call_moonshot_with_tools", return_value="ok") as mock_tools:
                response = self.client.post(self.url, {"message": "What changed in Python?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_tools.call_args.kwargs["model_override"], "moonshot-v1-auto")
        self.assertTrue(mock_tools.call_args.kwargs["enable_web_search"])

    def test_web_search_can_be_disabled(self):
        with patch.dict("os.environ", {"MOONSHOT_ENABLE_WEB_SEARCH": "false"}):
            with patch("chat.views.call_moonshot_with_tools", return_value="ok") as mock_tools:
                response = self.client.post(self.url, {"message": "Write a haiku"})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(mock_tools.call_args.kwargs["enable_web_search"])
        self.assertIsNone(mock_tools.call_args.kwargs["model_override"])

    def test_model_failure_returns_graceful_error(self):
        with patch("chat.views.call_moonshot_with_tools", side_effect=RuntimeError("boom")):
            response = self.client.post(self.url, {"message": "Hard query"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("slow or unavailable", payload["assistant_message"])

    def test_image_prompt_uses_retry_path_without_tool_calling(self):
        image = Image.new("RGB", (16, 16), (120, 30, 200))
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_file = SimpleUploadedFile("sample.png", image_bytes.getvalue(), content_type="image/png")

        with patch("chat.views.call_moonshot_with_retry", side_effect=["Image: purple square", "Image answer"]) as mock_retry:
            with patch("chat.views.call_moonshot_with_tools") as mock_tools:
                response = self.client.post(self.url, {"message": "What is in this image?", "image": image_file})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assistant_message"], "Image answer")
        self.assertEqual(mock_retry.call_count, 2)
        mock_tools.assert_not_called()


class ServicesTests(TestCase):
    def test_services_no_longer_exposes_mcp_helpers(self):
        self.assertFalse(hasattr(services, "mcp_search"))
        self.assertFalse(hasattr(services, "build_context_block"))

    def test_call_moonshot_with_tools_sends_web_search_payload(self):
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"choices": [{"message": {"content": "done"}}]}

        with patch.dict(
            "os.environ",
            {"MOONSHOT_API_KEY": "k", "MOONSHOT_MODEL": "kimi-k2.5", "MOONSHOT_API_BASE": "https://api.moonshot.ai/v1"},
        ):
            with patch("chat.services.requests.post", return_value=mock_response) as mock_post:
                result = services.call_moonshot_with_tools(
                    [{"role": "user", "content": "latest news"}],
                    enable_web_search=True,
                    model_override="moonshot-v1-auto",
                )

        self.assertEqual(result, "done")
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "moonshot-v1-auto")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["function"]["name"], "$web_search")

    def test_call_moonshot_with_tools_retries_on_timeout(self):
        ok_response = Mock()
        ok_response.ok = True
        ok_response.json.return_value = {"choices": [{"message": {"content": "retry success"}}]}

        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "k", "MOONSHOT_MODEL": "kimi-k2.5"}):
            with patch("chat.services.requests.post", side_effect=[requests.ReadTimeout(), ok_response]) as mock_post:
                result = services.call_moonshot_with_tools(
                    [{"role": "user", "content": "search me"}],
                    enable_web_search=True,
                )

        self.assertEqual(result, "retry success")
        self.assertEqual(mock_post.call_count, 2)

    def test_call_moonshot_with_tools_continues_after_tool_calls(self):
        first = Mock()
        first.ok = True
        first.json.return_value = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "$web_search",
                                    "arguments": "{\"query\":\"latest python version\"}",
                                },
                            }
                        ],
                    },
                }
            ]
        }
        second = Mock()
        second.ok = True
        second.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Python latest is ...",
                    },
                }
            ]
        }

        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "k", "MOONSHOT_MODEL": "kimi-k2.5"}):
            with patch("chat.services.requests.post", side_effect=[first, second]) as mock_post:
                result = services.call_moonshot_with_tools(
                    [{"role": "user", "content": "latest python version"}],
                    enable_web_search=True,
                )

        self.assertEqual(result, "Python latest is ...")
        self.assertEqual(mock_post.call_count, 2)
        second_payload = mock_post.call_args_list[1].kwargs["json"]
        tool_messages = [m for m in second_payload["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertEqual(tool_messages[0]["name"], "$web_search")
        self.assertEqual(tool_messages[0]["content"], "{\"query\": \"latest python version\"}")
