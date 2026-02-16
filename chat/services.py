import json
import logging
import os
from typing import Any
from copy import deepcopy

import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are Axiom, an accuracy-first research assistant. "
    "Primary objective: produce the most accurate answer possible, even if that means being slower or saying you cannot verify. "
    "When the $web_search tool is available, you must call it before answering any prompt that is likely time-sensitive, version-sensitive, or fact-risky. "
    "Mandatory search trigger patterns include: "
    "requests using words like latest/current/today/now/recent; "
    "news or events; prices/market data; laws/regulations/policies; "
    "software/package/model versions and release status; "
    "company/org leadership, product announcements, roadmap claims, or metrics that may change; "
    "and any prompt where a wrong factual claim would be costly. "
    "If retrieval fails or results are insufficient/conflicting, state that clearly and do not guess. "
    "Do not sugar-coat uncertainty. Do not fabricate citations, facts, dates, or numbers. "
    "Respond in English unless the user explicitly asks for another language."
)

WEB_SEARCH_TOOL = {
    "type": "builtin_function",
    "function": {
        "name": "$web_search",
    },
}
MAX_TOOL_CALL_STEPS = 4


def _extract_text_content(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part).strip()
    return ""


def _build_tool_result_messages(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = tool_call.get("id")
        function = tool_call.get("function") or {}
        function_name = function.get("name") or "tool"
        arguments_raw = function.get("arguments") or "{}"
        if not tool_call_id:
            continue
        try:
            arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
        except Exception:
            arguments = {"raw": str(arguments_raw)}

        # Moonshot builtin `$web_search` expects the tool response to echo parsed arguments.
        if function_name != "$web_search":
            tool_result: Any = {
                "status": "unsupported_tool",
                "name": function_name,
                "arguments": arguments,
            }
        else:
            tool_result = arguments
        results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps(tool_result),
            }
        )
    return results

def _moonshot_request(
    messages: list[dict[str, Any]],
    *,
    model_override: str | None = None,
    enable_web_search: bool = False,
) -> str:
    api_key = os.getenv("MOONSHOT_API_KEY")
    model = model_override or os.getenv("MOONSHOT_MODEL")
    base_url = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1")
    timeout = float(os.getenv("MOONSHOT_TIMEOUT", "60"))
    temperature = float(os.getenv("MOONSHOT_TEMPERATURE", "0.2"))

    if not api_key or not model:
        return "Moonshot API is not configured. Please set MOONSHOT_API_KEY and MOONSHOT_MODEL."

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
    }
    if enable_web_search:
        payload["tools"] = [WEB_SEARCH_TOOL]
        payload["tool_choice"] = "auto"
    working_messages = deepcopy(messages)
    for _ in range(MAX_TOOL_CALL_STEPS):
        payload["messages"] = working_messages
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if not response.ok:
            detail = response.text
            try:
                data = response.json()
                detail = json.dumps(data)
            except Exception:
                pass
            raise requests.HTTPError(
                f"Moonshot API error {response.status_code}: {detail}",
                response=response,
            )
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = _extract_text_content(message)
        finish_reason = choice.get("finish_reason")
        has_tool_calls = bool(message.get("tool_calls"))

        if text:
            return text
        if finish_reason != "tool_calls" and not has_tool_calls:
            break

        # For Moonshot builtin tools, continue the reasoning chain by appending
        # the assistant tool-call turn and matching tool_result messages.
        working_messages.append(message)
        tool_result_messages = _build_tool_result_messages(message.get("tool_calls") or [])
        if not tool_result_messages:
            break
        working_messages.extend(tool_result_messages)
    return "I couldn't complete a grounded answer for that request. Please try rephrasing."


def call_moonshot(messages: list[dict[str, Any]]) -> str:
    return _moonshot_request(messages)


def call_moonshot_with_retry(messages: list[dict[str, Any]]) -> str:
    try:
        return _moonshot_request(messages)
    except requests.ReadTimeout:
        logger.warning("Moonshot timeout, retrying once.")
        return _moonshot_request(messages)


def call_moonshot_with_tools(
    messages: list[dict[str, Any]],
    *,
    enable_web_search: bool,
    model_override: str | None = None,
) -> str:
    try:
        return _moonshot_request(
            messages,
            model_override=model_override,
            enable_web_search=enable_web_search,
        )
    except requests.ReadTimeout:
        logger.warning("Moonshot timeout, retrying once.")
        return _moonshot_request(
            messages,
            model_override=model_override,
            enable_web_search=enable_web_search,
        )
