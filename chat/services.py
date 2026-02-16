import json
import logging
import os
from typing import Any, Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are Axiom, a fast research assistant. Prioritize speed and accuracy. "
    "Use provided web context when available. Do not invent facts; acknowledge gaps."
)


def _normalize_search_results(raw: Any) -> List[Dict[str, str]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = raw.get("results") or raw.get("sources") or raw.get("items") or []
    if not isinstance(raw, list):
        return []

    results: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        snippet = str(item.get("snippet") or item.get("summary") or item.get("excerpt") or "").strip()
        if not url:
            continue
        results.append({"title": title or url, "url": url, "snippet": snippet})
    return results


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    # direct parse first
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # fenced block fallback
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    # first object-looking span fallback
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _moonshot_base_config() -> Tuple[str, Dict[str, str], str, float]:
    api_key = os.getenv("MOONSHOT_API_KEY")
    model = os.getenv("MOONSHOT_MODEL")
    base_url = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1")
    timeout = float(os.getenv("MOONSHOT_TIMEOUT", "60"))

    if not api_key or not model:
        raise ValueError("Moonshot API is not configured. Please set MOONSHOT_API_KEY and MOONSHOT_MODEL.")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return url, headers, model, timeout


def _request_moonshot_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    url, headers, _, timeout = _moonshot_base_config()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not response.ok:
        detail = response.text
        try:
            detail = json.dumps(response.json())
        except Exception:
            pass
        raise requests.HTTPError(f"Moonshot API error {response.status_code}: {detail}", response=response)
    return response.json()


def moonshot_web_search(query: str, max_fetch: int = 2) -> Dict[str, Any]:
    """Use Moonshot built-in web search tool to gather context-like sources."""
    _, _, model, _ = _moonshot_base_config()
    prompt = (
        "Use your built-in web search capability to research the user query. "
        "Return only valid JSON with this shape: "
        '{"results":[{"title":"...","url":"...","snippet":"..."}],'
        '"fetched":[{"url":"...","content":"..."}]}. '
        f"Limit fetched to at most {max_fetch} items and keep content short."
    )

    tool_variants = [
        [{"type": "web_search"}],
        [{"type": "builtin_function", "function": {"name": "$web_search"}}],
        [{"type": "builtin_function", "function": {"name": "web_search"}}],
    ]

    messages = [
        {"role": "system", "content": "You are a web research helper."},
        {"role": "user", "content": f"{prompt}\n\nQuery: {query}"},
    ]

    last_error: Exception | None = None
    for tools in tool_variants:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            data = _request_moonshot_chat(payload)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = _extract_json_object(content)
            results = _normalize_search_results(parsed.get("results") or [])
            fetched_raw = parsed.get("fetched") if isinstance(parsed.get("fetched"), list) else []
            fetched: List[Dict[str, str]] = []
            for item in fetched_raw[:max_fetch]:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                content_text = str(item.get("content") or item.get("snippet") or "").strip()
                if url and content_text:
                    fetched.append({"url": url, "content": content_text[:1500]})
            return {"results": results, "fetched": fetched}
        except Exception as exc:
            last_error = exc
            logger.warning("Moonshot web search attempt failed with tools=%s: %s", tools, exc)

    if last_error:
        logger.warning("Moonshot web search failed: %s", last_error)
    return {"results": [], "fetched": []}


def build_context_block(search_payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    results = search_payload.get("results") or []
    fetched = search_payload.get("fetched") or []
    lines = ["Web search context:"]
    for item in results[:6]:
        lines.append(f"- {item.get('title')}: {item.get('url')}")
        if item.get("snippet"):
            lines.append(f"  snippet: {item.get('snippet')}")
    if fetched:
        lines.append("Fetched content excerpts:")
        for item in fetched:
            lines.append(f"- {item.get('url')}: {item.get('content')}")
    return "\n".join(lines), results[:6]


def call_moonshot(messages: List[Dict[str, Any]]) -> str:
    try:
        _, _, model, _ = _moonshot_base_config()
    except ValueError:
        return "Moonshot API is not configured. Please set MOONSHOT_API_KEY and MOONSHOT_MODEL."

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1,
    }
    data = _request_moonshot_chat(payload)
    return data["choices"][0]["message"]["content"]


def call_moonshot_with_retry(messages: List[Dict[str, Any]]) -> str:
    try:
        return call_moonshot(messages)
    except requests.ReadTimeout:
        logger.warning("Moonshot timeout, retrying once.")
        return call_moonshot(messages)
