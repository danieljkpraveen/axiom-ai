import asyncio
import json
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are Axiom, a fast research assistant. Prioritize speed and accuracy. "
    "Use provided web context when available. Do not invent facts; acknowledge gaps."
)


def _parse_args(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return shlex.split(value)


def _normalize_search_results(raw: Any) -> List[Dict[str, str]]:
    if raw is None:
        return []
    if hasattr(raw, "structuredContent"):
        structured = getattr(raw, "structuredContent", None)
        if isinstance(structured, dict) and structured.get("result"):
            raw = structured.get("result")
        elif isinstance(structured, str):
            raw = structured
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, dict) and raw.get("structuredContent"):
        raw = raw["structuredContent"].get("result") or raw
    if isinstance(raw, str):
        # Parse simple numbered list format
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        results: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for line in lines:
            if line[0].isdigit() and ". " in line:
                if current.get("url"):
                    results.append(current)
                current = {"title": line.split(". ", 1)[1].strip()}
            elif line.lower().startswith("url:"):
                current["url"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                current["snippet"] = line.split(":", 1)[1].strip()
        if current.get("url"):
            results.append(current)
        if results:
            return results
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        text_parts: List[str] = []
        for item in raw:
            if hasattr(item, "text"):
                text_parts.append(str(getattr(item, "text")))
            elif isinstance(item, dict) and item.get("text"):
                text_parts.append(str(item.get("text")))
        if text_parts:
            combined = "\n".join(text_parts)
            return _normalize_search_results(combined)
    if isinstance(raw, dict) and "content" in raw:
        raw = raw["content"]
    if not isinstance(raw, list):
        return []
    results: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("heading") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        snippet = str(item.get("snippet") or item.get("body") or "").strip()
        if not url:
            continue
        results.append({"title": title or url, "url": url, "snippet": snippet})
    return results


def _normalize_fetch_content(raw: Any) -> str:
    if raw is None:
        return ""
    if hasattr(raw, "content"):
        raw = raw.content
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict) and "content" in raw:
        return str(raw["content"])
    return str(raw)


async def _run_mcp_search(query: str, max_fetch: int = 2) -> Dict[str, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("MCP client unavailable: %s", exc)
        return {"results": [], "fetched": []}

    command = os.getenv("DDG_MCP_COMMAND", "duckduckgo-mcp-server")
    args = _parse_args(os.getenv("DDG_MCP_ARGS", "--stdio"))
    debug = os.getenv("MCP_DEBUG", "").lower() in {"1", "true", "yes"}

    params = StdioServerParameters(command=command, args=args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            search_result = await session.call_tool("search", {"query": query})
            if debug:
                logger.debug("MCP raw search result: %s", search_result)
            results = _normalize_search_results(search_result)

            fetched: List[Dict[str, str]] = []
            for item in results[:max_fetch]:
                url = item.get("url")
                if not url:
                    continue
                fetch_result = await session.call_tool("fetch_content", {"url": url})
                if debug:
                    logger.debug("MCP raw fetch result (%s): %s", url, fetch_result)
                content = _normalize_fetch_content(fetch_result)
                fetched.append({"url": url, "content": content[:1500]})
            return {"results": results, "fetched": fetched}


def mcp_search(query: str, max_fetch: int = 2) -> Dict[str, Any]:
    try:
        return asyncio.run(_run_mcp_search(query, max_fetch=max_fetch))
    except Exception as exc:
        logger.warning("MCP search failed: %s", exc)
        return {"results": [], "fetched": []}


def build_context_block(mcp_payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    results = mcp_payload.get("results") or []
    fetched = mcp_payload.get("fetched") or []
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


def call_moonshot(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("MOONSHOT_API_KEY")
    model = os.getenv("MOONSHOT_MODEL")
    base_url = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1")
    timeout = float(os.getenv("MOONSHOT_TIMEOUT", "60"))

    if not api_key or not model:
        return "Moonshot API is not configured. Please set MOONSHOT_API_KEY and MOONSHOT_MODEL."

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1,
    }
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
    return data["choices"][0]["message"]["content"]


def call_moonshot_with_retry(messages: List[Dict[str, str]]) -> str:
    try:
        return call_moonshot(messages)
    except requests.ReadTimeout:
        logger.warning("Moonshot timeout, retrying once.")
        return call_moonshot(messages)
