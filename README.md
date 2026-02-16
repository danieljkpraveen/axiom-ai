# axiom-ai
Reflecting a starting point for truth and research.

## Configuration
Set these environment variables in `.env`:

- `MOONSHOT_API_KEY`
- `MOONSHOT_MODEL` (default chat/vision model)
- `MOONSHOT_SEARCH_MODEL` (recommended: `moonshot-v1-auto`)
- `MOONSHOT_ENABLE_WEB_SEARCH` (`true` or `false`)
- `MOONSHOT_KNOWLEDGE_CUTOFF` (optional, e.g. `2025-01`, used to bias mandatory search)
- `MOONSHOT_API_BASE` (default: `https://api.moonshot.ai/v1`)
- `MOONSHOT_TIMEOUT` (seconds)
- `MOONSHOT_TEMPERATURE` (optional, default `0.2` for factual stability)

## Web Search
Axiom uses Moonshot's built-in `$web_search` tool for text prompts.
No DuckDuckGo MCP server is required.
