# OpenAgent

> Any-model terminal AI coding agent (Apache 2.0).

OpenAgent is a model-agnostic, terminal-based AI coding agent designed to support virtually any LLM provider (OpenAI-compatible, Anthropic, Gemini, Ollama, custom HTTP/JSONPath, etc.) with streaming, tool calling, context management, and session persistence.

## Features

- **Universal Model Support**: Native adapters, 40+ OpenAI-compatible presets, and JSONPath mapping for arbitrary HTTP endpoints.
- **Robust Tool Protocol**: Native schema-based tool calling with fallback to text-protocol tool parsing for models without tool support.
- **Context Management**: Token budgeting, smart compaction, and sliding window boundaries.
- **Safety First**: Tool permission controls (allow/ask/deny), workspace path sandboxing, and secret masking.
- **MCP Integration**: Model Context Protocol client support (stdio and SSE/HTTP).
- **Interactive TUI & CLI**: Rich streaming UI and REPL.

## License

[Apache 2.0](LICENSE)
