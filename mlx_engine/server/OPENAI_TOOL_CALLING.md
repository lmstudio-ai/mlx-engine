# OpenAI-compatible tool calling MVP

This document describes the current `/v1/chat/completions` tool-calling contract for the mlx-engine server.

## Supported

- Streaming chat completions only (`stream: true`).
- OpenAI `tools` entries with `type: "function"` and names matching `^[A-Za-z0-9_-]{1,64}$`.
- `tool_choice` omitted or `"auto"` for active tools.
- `tool_choice: "none"` to ignore provided tools and run as a normal chat request.
- Serial tool-calling only: set `parallel_tool_calls: false` when tools are active.
- Model-format tool calls emitted by supported chat templates/parsers:
  - Qwen 3.5 `<tool_call>...</tool_call>`
  - Gemma 4 `<|tool_call>...<tool_call|>`
  - Muse Glimmer `<atem:function_calls>...</atem:function_calls>`
- Text responses when tools are available but no valid tool call is parsed.
- Exclusive tool-call turns: when a valid tool call is parsed, the response emits `tool_calls` with `finish_reason: "tool_calls"` and suppresses surrounding model text.
- `strict: true` validates parsed tool-call arguments against the tool parameter schema after generation.
- Native tool-call delimiter strings are reserved protocol text. In malformed Gemma calls, `<tool_call|>` may be treated as the intended call boundary and used to recover an unterminated string; literal delimiter text inside arguments is not guaranteed to round-trip.

## Intentionally unsupported for this MVP

- `parallel_tool_calls: true` with active tools. The server rejects the request instead of truncating extra calls.
- Forced tool choice (`tool_choice: "required"` or named function tool choices). Use `"auto"` or `"none"` for now.
- `response_format` with active tools. Use `tool_choice: "none"` if structured output should take precedence over provided tools.
- Assistant prefill with active tools or structured output. Plain no-tool requests still support assistant prefill.
- Strict constrained decoding for tool arguments. `strict: true` is validation only; invalid generated arguments produce a stream error.
- Mixed model-format dialects in one response. The server selects the parser from the loaded model type when known, otherwise from the first tool-call marker in the generated text.

## Streaming behavior

When tools are active, the server buffers generated text, up to 1 MiB, until generation finishes. While buffering, it emits SSE comment heartbeats so client disconnects still cancel generation. It then parses the buffer. If a valid tool call is parsed, only the structured tool call is emitted. If no valid tool call is parsed, the buffered text is returned as ordinary assistant content.

If the model emits more than one valid tool call in serial mode, the server returns a stream error instead of silently dropping calls.
