# OpenAI-compatible tool calling MVP

This document describes the current `/v1/chat/completions` tool-calling contract for the mlx-engine server.

## Supported

- Streaming chat completions only (`stream: true`).
- OpenAI `tools` entries with `type: "function"`.
- `tool_choice` omitted or `"auto"` for active tools.
- `tool_choice: "none"` to ignore provided tools and run as a normal chat request.
- Serial tool-calling only: set `parallel_tool_calls: false` when tools are active.
- Model-format tool calls emitted by supported chat templates/parsers:
  - Qwen 3.5 `<tool_call>...</tool_call>`
  - Gemma 4 `<|tool_call>...<tool_call|>`
  - Muse Glimmer `<atem:function_calls>...</atem:function_calls>`
- Text responses when tools are available but the model does not emit a supported tool-call marker.
- Text around a valid tool call is emitted as normal content deltas; the terminal event still uses `finish_reason: "tool_calls"`.
- `strict: true` validates parsed tool-call arguments against the tool parameter schema after generation.

## Intentionally unsupported for this MVP

- `parallel_tool_calls: true` with active tools. The server rejects the request instead of truncating extra calls.
- Forced tool choice (`tool_choice: "required"` or named function tool choices). Use `"auto"` or `"none"` for now.
- `response_format` with active tools. Use `tool_choice: "none"` if structured output should take precedence over provided tools.
- Assistant prefill with active tools or structured output. Plain no-tool requests still support assistant prefill.
- Strict constrained decoding for tool arguments. `strict: true` is validation only; invalid generated arguments produce a stream error.
- Mixed model-format dialects in one response. A response must use only one of the supported Qwen, Gemma, or Muse tool-call formats.

## Streaming behavior

When tools are active, the server streams normal content until it sees a known tool-call start marker. From the first marker onward it buffers output, up to 1 MiB, so the completed tool-call block can be parsed safely. If no valid tool call is parsed, the buffered text is returned as ordinary assistant content.

If the model emits more than one valid tool call in serial mode, the server returns a stream error instead of silently dropping calls.
