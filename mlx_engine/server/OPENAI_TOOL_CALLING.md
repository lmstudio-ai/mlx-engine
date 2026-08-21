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

## Tool parameter schema validation

If `parameters` is omitted, the server uses an empty object schema. If present, `parameters` must be a JSON Schema object; explicit `null` is rejected. Tool parameter schemas must pass `jsonschema`'s Draft 2020-12 schema-shape validation. Because OpenAI function arguments are JSON objects, a root `type` keyword, when present, must allow `"object"`. The server intentionally checks only this root `type`; schemas that use `$ref`, `$dynamicRef`, `anyOf`, or other composition keywords are left to strict validation.

The server intentionally does not eagerly resolve `$ref` or `$dynamicRef` at request time. This matches the vLLM-style split between request validation and constrained/strict execution: in non-strict auto tool calling, schemas are prompt context; in `strict: true`, `jsonschema` resolves references only when validating a parsed tool call. If strict validation encounters an unresolvable schema reference, the server reports it as a tool-calling validation error instead of leaking a raw `jsonschema`/`referencing` exception.

This avoids maintaining a partial JSON Schema reference walker and avoids falsely rejecting valid schemas where literal instance data contains keys named `$ref`, such as values under `const`, `enum`, `default`, or `examples`.

## Intentionally unsupported for this MVP

- `parallel_tool_calls: true` with active tools. The server rejects the request instead of truncating extra calls.
- Forced tool choice (`tool_choice: "required"` or named function tool choices). Use `"auto"` or `"none"` for now.
- `response_format` with active tools. Use `tool_choice: "none"` if structured output should take precedence over provided tools.
- Assistant prefill with active tools or structured output. Plain no-tool requests still support assistant prefill.
- Strict constrained decoding for tool arguments. `strict: true` is validation only; invalid generated arguments produce a stream error.
- Mixed model-format dialects in one response. The server selects one parser per request from loaded parser metadata, chat-template markers, model type, or finally the first tool-call marker in generated text.

## Parser selection and maintenance

Parser selection is anchored to the renderer used for the request before falling back to broader runtime metadata. For vision requests, a processor chat template wins over separately loaded model-tokenizer metadata because the processor is what renders the prompt. Within each source, known chat-template markers are checked before parser metadata so the parser follows the format the renderer can actually emit.

Source order is:

1. actual renderer chat-template markers;
2. actual renderer `tool_parser_type` / parser function metadata;
3. renderer tokenizer chat-template or parser metadata;
4. separately loaded model-tokenizer chat-template or parser metadata;
5. loaded `model_type`; and
6. first generated tool-call marker when no runtime hint is known.

The server keeps a small local format registry for the supported LSEP formats. That registry owns the parser identity, start/end markers, model aliases, template markers, and optional upstream `mlx-lm`/`mlx-vlm` parser module fallback. Local parsing remains the source of the server contract; upstream parsers are lazy optional fallbacks and their output is normalized into the same OpenAI `tool_calls` shape before strict validation.

## Streaming behavior

When tools are active, the server buffers generated text, up to 1 MiB, until generation finishes. While buffering, it emits SSE comment heartbeats so client disconnects still cancel generation. It then parses the buffer. If a valid tool call is parsed, only the structured tool call is emitted. If no valid tool call is parsed, the buffered text is returned as ordinary assistant content.

If the model emits more than one valid tool call in serial mode, the server returns a stream error instead of silently dropping calls.
