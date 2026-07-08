from mlx_engine.tool_runtime import create_gemma4_tool_context_from_prompt


GEMMA4_TOOL_PROMPT = '''<bos><|turn>system
<|tool>declaration:get_weather{description:<|"|>Get weather<|"|>,parameters:{properties:{location:{type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|><|tool>declaration:search{description:<|"|>Search<|"|>,parameters:{properties:{query:{type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
What is the weather in Paris?'''


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, _token_ids):
        return self.text


def test_gemma4_context_extracts_declared_tool_names():
    context = create_gemma4_tool_context_from_prompt(
        tokenizer=_Tokenizer(GEMMA4_TOOL_PROMPT),
        prompt_tokens=[1, 2, 3],
        model_type="gemma4",
    )

    assert context is not None
    assert context.tool_names == ("get_weather", "search")
    assert not context.reasoning_open


def test_gemma4_context_requires_gemma4_model_type():
    context = create_gemma4_tool_context_from_prompt(
        tokenizer=_Tokenizer(GEMMA4_TOOL_PROMPT),
        prompt_tokens=[1, 2, 3],
        model_type="qwen3_5_vl",
    )

    assert context is None


def test_gemma4_context_requires_tool_declarations():
    context = create_gemma4_tool_context_from_prompt(
        tokenizer=_Tokenizer("<|turn>user\nHello<turn|>"),
        prompt_tokens=[1, 2, 3],
        model_type="gemma4",
    )

    assert context is None


def test_gemma4_context_tracks_open_reasoning_from_prompt_tail():
    context = create_gemma4_tool_context_from_prompt(
        tokenizer=_Tokenizer(GEMMA4_TOOL_PROMPT + "<|channel>thought\nNeed a tool."),
        prompt_tokens=[1, 2, 3],
        model_type="gemma4",
    )

    assert context is not None
    assert context.reasoning_open
