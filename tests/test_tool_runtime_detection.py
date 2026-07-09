from mlx_engine.tool_runtime import (
    create_gemma4_tool_context_from_prompt,
    create_qwen35_tool_context_from_prompt,
)


GEMMA4_TOOL_PROMPT = """<bos><|turn>system
<|tool>declaration:get_weather{description:<|"|>Get weather<|"|>,parameters:{properties:{location:{type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|><|tool>declaration:search{description:<|"|>Search<|"|>,parameters:{properties:{query:{type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
What is the weather in Paris?"""

QWEN35_TOOL_PROMPT = """<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type":"function","function":{"name":"get_weather","parameters":{"properties":{"location":{"type":"string"}}}}}
{"type":"function","function":{"name":"search","parameters":{"properties":{"query":{"type":"string"}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
What is the weather in Paris?"""


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"

    def decode(self, _token_ids):
        return self.text


class _TokenizerWithoutQwenTools(_Tokenizer):
    def __init__(self, text: str):
        super().__init__(text)
        self.tool_call_start = None
        self.tool_call_end = None


def test_gemma4_context_extracts_declared_tool_names():
    context = create_gemma4_tool_context_from_prompt(
        tokenizer=_Tokenizer(GEMMA4_TOOL_PROMPT),
        prompt_tokens=[1, 2, 3],
        model_type="gemma4",
    )

    assert context is not None
    assert context.tool_names == ("get_weather", "search")
    assert not context.reasoning_open


def test_gemma4_context_extracts_colon_namespaced_tool_name():
    context = create_gemma4_tool_context_from_prompt(
        tokenizer=_Tokenizer("<|tool>declaration:mcp:search{}<tool|>"),
        prompt_tokens=[1, 2, 3],
        model_type="gemma4",
    )

    assert context is not None
    assert context.tool_names == ("mcp:search",)

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



def test_qwen35_context_extracts_declared_tool_names():
    context = create_qwen35_tool_context_from_prompt(
        tokenizer=_Tokenizer(QWEN35_TOOL_PROMPT),
        prompt_tokens=[1, 2, 3],
        model_type="qwen3_5_vl",
    )

    assert context is not None
    assert context.tool_names == ("get_weather", "search")
    assert not context.reasoning_open



def test_qwen35_context_requires_qwen35_model_type():
    context = create_qwen35_tool_context_from_prompt(
        tokenizer=_Tokenizer(QWEN35_TOOL_PROMPT),
        prompt_tokens=[1, 2, 3],
        model_type="qwen2_5_vl",
    )

    assert context is None



def test_qwen35_context_requires_native_tool_markers():
    context = create_qwen35_tool_context_from_prompt(
        tokenizer=_TokenizerWithoutQwenTools(QWEN35_TOOL_PROMPT),
        prompt_tokens=[1, 2, 3],
        model_type="qwen3_5_vl",
    )

    assert context is None



def test_qwen35_context_requires_tool_declarations():
    context = create_qwen35_tool_context_from_prompt(
        tokenizer=_Tokenizer("<tool_call>\n<function=example>\n</function>\n</tool_call>"),
        prompt_tokens=[1, 2, 3],
        model_type="qwen3_5_vl",
    )

    assert context is None



def test_qwen35_context_tracks_open_reasoning_from_prompt_tail():
    context = create_qwen35_tool_context_from_prompt(
        tokenizer=_Tokenizer(QWEN35_TOOL_PROMPT + "<think>\nNeed a tool."),
        prompt_tokens=[1, 2, 3],
        model_type="qwen3_5_vl",
    )

    assert context is not None
    assert context.reasoning_open
