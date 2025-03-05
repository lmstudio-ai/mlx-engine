from outlines.processors.structured import JSONLogitsProcessor
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from mlx_engine.model_kit import ModelKit
import mlx.core as mx
import json


class OutlinesLogitsProcessor:
    def __init__(self, model_kit: ModelKit, json_schema: str, prompt_tokens: mx.array):
        # Sanity check the json schema
        json.loads(json_schema)

        self.logits_processor = JSONLogitsProcessor(
            json_schema,
            OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
        )

        self.prompt_tokens = prompt_tokens

    def __call__(self, tokens: mx.array, logits: mx.array):
        generated_tokens = tokens[len(self.prompt_tokens) :]

        if logits.dtype == mx.bfloat16:
            logits = logits.astype(mx.float32)
        logits_1d = logits.reshape(-1)
        logits_1d = self.logits_processor(generated_tokens, logits_1d)
        logits = logits_1d.reshape(1, -1)

        return logits
