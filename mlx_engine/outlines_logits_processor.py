from outlines.processors.structured import JSONLogitsProcessor
from outlines.models.transformers import TransformerTokenizer
from mlx_engine.model_kit import ModelKit
import mlx.core as mx
import json


class OutlinesLogitsProcessor:
    processed_token_count: int = 0

    def __init__(self, model_kit: ModelKit, json_schema: str):
        # Sanity check the json schema
        json.loads(json_schema)

        self.logits_processor = JSONLogitsProcessor(
            json_schema,
            TransformerTokenizer(model_kit.tokenizer._tokenizer),
            whitespace_pattern="",
        )

    def __call__(self, tokens: mx.array, logits: mx.array):
        generated_tokens = (
            tokens[-self.processed_token_count :]
            if self.processed_token_count > 0
            else []
        )

        if logits.dtype == mx.bfloat16:
            logits = logits.astype(mx.float32)
        logits_1d = logits.reshape(-1)
        logits_1d = self.logits_processor(generated_tokens, logits_1d)
        logits = logits_1d.reshape(1, -1)

        self.processed_token_count += 1
        return logits
