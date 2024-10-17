from typing import Callable, Iterator, List, Optional
import random
import json
from pathlib import Path

import mlx.core as mx
import mlx_lm

from mlx_engine.model_kit import ModelKit
from mlx_engine.vision.vision_model_kit import VisionModelKit
from mlx_engine.outlines_logits_processor import OutlinesLogitsProcessor


def load_model(
    model_path: str | Path, max_kv_size: int, trust_remote_code: bool
) -> ModelKit | VisionModelKit:
    model_path = Path(model_path)
    config_json = json.loads((model_path / "config.json").read_text())

    if "vision_config" in config_json:
        return VisionModelKit(model_path, trust_remote_code)
    else:
        return ModelKit(model_path, max_kv_size)


# Adapted from mlx_lm.utils.stream_generate
def create_generator(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    prompt_progress_callback: Optional[Callable[[float], None]],
    images_b64: Optional[List[str]],
    generate_args: dict,
) -> Iterator[str]:
    generate_step_input = model_kit.process_prompt(
        prompt_tokens, images_b64, prompt_progress_callback, generate_args
    )

    # Generate a random seed if not provided or if seed was explicitly set to -1
    seed = generate_args.pop("seed", -1)
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)

    # Add outlines logits processor if json_schema is provided
    logits_processor = []
    json_schema = generate_args.pop("json_schema", None)
    if json_schema is not None:
        logits_processor.append(OutlinesLogitsProcessor(model_kit, json_schema))
    generate_args["logits_processor"] = logits_processor

    max_tokens = generate_args.pop("max_tokens")
    tokenizer = model_kit.tokenizer
    detokenizer = model_kit.detokenizer
    detokenizer.reset()

    for (token, _), n in zip(
        mlx_lm.utils.generate_step(
            generate_step_input, model_kit.model, **generate_args
        ),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        # Yield the last segment if streaming
        yield detokenizer.last_segment

    detokenizer.finalize()
    yield detokenizer.last_segment


def tokenize(model_kit: ModelKit | VisionModelKit, prompt: str) -> List[int]:
    return model_kit.tokenize(prompt)
