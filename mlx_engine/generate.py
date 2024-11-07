from typing import Callable, Iterator, List, NamedTuple, Optional
import json
from pathlib import Path

import mlx_lm

from mlx_engine.model_kit import ModelKit
from mlx_engine.vision.vision_model_kit import VisionModelKit
from mlx_engine.outlines_logits_processor import OutlinesLogitsProcessor
from mlx_engine.stop_processor import StopProcessor, GenerationStopCondition
from mlx_engine.utils.set_seed import set_seed


class GenerationResult(NamedTuple):
    text: str
    tokens: List[int]
    stop_condition: Optional[GenerationStopCondition]


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
    stop_strings: Optional[List[str]],
    generate_args: dict,
) -> Iterator[GenerationResult]:
    set_seed(generate_args.pop("seed", None))

    # Process prompt
    generate_step_input = model_kit.process_prompt(
        prompt_tokens, images_b64, prompt_progress_callback, generate_args
    )

    # Add outlines logits processor if json_schema is provided
    logits_processor = []
    json_schema = generate_args.pop("json_schema", None)
    is_structured_output_request = json_schema is not None
    if is_structured_output_request:
        logits_processor.append(OutlinesLogitsProcessor(model_kit, json_schema))
    generate_args["logits_processor"] = logits_processor

    max_tokens = generate_args.pop("max_tokens")
    tokenizer = model_kit.tokenizer
    detokenizer = model_kit.detokenizer
    detokenizer.reset()
    # keep track of tokens buffered by detokenizer to yield accurate generation results
    token_buffer: List[int] = []

    stop_sequences = [
        tokenize(model_kit, sequence) for sequence in (stop_strings or [])
    ]
    stop_processor = StopProcessor(tokenizer, stop_sequences)
    stop_processor_result = None

    for (token, _), n in zip(
        mlx_lm.utils.generate_step(
            generate_step_input, model_kit.model, **generate_args
        ),
        range(max_tokens),
    ):
        model_kit.record_generated_token(token)
        detokenizer.add_token(token)
        token_buffer.append(token)

        stop_processor_result = stop_processor.process_token(token)

        if stop_processor_result.status == "full_stop":
            break
        # If we currently have generated a partial match with a stop sequence, generate new
        # tokens until we know if the stop sequence is hit or not (i.e., make sure not to yield yet)
        if stop_processor_result.status == "partial_match":
            continue

        # only yield a generation result the detokenizer has a segment to yield
        new_text = detokenizer.last_segment
        if new_text:
            yield GenerationResult(
                text=new_text,
                tokens=token_buffer,
                stop_condition=None,
            )
            token_buffer = []

    # check is there any remaining text to send
    detokenizer.finalize()
    last_segment = detokenizer.last_segment
    last_segment, generation_stop_condition = stop_processor.finalize(
        last_segment, stop_processor_result
    )
    yield GenerationResult(
        text=last_segment,
        tokens=token_buffer,
        stop_condition=generation_stop_condition,
    )


def tokenize(model_kit: ModelKit | VisionModelKit, prompt: str) -> List[int]:
    return model_kit.tokenize(prompt)
