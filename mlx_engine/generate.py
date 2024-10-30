from typing import Callable, Iterator, List, NamedTuple, Optional
import random
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx_lm

from mlx_engine.model_kit import ModelKit
from mlx_engine.vision.vision_model_kit import VisionModelKit
from mlx_engine.outlines_logits_processor import OutlinesLogitsProcessor
from mlx_engine.stop_processor import StopProcessor, StopReason


class GenerationStopCondition(NamedTuple):
    stop_reason: StopReason
    stop_string: str
    stop_tokens: List[int]

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
        stop_sequences: List[List[int]],
        generate_args: dict,
) -> Iterator[GenerationResult]:
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
    is_structured_output_request = json_schema is not None
    if is_structured_output_request:
        logits_processor.append(OutlinesLogitsProcessor(model_kit, json_schema))
    generate_args["logits_processor"] = logits_processor

    max_tokens = generate_args.pop("max_tokens")
    tokenizer = model_kit.tokenizer
    detokenizer = model_kit.detokenizer
    detokenizer.reset()

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
        
        stop_processor_result = stop_processor.process_token(token)

        if stop_processor_result.status == "full_stop":
            sys.stderr.write(f"[mlx-engine] Full stop criteria match found: {stop_processor_result.stop_reason}\n")
            break
        # If we currently have generated a partial match with a stop sequence, generate new
        # tokens until we know if the stop sequence is hit or not (i.e., make sure not to yield yet)
        if stop_processor_result.status == "partial_match":
            sys.stderr.write("[mlx-engine] Partial stop criteria match found, buffering output\n")
            continue

        new_text = detokenizer.last_segment
        if new_text:
            yield GenerationResult(
                text=new_text,
                tokens=tokenize(model_kit, new_text),
                stop_condition=None,
            )

    # check is there any remaining text to send
    detokenizer.finalize()
    last_segment = detokenizer.last_segment
    if last_segment:
        if stop_processor.stop_sequence_suffix is not None:
            last_segment = last_segment[: -len(stop_processor.stop_sequence_suffix)]

    # build up the final generation stop condition safely, although stop_processor_result
    # should always be set at this point
    generation_stop_condition = None
    if stop_processor_result and stop_processor_result.stop_reason:
        stop_tokens = stop_processor_result.stop_tokens or []
        stop_string = tokenizer.decode(stop_tokens) if stop_tokens else ""
        generation_stop_condition = GenerationStopCondition(
            stop_reason=stop_processor_result.stop_reason,
            stop_string=stop_string,
            stop_tokens=stop_tokens,
        )
    yield GenerationResult(
        text=last_segment,
        tokens=tokenize(model_kit, last_segment),
        stop_condition=generation_stop_condition,
    )


def tokenize(model_kit: ModelKit | VisionModelKit, prompt: str) -> List[int]:
    return model_kit.tokenize(prompt)
