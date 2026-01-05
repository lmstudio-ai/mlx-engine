from typing import Annotated, Literal
import argparse
import base64
import time
import os
 
from mlx_engine.generate import load_model, load_draft_model, create_generator, tokenize
from mlx_engine.utils.token import Token
from mlx_engine.utils.kv_cache_quantization import VALID_KV_BITS, VALID_KV_GROUP_SIZE
from transformers import AutoTokenizer, AutoProcessor
from fastapi import FastAPI, Query, Depends
from pydantic import BaseModel, Field

from huggingface_hub import snapshot_download

app = FastAPI()
@app.put("/download/")
async def download(repo_id: str):
    creator, model = repo_id.split('/')
    snapshot_download(
    repo_id=repo_id,
    local_dir=f"./models/{creator}/{model}"
    )

DEFAULT_PROMPT = "Explain the rules of chess in one sentence" 
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMP = 0.8

class InferenceParams(BaseModel):
    """
    Pydantic model representing the query parameters for MLX Engine Inference.
    """
    model: str = Field(
        description="The file system path to the model"
    )
    prompt: str | None = Field(
        default=DEFAULT_PROMPT, 
        description="Message to be processed by the model"
    )
    system: str | None = Field(
        default=DEFAULT_SYSTEM_PROMPT, 
        description="System prompt for the model"
    )
    no_system: bool | None  = Field(
        default=False, 
        alias="no-system",
        description="Disable the system prompt"
    )
    # Use Query for lists to ensure FastAPI hanldes ?images=a&images=b correctly
    images: list[str] | None  = Field(
        default=None, 
        description="Path of the images to process"
    )
    temp: float | None = Field(
        default=DEFAULT_TEMP, 
        description="Sampling temperature"
    )
    stop_strings: list[str] | None = Field(
        default=None, 
        description="Strings that will stop the generation"
    )
    top_logprobs: int | None = Field(
        default=0, 
        description="Number of top logprobs to return"
    )
    max_kv_size: int | None = Field(
        default=None, 
        description="Max context size of the model"
    )
    # Note: Pydantic specific validation (ge/le) can replace 'choices' logic, 
    # or you can use a custom validator if the choices are non-linear.
    kv_bits: int | None = Field(
        default=None, 
        alias="kv-bits",
        ge=3,
        le=8,
        description="Number of bits for KV cache quantization. Must be between 3 and 8"
    )
    kv_group_size: int | None = Field(
        default=None, 
        description="Group size for KV cache quantization"
    )
    quantized_kv_start: int | None = Field(
        default=None, 
        description="When --kv-bits is set, start quantizing the KV cache from this step onwards"
    )
    draft_model: str | None =  Field(
        default=None, 
        description="The file system path to the draft model for speculative decoding"
    )
    num_draft_tokens: int | None = Field(
        default=None, 
        description="Number of tokens to draft when using speculative decoding"
    )
    print_prompt_progress: bool | None = Field(
        default=False, 
        description="Enable printed prompt processing progress callback"
    )
    max_img_size: int | None = Field(
        default=None, 
        description="Downscale images to this side length (px)"
    )


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class GenerationStatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0
        self.num_accepted_draft_tokens: int | None = None

    def add_tokens(self, tokens: list[Token]):
        """Record new tokens and their timing."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

        draft_tokens = sum(1 for token in tokens if token.from_draft)
        if self.num_accepted_draft_tokens is None:
            self.num_accepted_draft_tokens = 0
        self.num_accepted_draft_tokens += draft_tokens

        self.total_tokens += len(tokens)

    def print_stats(self):
        """Print generation statistics."""
        end_time = time.time()
        total_time = end_time - self.start_time
        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token
        tokens_per_second = (
            self.total_tokens / effective_time if effective_time > 0 else float("inf")
        )
        print("\n\nGeneration stats:")
        print(f" - Tokens per second: {tokens_per_second:.2f}")
        if self.num_accepted_draft_tokens is not None:
            print(
                f" - Number of accepted draft tokens: {self.num_accepted_draft_tokens}"
            )
        print(f" - Time to first token: {time_to_first_token:.2f}s")
        print(f" - Total tokens generated: {self.total_tokens}")
        
        print(f" - Total time: {total_time:.2f}s")

    
@app.put("/generate/")
async def generate(generate_query: Annotated[InferenceParams, Depends()]):
    def prompt_progress_callback(percent):
        if generate_query.print_prompt_progress:
            width = 40  # bar width
            filled = int(width * percent / 100)
            bar = "█" * filled + "░" * (width - filled)
            print(f"\rProcessing prompt: |{bar}| ({percent:.1f}%)", end="", flush=True)
            if percent >= 100:
                print()  # new line when done
        return True  # Progress callback must return True to continue
        
    print("Loading model...", end="\n", flush=True)
    print(generate_query.model)
    model_kit = load_model(
        str(generate_query.model),
        max_kv_size=generate_query.max_kv_size,
        trust_remote_code=False,
        kv_bits=generate_query.kv_bits,
        kv_group_size=generate_query.kv_group_size,
        quantized_kv_start=generate_query.quantized_kv_start,
    )
    print("\rModel load complete ✓", end="\n", flush=True)

    # Tokenize the prompt
    prompt = generate_query.prompt

    # Build conversation with optional system prompt
    conversation = []
    if not generate_query.no_system:
        conversation.append({"role": "system", "content": generate_query.system})

    # Handle the prompt according to the input type
    # If images are provided, add them to the prompt
    images_base64 = []
    if len(generate_query.images) == 0:
        tf_tokenizer = AutoProcessor.from_pretrained(generate_query.model)
        images_base64 = [image_to_base64(img_path) for img_path in generate_query.images]
        conversation.append(
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "base64": image_b64}
                        for image_b64 in images_base64
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        )
    else:
        tf_tokenizer = AutoTokenizer.from_pretrained(generate_query.model)
        conversation.append({"role": "user", "content": prompt})
    prompt = tf_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenize(model_kit, prompt)
    
   # Record top logprobs
    logprobs_list = []

    # Initialize generation stats collector
    stats_collector = GenerationStatsCollector()

    # Clamp image size
    max_img_size = (generate_query.max_img_size, generate_query.max_img_size) if generate_query.max_img_size else None

    # Generate the response
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_base64,
        max_image_size=max_img_size,
        stop_strings=generate_query.stop_strings,
        max_tokens=1024,
        top_logprobs=generate_query.top_logprobs,
        prompt_progress_callback=prompt_progress_callback,
        num_draft_tokens=generate_query.num_draft_tokens,
        temp=generate_query.temp,
    )
    result = ""
    for generation_result in generator:
        print(generation_result.text, end="", flush=True)
        result += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)
        logprobs_list.extend(generation_result.top_logprobs)

        if generation_result.stop_condition:
            stats_collector.print_stats()
            print(
                f"\nStopped generation due to: {generation_result.stop_condition.stop_reason}"
            )
            if generation_result.stop_condition.stop_string:
                print(f"Stop string: {generation_result.stop_condition.stop_string}")

    if generate_query.top_logprobs:
        [print(x) for x in logprobs_list]
    return result
