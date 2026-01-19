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
from mlx_vlm.convert import convert
from fastapi.responses import StreamingResponse
import json

from mlx_audio.tts.utils import load_model as load_tts_model
#remove this import after finished with audio
import sounddevice as sd


app = FastAPI()

class convertionParams(BaseModel):
    model_id: str = Field(description="The Hugging Face model ID"),
    quantize: bool | None = Field(description="Quantize the model", default=True),
    q_bits: int | None = Field(description="Number of bits for KV cache quantization", default=4),
    q_group_size: int | None = Field(description="Group size for KV cache quantization", default=64),
    upload_repo: str | None = Field(description="Upload the model to Hugging Face", default=None),

@app.put("/download/convert")
async def convertion(params: convertionParams):
    creator, model = params.model_id.split('/')
    convert(
        hf_path=params.model_id,
        mlx_path=f"./models/{creator}/{model}",
        quantize=params.quantize,
        q_bits=params.q_bits,
        q_group_size=params.q_group_size,
        upload_repo=params.upload_repo
    )
    print(f"Model converted and saved to {output_path}")

@app.put("/download/huggingface")
async def download(repo_id: str):
    creator, model = repo_id.split('/')
    snapshot_download(
    repo_id=repo_id,
    local_dir=f"./models/{creator}/{model}"
    )

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMP = 0.8

class Content(BaseModel):
    type: Literal["text", "image_url"] = Field(description="The type of the content")
    text: str | None = Field(description="The text", default=None)
    url: str | None = Field(default=None, description="Image URL object")

class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(description="The role of the message")
    content: list[Content] | str = Field(description="The content of the message")
    name: str | None = Field(description="The name of the message", default=None)

class InferenceParams(BaseModel):
    #Required Parameters
    model: str = Field(
        description="The file system path to the model"
    )
    messages: list[Message] = Field(
        description="Message to be processed by the model"
    )
    #Core Sampling Parameters
    temperature: float | None = Field(
        default=DEFAULT_TEMP, 
        description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    stream: bool | None = Field(
        default=False, 
        description="Enable streaming of the response"
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
    json_schema: str | None = Field(
        default=None, 
        description="JSON schema for the response"
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

@app.get("/v1/models")
async def models():
    models = []
    for model in os.listdir("./models"):
        for file in os.listdir(f"./models/{model}"):
            models.append(model + "/" + file)
    models.sort()
    return models

async def generate_stream(generator):
    for generation_result in generator:
        #print(generation_result.text, end="", flush=True)
        yield json.dumps({"content" : generation_result.text, "ID" : "12345"})

async def generate_output(generator):
    result = ""
    for generation_result in generator:
        result += generation_result.text
    #     stats_collector.add_tokens(generation_result.tokens)
    #     logprobs_list.extend(generation_result.top_logprobs)

    # if generation_result.stop_condition:
    #     stats_collector.print_stats()
    #     print(
    #         f"\nStopped generation due to: {generation_result.stop_condition.stop_reason}"
    #     )
    # if generation_result.stop_condition.stop_string:
    #     print(f"Stop string: {generation_result.stop_condition.stop_string}")

    # if generate_query.top_logprobs:
    #     [print(x) for x in logprobs_list] 

    return result

@app.post("/v1/chat/completions")
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
    tf_tokenizer = AutoProcessor.from_pretrained(generate_query.model)
    images_base64 = []
    for message in generate_query.messages:
        if message.role == "user" and isinstance(message.content, list):
            for content in message.content:
                if  content.url is not None and content.type == "image_url":
                    images_base64.append(image_to_base64(content.url))
                    #content.url = image_to_base64(content.url)
    # Build conversation with optional system prompt
    conversation = generate_query.messages

    tf_tokenizer = AutoTokenizer.from_pretrained(generate_query.model)
    prompt = tf_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenize(model_kit, prompt)
   # Record top logprobs
    logprobs_list = []

    # Initialize generation stats collector
    stats_collector = GenerationStatsCollector()

    # Generate the response
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_base64,
        stop_strings=generate_query.stop_strings,
        max_tokens=1024,
        top_logprobs=generate_query.top_logprobs,
        prompt_progress_callback=prompt_progress_callback,
        num_draft_tokens=generate_query.num_draft_tokens,
        temp=generate_query.temperature 
    )
    if generate_query.stream:
        return StreamingResponse(generate_stream(generator))
    else:
        return await generate_output(generator)

@app.post("/v1/audio")
async def tts(tts_query : str):
    model = load_tts_model("mlx-community/Kokoro-82M-bf16")
    sample_rate = 24000
    print("Playing audio in real-time...")
    output = model.generate(tts_query, voice="af_heart")
    for result in output:
        # Play this chunk immediately
        sd.play(result.audio, sample_rate)
        sd.wait()  # Wait for this chunk to finish playing
    print("Playback complete!")
    return output 