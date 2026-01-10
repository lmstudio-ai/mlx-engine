# MLX Engine

A high-performance API server for running Large Language Models (LLMs) and Vision Language Models (VLMs) on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

> [!NOTE]
> This project is in an early version and is not OpenAI-compatible just yet.

This engine is built with FastAPI and designed to easily deploy local models with advanced features like speculative decoding and KV cache quantization.

## Features

- **Apple Silicon Optimized**: Leverages MLX for efficient inference on Mac.
- **Text & Image Generation**: Supports both text-only LLMs and Vision Language Models (VLMs).
- **Speculative Decoding**: Accelerate generation using a smaller draft model.
- **KV Cache Quantization**: Reduce memory usage and increase throughput with 3-8 bit KV cache quantization.
- **Model Management**: API endpoints to download and convert Hugging Face models directly to MLX format.

## API Reference

### Generate Text/Image
`PUT /generate/`

Generates text based on text or image inputs.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | **Required** | The file system path to the model directory. |
| `prompt` | `str` | *"Explain the rules of chess..."* | Message to be processed by the model. |
| `system` | `str` | *"You are a helpful assistant."* | System prompt. |
| `no-system` | `bool` | `False` | Disable the system prompt. |
| `images` | `list[str]` | `None` | List of file paths to images for VLM inference. |
| `temp` | `float` | `0.8` | Sampling temperature. |
| `stop_strings` | `list[str]` | `None` | Strings that stop generation. |
| `top_logprobs` | `int` | `0` | Number of top logprobs to return. |
| `max_kv_size` | `int` | `None` | Max context size of the model. |
| `kv-bits` | `int` | `None` | Bits for KV quantization (3-8). |
| `kv_group_size` | `int` | `None` | Group size for KV quantization. |
| `quantized_kv_start` | `int` | `None` | Step to start quantizing KV cache. |
| `draft_model` | `str` | `None` | Path to draft model for speculative decoding. |
| `num_draft_tokens` | `int` | `None` | Number of tokens to draft. |
| `print_prompt_progress` | `bool` | `False` | Show prompt processing progress bar. |
| `max_img_size` | `int` | `None` | Downscale images to this, side length in px. |

### Convert Model
`PUT /convert/`

Converts a Hugging Face model to MLX format (quantized to 4-bit by default).

**Parameters:**
- `model_id` (input query): Hugging Face repo ID (e.g., `mistralai/Mistral-7B-v0.1`).
- Saves to: `./models/{creator}/{model}`

### Download Model
`PUT /download/`

Downloads a model from Hugging Face.

**Parameters:**
- `repo_id` (input query): Hugging Face repo ID.
- Saves to: `./models/{creator}/{model}`

## Usage Examples

### Text Generation

```bash
curl -X PUT "http://localhost:8000/generate/?model=./models/mistralai/Mistral-7B-Instruct-v0.2&prompt=Hello"
```

### Image Generation (VLM)

```bash
curl -X 'PUT' \
  'http://localhost:8000/generate/?model=./models/llava-hf/llava-1.5-7b-hf&prompt=What is in this image?' \
  -H 'Content-Type: application/json' \
  -d '{
  "images": ["/path/to/image.jpg"]
}'
```

### Speculative Decoding

```bash
curl -X PUT "http://localhost:8000/generate/?model=./models/mlx-community/Mistral-7B-Instruct-v0.2-4bit&draft_model=./models/mlx-community/Mistral-7B-Instruct-v0.2-4bit-draft&num_draft_tokens=5&prompt=Write a long story"
```

## Acknowledgements

This project is based on the open-source work from [LM Studio](https://lmstudio.ai/). Thank you to the LM Studio team!
