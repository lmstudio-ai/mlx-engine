<p align="center">
  <picture> 
    <img alt="lmstudio + MLX" src="https://github.com/user-attachments/assets/128bf3ba-d8d6-4fc8-85c9-4d0113ba5499">
  </picture>
</p>

<p align="center"><bold><code>mlx-engine</code> - <a href="https://github.com/ml-explore/mlx">Apple MLX</a> LLM Engine for <a href="https://lmstudio.ai/">LM Studio</a></bold></p>
<br/>
<p align="center"><a href="https://discord.gg/aPQfnNkxGC"><img alt="Discord" src="https://img.shields.io/discord/1110598183144399058?logo=discord&style=flat&logoColor=white"></a></p>

# mlx-engine
MLX engine for LM Studio

<br/>

## Built with
- [mlx-lm](https://github.com/ml-explore/mlx-examples) - Apple MLX inference engine (MIT)
- [Outlines](https://github.com/dottxt-ai/outlines) - Structured output for LLMs (Apache 2.0)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision model inferencing for MLX (MIT)

<br/>

## How to use in LM Studio
LM Studio 0.3.4 and newer for Mac ships pre-bundled with mlx-engine.
Download LM Studio from [here](https://lmstudio.ai/download?os=mac)

<br/>

## Standalone Demo

### Prerequisites

- macOS 14.0 (Sonoma) or greater.
- python3.11
  - The requirements.txt file is compiled specifically for python3.11. python3.11 is the python version bundled within the LM Studio MLX runtime
  - `brew install python@3.11` is a quick way to add python3.11 to your path that doesn't break your default python setup

### Install Steps
To run a demo of model load and inference:
1. Clone the repository
```
git clone https://github.com/lmstudio-ai/mlx-engine.git
cd mlx-engine
```
2. Create a virtual environment (optional)
```
 python3.11 -m venv .venv
 source .venv/bin/activate
```
3. Install the required dependency packages
```
pip install -U -r requirements.txt
```

### Text Model Demo
Download models with the `lms` CLI tool. The `lms` CLI documentation can be found here: https://lmstudio.ai/docs/cli
Run the `demo.py` script with an MLX text generation model:
```bash
lms get mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
python demo.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit 
```
[mlx-community/Meta-Llama-3.1-8B-Instruct-4bit](https://model.lmstudio.ai/download/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit) - 4.53 GB

This command will use a default prompt. For a different prompt, add a custom `--prompt` argument like:
```bash
lms get mlx-community/Mistral-Small-Instruct-2409-4bit
python demo.py --model mlx-community/Mistral-Small-Instruct-2409-4bit --prompt "How long will it take for an apple to fall from a 10m tree?"
```
[mlx-community/Mistral-Small-Instruct-2409-4bit](https://model.lmstudio.ai/download/mlx-community/Mistral-Small-Instruct-2409-4bit) - 12.52 GB

### Vision Model Demo
Run the `demo.py` script with an MLX vision model:
```bash
lms get mlx-community/pixtral-12b-4bit
python demo.py --model mlx-community/pixtral-12b-4bit --prompt "Compare these images" --images demo-data/chameleon.webp demo-data/toucan.jpeg
```
Currently supported vision models include:
 - [Llama-3.2-Vision](https://model.lmstudio.ai/download/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit)
   - `lms get mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`
 - [Pixtral](https://model.lmstudio.ai/download/mlx-community/pixtral-12b-4bit)
   - `lms get mlx-community/pixtral-12b-4bit`
 - [Qwen2-VL](https://model.lmstudio.ai/download/mlx-community/Qwen2-VL-7B-Instruct-4bit)
   - `lms get mlx-community/Qwen2-VL-7B-Instruct-4bit`
 - [Llava-v1.6](https://model.lmstudio.ai/download/mlx-community/llava-v1.6-mistral-7b-4bit)
   - `lms get mlx-community/llava-v1.6-mistral-7b-4bit`

### Speculative Decoding Demo
Run the `demo.py` script with an MLX text generation model and a compatible `--draft-model`
```bash
lms get mlx-community/Qwen2.5-7B-Instruct-4bit
lms get lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit
python demo.py \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --draft-model lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit \
    --prompt "<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a quick sort algorithm in C++<|im_end|>
<|im_start|>assistant
"
```

## Development Setup

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality. Before contributing, please:

1. Install pre-commit:
   ```bash
   pip install pre-commit && pre-commit install
    ```
2. Run pre-commit:
   ```bash
   pre-commit run --all-files
   ```
3. Fix any issues before submitting your PR

## Testing

To run tests, run the following from the root of this repo:
```bash
python -m pip install pytest
python -m pytest tests/
```

To test specific vision models:
```bash
python -m pytest tests/test_vision_models.py -k pixtral
```

## Attribution

Ernie 4.5 modeling code is sourced from [Baidu](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT/tree/da6f3b1158d5d0d2bbf552bfc3364c9ec64e8aa5)