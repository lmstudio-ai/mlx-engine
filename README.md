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
Run the `demo.py` script with an MLX text model:
```
python demo.py --model ~/.cache/lm-studio/models/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit 
```
[mlx-community/Meta-Llama-3.1-8B-Instruct-4bit](https://model.lmstudio.ai/download/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit) - 4.53 GB

This command will use a default prompt that is formatted for Llama-3.1. For other models, add a custom `--prompt` argument with the correct prompt formatting:
```
python demo.py --model ~/.cache/lm-studio/models/mlx-community/Mistral-Small-Instruct-2409-4bit --prompt "<s> [INST] How long will it take for an apple to fall from a 10m tree? [/INST]"
```
[mlx-community/Mistral-Small-Instruct-2409-4bit](https://model.lmstudio.ai/download/mlx-community/Mistral-Small-Instruct-2409-4bit) - 12.52 GB

### Vision Model Demo
Run the `demo.py` script with an MLX vision model:
```
python demo.py --model ~/.cache/lm-studio/models/mlx-community/pixtral-12b-4bit --prompt "<s>[INST]Compare these images[IMG][IMG][/INST]" --images demo-data/chameleon.webp demo-data/toucan.jpeg
```
Currently supported vision models and download links:
 - Llama-3.2-Vision
   - [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit](https://model.lmstudio.ai/download/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit)
 - Pixtral
   - [mlx-community/pixtral-12b-4bit](https://model.lmstudio.ai/download/mlx-community/pixtral-12b-4bit) - 7.15 GB
 - Qwen2-VL
   - [mlx-community/Qwen2-VL-2B-4bit](https://model.lmstudio.ai/download/mlx-community/Qwen2-VL-2B-4bit) - 1.26 GB
   - [mlx-community/Qwen2-VL-7B-Instruct-4bit](https://model.lmstudio.ai/download/mlx-community/Qwen2-VL-7B-Instruct-4bit) - 4.68 GB
 - Llava-v1.6
   - [mlx-community/llava-v1.6-mistral-7b-4bit](https://model.lmstudio.ai/download/mlx-community/llava-v1.6-mistral-7b-4bit) - 4.26 GB
