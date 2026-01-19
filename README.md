# MLX Engine

A high-performance FastAPI server for running Large Language Models (LLMs), Vision Language Models (VLMs), and Text-to-Speech (TTS) models on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

## Features

- **🚀 OpenAI-Compatible API (WIP)**: Implements OpenAI's chat completions format for easy integration 
- **🍎 Apple Silicon Optimized**: Leverages MLX for efficient inference on Mac
- **💬 Chat Completions**: Full support for multi-turn conversations with system/user/assistant roles
- **🖼️ Vision Language Models**: Process images alongside text using VLMs
- **📡 Streaming Support**: Real-time token streaming for responsive applications
- **⚡ Speculative Decoding**: Accelerate generation using a smaller draft model
- **💾 KV Cache Quantization**: Reduce memory usage with 3-8 bit quantization
- **🔊 Text-to-Speech**: Generate audio from text using MLX-based TTS models
- **📦 Model Management**: Download and convert Hugging Face models to MLX format

## Acknowledgements

This project is based on the open-source work from [LM Studio](https://lmstudio.ai/). Thank you to the LM Studio team!
