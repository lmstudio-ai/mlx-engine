from mlx_engine.generate import load_model, create_generator, tokenize
from pathlib import Path

if __name__ == "__main__":
    model_path = Path("~/.cache/lm-studio/models/mlx-community/Llama-3.2-1B-Instruct-4bit").expanduser()
    load_model(str(model_path), max_kv_size=4096, trust_remote_code=False)

    prompt = "Explain the rules of sudoku"
    prompt_tokens = tokenize(f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""")
    generator = create_generator(prompt_tokens, None, None, {"max_tokens": 1024})
    for token in generator:
        print(token, end="")
    print()
