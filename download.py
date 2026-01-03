from huggingface_hub import snapshot_download
from pathlib import Path

Path("./models/Qwen3-4B-MLX-4bit").mkdir(exist_ok=True)
snapshot_download(
    repo_id="Qwen/Qwen3-4B-MLX-4bit",
    local_dir="./models/Qwen3-4B-MLX-4bit"
)