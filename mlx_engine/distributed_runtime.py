"""Inspect an installed distributed runtime without initializing a rank group.

Run this file by absolute path, with Python isolated mode. Importing the
mlx_engine package before group initialization changes Metal stream ordering.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def describe_runtime():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        digest.update(str(source.relative_to(package)).encode())
        digest.update(b"\\0")
        digest.update(source.read_bytes())
    versions = {}
    for dependency in ("mlx", "mlx-lm", "mlx-vlm", "transformers"):
        versions[dependency] = importlib.metadata.version(dependency)
    digest.update(json.dumps(versions, sort_keys=True).encode())
    return {"protocol": 1, "digest": digest.hexdigest(), "versions": versions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    arguments = parser.parse_args()
    result = describe_runtime()
    if arguments.model is not None:
        config = json.loads((Path(arguments.model) / "config.json").read_text())
        from mlx_lm.utils import _get_classes

        model_class, _ = _get_classes(config)
        if not hasattr(model_class, "shard"):
            raise ValueError("This model architecture does not support tensor sharding.")
        result["model_type"] = config.get("model_type")
    print("MLX_DISTRIBUTED_RUNTIME=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
