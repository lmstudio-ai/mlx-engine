import mlx.core as mx
from pathlib import Path
from os import makedirs
from datetime import datetime
from csv import DictWriter
from typing import Dict

"""
Wrapper to dump logits to a directory for debugging.
"""


class DumpLogitsProcessor:
    def __init__(
        self,
        vocab: Dict[str, int],
        dump_directory: Path,
    ):
        token_id_to_str_map = {}
        for token_str, token_id in vocab.items():
            token_id_to_str_map[token_id] = token_str
        self._vocab = [token_id_to_str_map[i] for i in range(len(token_id_to_str_map))]
        if len(self._vocab) != len(vocab):
            raise RuntimeError(
                f"Vocab of size {len(vocab)} had {len(self._vocab)} unique token IDs."
            )
        # Append current time so that we can re-run the same command without
        # overwriting previous outputs
        self._dump_directory = dump_directory / datetime.now().isoformat()
        makedirs(self._dump_directory, exist_ok=True)
        print(f"Will dump logits to {self._dump_directory}")

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Dump the logits to a file in the specified directory

        Args:
            tokens: The tokens to be processed.
            logits: The logits to be processed.
        """
        dump_file = self._dump_directory / f"logits_{len(tokens):04d}.csv"
        flat_logits = logits.squeeze(0).tolist()
        probs = mx.softmax(logits, 1).squeeze(0).tolist()
        vocab = self._vocab.copy()
        if len(flat_logits) < len(vocab):
            # Does not make sense for number of logits to be smaller than vocab
            raise RuntimeError(
                f"Got {len(flat_logits)} logits but vocab size {len(vocab)}"
            )
        elif len(flat_logits) > len(vocab):
            # Also weird, but (maybe) expected for language models.
            # Pad the vocab with "!!!OUT OF RANGE!!!"
            vocab.extend(
                ["!!!OUT OF RANGE!!!" for _ in range(len(flat_logits) - len(vocab))]
            )
        output = sorted(
            [
                {
                    "token_id": token_id,
                    "token_str": token_str,
                    "logit": logit,
                    "prob": prob,
                }
                for (token_id, (token_str, logit, prob)) in enumerate(
                    zip(vocab, flat_logits, probs, strict=True)
                )
            ],
            key=lambda d: d["prob"],
            reverse=True,
        )
        with open(dump_file, "w") as f:
            writer = DictWriter(f, ["token_id", "token_str", "logit", "prob"])
            writer.writeheader()
            for row in output:
                writer.writerow(row)
        return logits
