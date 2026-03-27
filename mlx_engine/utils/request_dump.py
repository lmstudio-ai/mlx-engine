import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import mlx.core as mx


REQUEST_DUMP_DIR_ENV_VAR = "LLMSTER_REQUEST_DUMP_DIR"
DEFAULT_REQUEST_DUMP_DIR_NAME = "request_dump_output"


def _request_dump_dir() -> Path:
    override = os.environ.get(REQUEST_DUMP_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / DEFAULT_REQUEST_DUMP_DIR_NAME


def _token_text(tokenizer, token_id: int) -> str:
    try:
        token_text = tokenizer.convert_ids_to_tokens(int(token_id))
        if token_text is not None:
            return str(token_text)
    except Exception:
        pass

    for decode_input in ([int(token_id)], int(token_id)):
        try:
            decoded = tokenizer.decode(decode_input)
            if decoded is not None:
                return str(decoded)
        except Exception:
            continue

    return ""


def write_request_settings(prefix: str, settings: dict[str, Any]) -> None:
    output_dir = _request_dump_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Request dump directory: {output_dir}", file=sys.stderr, flush=True)
    (output_dir / f"{prefix}.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True, default=str)
    )


def write_request_prompt(
    prefix: str,
    prompt: str,
    token_ids: Sequence[int],
    tokenizer,
) -> None:
    output_dir = _request_dump_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{prefix}.txt").write_text(prompt)
    with (output_dir / f"{prefix}_prompt_tokens.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Index", "ID", "Text"])
        for index, token_id in enumerate(token_ids):
            writer.writerow(
                [index, int(token_id), _token_text(tokenizer, int(token_id))]
            )


def write_first_token_logits(prefix: str, logits) -> None:
    output_dir = _request_dump_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    logits_row = mx.array(logits[0])
    mx.eval(logits_row)
    with (output_dir / f"{prefix}_first_token_logits.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Logit"])
        for token_id, logit in enumerate(logits_row.tolist()):
            writer.writerow([token_id, logit])


class FirstTokenLogitsProcessor:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self._has_dumped = False

    def __call__(self, tokens, logits):
        if not self._has_dumped:
            write_first_token_logits(self.prefix, logits)
            self._has_dumped = True
        return logits


def create_first_token_logits_processor(prefix: str) -> FirstTokenLogitsProcessor:
    return FirstTokenLogitsProcessor(prefix)
