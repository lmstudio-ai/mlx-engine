import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.utils import sharded_load

from mlx_engine.utils.disable_hf_download import _original_snapshot_download
from mlx_engine.utils.fix_mistral_pre_tokenizer import fix_mistral_pre_tokenizer
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter


logger = logging.getLogger(__name__)


def _format_size_bytes(size_bytes: int) -> str:
    gibibytes = size_bytes / (1024 * 1024 * 1024)
    return f"{gibibytes:.2f} GiB"


def _model_snapshot_summary(model_path: Path) -> str:
    try:
        safetensor_files = list(model_path.glob("*.safetensors"))
        total_safetensor_size = sum(
            safetensor_file.stat().st_size for safetensor_file in safetensor_files
        )
        return (
            f"safetensors={len(safetensor_files)} "
            f"safetensorsSize={_format_size_bytes(total_safetensor_size)}"
        )
    except Exception as caught_error:
        return f"snapshotSummaryError={caught_error}"


class DistributedModelKit:
    """
    Minimal text-only model kit for MLX distributed tensor parallel inference.

    This is intentionally smaller than ModelKit. It exists so cluster-tool can
    validate distributed loading and generation through mlx-engine before that
    path is wired into the production app runtime.
    """

    def __init__(
        self,
        model_path: str | Path,
        prefill_step_size: int,
        *,
        max_kv_size: int | None = None,
        trust_remote_code: bool = False,
        distributed_group: Any = None,
    ):
        self.generation_lock = threading.Lock()
        self.pending_requests: dict[str, threading.Event] = {}
        self._shutdown = threading.Event()
        self.prefill_step_size = prefill_step_size
        self.max_kv_size = max_kv_size
        self.kv_bits = None
        self.kv_group_size = None
        self.quantized_kv_start = None
        self.draft_model = None
        self._cross_prompt_cache_active = False

        logger.info("Resolving distributed model path from %s", model_path)
        self.model_path = self._resolve_model_path(model_path)
        logger.info(
            "Resolved distributed model path to %s (%s)",
            self.model_path,
            _model_snapshot_summary(self.model_path),
        )
        config_json = json.loads((self.model_path / "config.json").read_text())
        if "vision_config" in config_json:
            raise ValueError(
                "DistributedModelKit supports text-only models; vision models are not supported"
            )
        self.model_type = config_json.get("model_type", None)
        logger.info(
            "Loaded distributed model config model_type=%s max_kv_size=%s prefill_step_size=%s",
            self.model_type,
            self.max_kv_size,
            self.prefill_step_size,
        )

        self.group = distributed_group
        if self.group is None:
            logger.info("Initializing MLX distributed group inside DistributedModelKit")
            self.group = mx.distributed.init()
        if self.group.size() <= 1:
            raise ValueError("DistributedModelKit requires more than one MLX rank")

        logger.info(
            "Loading distributed model shard from %s on rank %s/%s...",
            self.model_path,
            self.group.rank(),
            self.group.size(),
        )
        sharded_load_started_at = time.monotonic()
        self.model, self.tokenizer = sharded_load(
            self.model_path,
            tensor_group=self.group,
            pipeline_group=None,
        )
        logger.info(
            "sharded_load returned on rank %s/%s after %.1fs",
            self.group.rank(),
            self.group.size(),
            time.monotonic() - sharded_load_started_at,
        )
        logger.info("Applying tokenizer fixes on rank %s", self.group.rank())
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer,
            model_path=self.model_path,
            model_type=self.model_type,
        )
        self.detokenizer = self.tokenizer.detokenizer
        logger.info("Distributed model shard loaded on rank %s", self.group.rank())

    def _resolve_model_path(self, model_path: str | Path) -> Path:
        candidate_path = Path(model_path).expanduser()
        if candidate_path.exists():
            return candidate_path

        try:
            return Path(
                _original_snapshot_download(
                    str(model_path),
                    local_files_only=True,
                )
            )
        except Exception as caught_error:
            raise FileNotFoundError(
                f"Model is not available in the local Hugging Face cache: {model_path}"
            ) from caught_error

    def start(self):
        logger.info(
            "Synchronizing distributed model start on rank %s/%s",
            self.group.rank(),
            self.group.size(),
        )
        mx.synchronize()
        logger.info("Distributed model start synchronized on rank %s", self.group.rank())

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def process_prompt(
        self,
        prompt_tokens,
        images_b64: Optional[List[str]],
        prompt_progress_reporter: PromptProgressReporter,
        generate_args: dict,
        max_image_size: tuple[int, int] | None,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        if images_b64 is not None and len(images_b64) > 0:
            raise ValueError("DistributedModelKit does not support images yet")
        if speculative_decoding_toggle is True:
            raise ValueError("DistributedModelKit does not support draft models yet")
        if len(prompt_tokens) == 0:
            logger.warning(
                "Received empty prompt. Generation quality will likely be poor"
            )
            prompt_tokens = self.tokenize(" ")
        return mx.array(prompt_tokens), None

    def is_cross_prompt_cache_active(self) -> bool:
        return self._cross_prompt_cache_active

    def record_token_to_cache(self, token: int) -> None:
        return

    def is_draft_model_compatible(self, path: str) -> bool:
        return False

    def load_draft_model(self, path: str) -> None:
        raise ValueError("DistributedModelKit does not support draft models")

    def unload_draft_model(self) -> None:
        return

    def cancel_request(self, request_id: str) -> bool:
        if request_id in self.pending_requests:
            logger.warning(
                "Ignoring local cancellation for distributed request %s. "
                "Distributed ranks must either drain the request together or be "
                "torn down by the rank supervisor.",
                request_id,
            )
        return False

    def shutdown(self) -> None:
        self._shutdown.set()

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()
