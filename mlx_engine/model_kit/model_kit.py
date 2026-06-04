import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Tuple
import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_engine.cache_wrapper import CacheWrapper
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
import logging
from mlx_engine.utils.prompt_processing import process_prompt_text_only
from mlx_engine.utils.fix_mistral_pre_tokenizer import fix_mistral_pre_tokenizer
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter
from mlx_engine.utils.eot_tokens import sanitize_eos_tokens
from mlx_engine.utils.set_seed import set_seed
import threading

logger = logging.getLogger(__name__)


class ModelKit:
    """
    Collection of objects and methods that are needed for operating a model.

    Args:
        model_path (Path): Path to the model directory containing model files.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        kv_bits (Optional[int]): Number of bits for KV cache quantization. None disables quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization. Defaults to 64.
        quantized_kv_start (Optional[int]): Step to begin KV cache quantization when enabled. Defaults to 0.
    """

    # model state tracking
    model: nn.Module = None
    tokenizer: TokenizerWrapper = None
    detokenizer: StreamingDetokenizer = None
    cache_wrapper: Optional[CacheWrapper] = None
    _cross_prompt_cache_active: bool = False
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    kv_group_size: Optional[int] = None
    quantized_kv_start: Optional[int] = None
    draft_model: Optional[nn.Module] = None
    model_type: Optional[str] = None
    generation_lock: threading.Lock
    pending_requests: dict[str, threading.Event]
    _shutdown: threading.Event

    def _vocab_only_init(self, model_path: Path):
        logger.info(f"Loading model (vocab-only) from {model_path}...")
        self.tokenizer = mlx_lm.tokenizer_utils.load(model_path)
        self.detokenizer = self.tokenizer.detokenizer
        logger.info("Model (vocab-only) loaded successfully")

    def _full_model_init(
        self,
        model_path: Path,
        prefill_step_size: int,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        if kv_bits and max_kv_size is not None:
            # Quantized KV cache is only supported for non-rotating KV cache
            logger.warning("max_kv_size is ignored when using KV cache quantization")
            max_kv_size = None
        self.model_path = model_path
        logger.info(f"Loading model from {model_path}...")
        config_json = json.loads((model_path / "config.json").read_text())
        self.model_type = config_json.get("model_type", None)

        self.model, self.tokenizer = mlx_lm.utils.load(self.model_path)
        fix_mistral_pre_tokenizer(
            tokenizer=self.tokenizer, model_path=model_path, model_type=self.model_type
        )
        self.detokenizer = self.tokenizer.detokenizer
        self.cache_wrapper = CacheWrapper(
            self.model,
            max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            chunk_size=prefill_step_size,
        )
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        logger.info("Model loaded successfully")

    def __init__(
        self,
        model_path: Path,
        prefill_step_size: int,
        vocab_only: bool = False,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.generation_lock = threading.Lock()
        self.pending_requests = {}
        self._shutdown = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-engine-sequential-generation",
        )
        # Public because the generation loop in generate.py reads this from the model_kit instance
        self.prefill_step_size = prefill_step_size

        def initialize() -> None:
            set_seed(seed)
            if vocab_only:
                self._vocab_only_init(model_path)
            else:
                self._full_model_init(
                    model_path,
                    prefill_step_size,
                    max_kv_size=max_kv_size,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                )
            sanitize_eos_tokens(self)

        try:
            self._executor.submit(initialize).result()
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def start(self):
        self._executor.submit(mx.synchronize).result()

    def tokenize(self, prompt: str) -> List[int]:
        return self._executor.submit(self._tokenize, prompt).result()

    def _tokenize(self, prompt: str) -> List[int]:
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
        is_text_only_processing = images_b64 is None or len(images_b64) == 0
        if not is_text_only_processing:
            raise ValueError(
                "Images are only supported by the batched vision model path"
            )
        self._cross_prompt_cache_active = True
        if len(prompt_tokens) == 0:
            logger.warning(
                "Received empty prompt. Generation quality will likely be poor"
            )
            # Models expect some sort of input, so add whitespace
            prompt_tokens = self._tokenize(" ")
        return process_prompt_text_only(
            mx.array(prompt_tokens),
            self.cache_wrapper,
            generate_args,
            self.draft_model,
            speculative_decoding_toggle,
            prompt_progress_reporter,
        ), None

    def is_cross_prompt_cache_active(self) -> bool:
        """
        Check if cross-prompt caching is currently enabled.
        Can be overridden by subclasses for custom behavior.
        """
        return self._cross_prompt_cache_active

    def record_token_to_cache(self, token: int) -> None:
        self.cache_wrapper.record_generated_token(token)

    def is_draft_model_compatible(self, path: str | Path) -> bool:
        return self._executor.submit(self._is_draft_model_compatible, path).result()

    def _is_draft_model_compatible(self, path: str | Path) -> bool:
        path = Path(path)
        if self.tokenizer is None:
            logger.warning(
                "Draft model compatibility check requires at least a vocab-only loaded main model"
            )
            return False
        draft_tokenizer = mlx_lm.tokenizer_utils.load(path)
        if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            return False
        return True

    def load_draft_model(self, path: str | Path) -> None:
        self._executor.submit(self._load_draft_model, path).result()

    def _load_draft_model(self, path: str | Path) -> None:
        logger.info(f"Loading draft model from {path}...")
        path = Path(path)
        if self.model is None:
            raise ValueError("Main model must be loaded before loading a draft model")
        if not self._is_draft_model_compatible(path):
            raise ValueError("Draft model is not compatible with main model")
        self.draft_model, _ = mlx_lm.utils.load(path)
        self.cache_wrapper.set_draft_model(self.draft_model)
        logger.info("Draft model loaded")

    def unload_draft_model(self) -> None:
        self._executor.submit(self._unload_draft_model).result()

    def _unload_draft_model(self) -> None:
        if self.draft_model is None:
            logger.info("No loaded draft model to unload")
        else:
            self.draft_model = None
            self.cache_wrapper.unset_draft_model()
        # Noticed that draft model memory would not be released without clearing metal cache
        mx.clear_cache()

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or active request by ID. Returns True if request was found."""
        if request_id in self.pending_requests:
            self.pending_requests[request_id].set()
            return True
        return False

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._executor.shutdown(wait=True)

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()
