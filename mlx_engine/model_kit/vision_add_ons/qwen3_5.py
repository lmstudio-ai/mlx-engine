import logging
from pathlib import Path

from mlx import nn
import mlx.core as mx

from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn
from mlx_engine.model_kit.vision_add_ons.load_utils import load_vision_addon
from mlx_engine.model_kit.vision_add_ons.qwen_vl_utils import compute_qwen_vl_embeddings

from mlx_vlm.models.qwen3_5 import (
    VisionModel as Qwen3_5VisionTower,
    ModelConfig as Qwen3_5ModelConfig,
    VisionConfig as Qwen3_5VisionConfig,
    TextConfig as Qwen3_5TextConfig,
    Model as Qwen3_5VLModel,
)

logger = logging.getLogger(__name__)


def _find_token_runs(tokens: list[int], target_token: int) -> list[tuple[int, int]]:
    """Return contiguous index ranges where a token repeats."""
    runs = []
    start = None
    for idx, token in enumerate(tokens):
        if token == target_token:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(tokens) - 1))
    return runs


def _compute_image_mrope_state(
    input_ids: mx.array,
    image_grid_thw: mx.array,
    config: Qwen3_5ModelConfig,
) -> tuple[mx.array, mx.array]:
    """Compute Qwen3.5 image MRoPE state from merged prompt tokens.

    This mirrors the intended Qwen rope-index construction, but derives image
    spans from the merged image token runs directly.
    """
    token_list = input_ids.tolist()
    image_runs = _find_token_runs(token_list, config.image_token_id)
    grid_list = image_grid_thw.tolist()

    if len(grid_list) == 3 and isinstance(grid_list[0], int):
        grid_list = [grid_list]

    if len(image_runs) != len(grid_list):
        raise ValueError(
            "Qwen3.5 image token runs do not match image_grid_thw entries: "
            f"{len(image_runs)} runs vs {len(grid_list)} grids."
        )

    spatial_merge_size = config.vision_config.spatial_merge_size
    seq_length = len(token_list)
    positions = [[], [], []]
    token_cursor = 0
    position_cursor = 0

    for (run_start, run_end), (t, h, w) in zip(image_runs, grid_list):
        text_len = run_start - token_cursor
        for dim in range(3):
            positions[dim].extend(range(position_cursor, position_cursor + text_len))
        position_cursor += text_len

        llm_grid_t = int(t)
        llm_grid_h = int(h) // spatial_merge_size
        llm_grid_w = int(w) // spatial_merge_size
        run_length = run_end - run_start + 1
        expected_run_length = llm_grid_t * llm_grid_h * llm_grid_w
        if run_length != expected_run_length:
            raise ValueError(
                "Qwen3.5 image token run length does not match grid_thw: "
                f"run length {run_length}, expected {expected_run_length} "
                f"from grid {(t, h, w)}."
            )

        image_position_offset = position_cursor
        for t_idx in range(llm_grid_t):
            for h_idx in range(llm_grid_h):
                for w_idx in range(llm_grid_w):
                    positions[0].append(image_position_offset + t_idx)
                    positions[1].append(image_position_offset + h_idx)
                    positions[2].append(image_position_offset + w_idx)

        position_cursor = image_position_offset + max(
            llm_grid_t,
            llm_grid_h,
            llm_grid_w,
        )
        token_cursor = run_end + 1

    trailing_text_len = seq_length - token_cursor
    for dim in range(3):
        positions[dim].extend(
            range(position_cursor, position_cursor + trailing_text_len)
        )
    position_cursor += trailing_text_len

    position_ids = mx.array(positions, dtype=input_ids.dtype).reshape(3, 1, seq_length)
    rope_deltas = mx.array(position_cursor - seq_length, dtype=input_ids.dtype)
    return position_ids, rope_deltas


class Qwen3_5VisionAddOn(BaseVisionAddOn):
    """
    Vision add-on for Qwen3.5 Dense models.
    """

    def __init__(self, model_path: Path):
        super().__init__()
        self._init_common(
            model_path=model_path,
            model_cls=Qwen3_5VLModel,
            model_config_class=Qwen3_5ModelConfig,
            vision_config_class=Qwen3_5VisionConfig,
            text_config_class=Qwen3_5TextConfig,
            vision_tower_class=Qwen3_5VisionTower,
            addon_logger=logger,
        )

    def _init_common(
        self,
        model_path,
        model_cls,
        model_config_class,
        vision_config_class,
        text_config_class,
        vision_tower_class,
        addon_logger,
    ):
        """Shared initialization for dense and MoE variants."""
        self.model_cls = model_cls
        self.vision_tower, _, self.config, self.processor = load_vision_addon(
            model_path=model_path,
            model_config_class=model_config_class,
            vision_config_class=vision_config_class,
            text_config_class=text_config_class,
            vision_tower_class=vision_tower_class,
            multi_modal_projector_class=None,
            logger=addon_logger,
        )

    def clear_prediction_state(self, text_model: nn.Module) -> None:
        """Reset MRoPE state injected by compute_embeddings."""
        if not hasattr(text_model.language_model.model, "reset_mrope_state"):
            raise ValueError(
                "Qwen3.5 vision support requires the Qwen3.5 patch, but this build does not apply it."
            )
        text_model.language_model.model.reset_mrope_state()

    def compute_embeddings(
        self,
        text_model: nn.Module,
        prompt_tokens: mx.array,
        images_b64: list[str],
        max_size: tuple[int, int] | None,
    ) -> tuple[mx.array, mx.array]:
        """
        Compute input_ids and embeddings for text with images,
        then inject MRoPE position IDs into the patched text model.
        """

        result = compute_qwen_vl_embeddings(
            addon=self,
            text_model=text_model,
            prompt_tokens=prompt_tokens,
            images_b64=images_b64,
            qwen_vl_version=3,
            max_size=max_size,
        )

        # Compute and inject MRoPE position IDs for vision tokens
        if result.grid_thw is not None:
            position_ids, rope_deltas = _compute_image_mrope_state(
                result.input_ids,
                result.grid_thw,
                self.config,
            )
            text_model.language_model.model.position_ids = position_ids
            text_model.language_model.model.rope_deltas = rope_deltas

        return result.input_ids, result.embeddings
