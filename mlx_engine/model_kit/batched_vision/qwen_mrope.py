from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class QwenMropeState:
    """Request-local MRoPE positions for one expanded Qwen image prompt."""

    position_ids: mx.array
    rope_deltas: mx.array


def apply_qwen_image_mrope_state(
    model,
    *,
    input_ids: mx.array,
    image_grid_thw: mx.array | None,
) -> QwenMropeState | None:
    """Patch mlx-vlm's model-side Qwen MRoPE state from image token runs."""
    language_model = getattr(model, "language_model", None)
    config = getattr(model, "config", None)
    if language_model is None or config is None or image_grid_thw is None:
        return None

    image_token_id = getattr(config, "image_token_id", None)
    vision_config = getattr(config, "vision_config", None)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
    if image_token_id is None or spatial_merge_size is None:
        return None

    state = build_qwen_image_mrope_state(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )
    language_model._position_ids = state.position_ids
    language_model._rope_deltas = state.rope_deltas
    return state


def build_qwen_image_mrope_state(
    *,
    input_ids: mx.array,
    image_grid_thw: mx.array,
    image_token_id: int,
    spatial_merge_size: int,
) -> QwenMropeState:
    """Build Qwen MRoPE state by walking expanded image-token spans."""
    token_list = input_ids.squeeze(0).tolist()
    image_runs = _find_token_runs(token_list, image_token_id)
    grid_list = image_grid_thw.tolist()
    if len(grid_list) == 3 and isinstance(grid_list[0], int):
        grid_list = [grid_list]
    if len(image_runs) != len(grid_list):
        raise ValueError(
            "Qwen image token runs do not match image_grid_thw entries: "
            f"{len(image_runs)} runs vs {len(grid_list)} grids."
        )

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
        run_length = run_end - run_start
        expected_run_length = llm_grid_t * llm_grid_h * llm_grid_w
        if run_length != expected_run_length:
            raise ValueError(
                "Qwen image token run length does not match image_grid_thw: "
                f"run length {run_length}, expected {expected_run_length}."
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
        token_cursor = run_end

    trailing_text_len = len(token_list) - token_cursor
    for dim in range(3):
        positions[dim].extend(
            range(position_cursor, position_cursor + trailing_text_len)
        )
    position_cursor += trailing_text_len

    return QwenMropeState(
        position_ids=mx.array(positions, dtype=input_ids.dtype).reshape(
            3, 1, len(token_list)
        ),
        rope_deltas=mx.array(position_cursor - len(token_list), dtype=input_ids.dtype),
    )


def _find_token_runs(tokens: list[int], target_token: int) -> list[tuple[int, int]]:
    """Return `[start, end)` ranges where `target_token` is contiguous."""
    runs = []
    start = None
    for idx, token in enumerate(tokens):
        if token == target_token:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(tokens)))
    return runs
