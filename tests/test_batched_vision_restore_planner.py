from mlx_engine.model_kit.batched_vision.prompt_cache.restore_planner import (
    PromptCacheRestorePlanner,
)
from mlx_engine.model_kit.batched_vision.prompt_cache.types import (
    PromptCacheLayout,
    PromptCacheRecordMetadata,
    PromptPrefixChunk,
    RECORD_KIND_KV_DELTA,
    RECORD_KIND_ROTATING_DELTA,
    RECORD_KIND_STATE_CHECKPOINT,
    RECORD_WRITE_ORDER,
    RecordKind,
    make_record_key,
)


def _chunk(start: int, end: int) -> PromptPrefixChunk:
    return PromptPrefixChunk(start=start, end=end, key=f"chunk-{start}-{end}")


def _layout() -> PromptCacheLayout:
    return PromptCacheLayout(
        layer_kinds=[
            RECORD_KIND_KV_DELTA,
            RECORD_KIND_ROTATING_DELTA,
            RECORD_KIND_STATE_CHECKPOINT,
        ],
        layer_indices_by_kind={
            RECORD_KIND_KV_DELTA: [0],
            RECORD_KIND_ROTATING_DELTA: [1],
            RECORD_KIND_STATE_CHECKPOINT: [2],
        },
        rotating_window_size=512,
    )


def _metadata_for(chunk: PromptPrefixChunk, record_kind: RecordKind):
    return PromptCacheRecordMetadata(
        chunk_key=chunk.key,
        record_kind=record_kind,
        layer_indices=[],
    )


def _planner(chunks, existing_records):
    metadata_by_key = {}
    for chunk in chunks:
        for record_kind in RECORD_WRITE_ORDER:
            record_key = _record_key(chunk, record_kind)
            if record_key in existing_records:
                metadata_by_key[record_key] = _metadata_for(chunk, record_kind)

    return PromptCacheRestorePlanner(
        layout=_layout(),
        record_metadata_by_key=metadata_by_key,
        record_exists=existing_records.__contains__,
    )


def _record_key(chunk: PromptPrefixChunk, record_kind: str) -> str:
    return make_record_key(chunk.key, record_kind)


def test_restore_planner_selects_records_by_cache_kind():
    """KV needs every chunk, SWA needs the target window, state needs target only."""
    chunks = [_chunk(0, 256), _chunk(256, 512), _chunk(512, 768)]
    existing_records = {
        _record_key(chunks[0], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_KV_DELTA),
        _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
        _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
    }

    record_keys_by_chunk = _planner(
        chunks,
        existing_records,
    ).restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk == {
        chunks[0].key: [_record_key(chunks[0], RECORD_KIND_KV_DELTA)],
        chunks[1].key: [
            _record_key(chunks[1], RECORD_KIND_KV_DELTA),
            _record_key(chunks[1], RECORD_KIND_ROTATING_DELTA),
        ],
        chunks[2].key: [
            _record_key(chunks[2], RECORD_KIND_KV_DELTA),
            _record_key(chunks[2], RECORD_KIND_ROTATING_DELTA),
            _record_key(chunks[2], RECORD_KIND_STATE_CHECKPOINT),
        ],
    }


def test_restore_planner_returns_none_when_required_record_is_missing():
    """A restore plan is all-or-nothing for the selected boundary."""
    chunks = [_chunk(0, 256), _chunk(256, 512)]
    existing_records = {
        _record_key(chunks[0], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_KV_DELTA),
        _record_key(chunks[1], RECORD_KIND_STATE_CHECKPOINT),
    }

    record_keys_by_chunk = _planner(
        chunks,
        existing_records,
    ).restore_record_keys_for_chunk_chain(chunks)

    assert record_keys_by_chunk is None
