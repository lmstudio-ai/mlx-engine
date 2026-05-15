import mlx.core as mx

from mlx_engine.processors.repetition_penalty_processor import (
    RepetitionPenaltyProcessor,
)


def test_repetition_penalty_last_token_fast_path_matches_full_context():
    logits = mx.array([[1.0, -1.0, 2.0, -2.0, 3.0]], dtype=mx.float32)
    processor = RepetitionPenaltyProcessor(
        token_history=[4],
        repetition_penalty=2.0,
        repetition_context_size=3,
    )

    processor(mx.array([1, 2], dtype=mx.int32), logits)
    fast_path = processor.process_last_token(mx.array([3], dtype=mx.int32), logits)

    expected_processor = RepetitionPenaltyProcessor(
        token_history=[4],
        repetition_penalty=2.0,
        repetition_context_size=3,
    )
    expected = expected_processor(mx.array([1, 2, 3], dtype=mx.int32), logits)

    mx.eval(fast_path, expected)
    assert fast_path.tolist() == expected.tolist()
