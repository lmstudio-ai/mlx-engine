import unittest
from unittest.mock import patch
import mlx.core as mx
from mlx_engine.cache_wrapper import CacheWrapper, StopPromptProcessing
from mlx_engine.prompt_cache_session import PromptCacheSession
from tests.shared import model_getter, RecordingReporter, CancellingReporter
from mlx_engine.generate import load_model, tokenize


class FakeCache:
    def __init__(self, offset: int = 0, trimmable: bool = True):
        self.offset = offset
        self._trimmable = trimmable

    @property
    def state(self):
        return []

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        if not self._trimmable:
            return 0
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def nbytes(self):
        return max(self.offset, 1)


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, tokens, cache):
        n_tokens = tokens.shape[1]
        self.calls.append(n_tokens)
        for entry in cache:
            entry.offset += n_tokens


class TestCacheWrapper(unittest.TestCase):
    def test_prompt_processing_cancellation(self):
        """Test that progress is saved when processing is cancelled and cache is reused on retry"""

        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        chunk_size = 20  # Small chunk size to ensure multiple progress callbacks
        num_tokens_to_exclude = 1
        model_kit.cache_wrapper = CacheWrapper(
            model_kit.model,
            max_kv_size=4096,
            chunk_size=chunk_size,
        )

        long_prompt = (
            "This is a test prompt that needs to be long enough to require multiple chunks for processing. "
            * 50
        )
        prompt_tokens = mx.array(tokenize(model_kit, long_prompt))

        # First attempt: Reporter that cancels after 3 events
        cancelling_reporter = CancellingReporter(cancel_after=3)

        with self.assertRaises(StopPromptProcessing):
            model_kit.cache_wrapper.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=cancelling_reporter,
                num_tokens_to_exclude=1,
            )

        # Second attempt: Reporter that doesn't cancel
        recording_reporter = RecordingReporter()

        result_tokens = model_kit.cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            reporter=recording_reporter,
            num_tokens_to_exclude=1,
        )
        cached_before_cancel = cancelling_reporter.events[-1][
            "prefill_tokens_processed"
        ]
        retry_begin_event = recording_reporter.events[0]
        self.assertEqual(retry_begin_event["type"], "begin")
        self.assertEqual(retry_begin_event["cached_tokens"], cached_before_cancel)
        self.assertEqual(recording_reporter.events[-1]["type"], "finish")

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)
        self.assertEqual(
            result_tokens.tolist(), prompt_tokens[-num_tokens_to_exclude:].tolist()
        )

    def test_flush_live_cache_only_stores_full_snapshot(self):
        session = PromptCacheSession(
            model=type("FakeLayeredModel", (), {"layers": [object()]})(),
            max_kv_size=None,
            chunk_size=8,
        )
        session._live_cache = [FakeCache(offset=8, trimmable=False)]
        session._live_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)

        stored_prefixes = []

        def store_snapshot(tokens, cache):
            stored_prefixes.append(tokens.tolist())

        session._store_snapshot = store_snapshot
        session._flush_live_cache()

        self.assertEqual(stored_prefixes, [[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_fetch_reusable_cache_falls_back_to_shorter_checkpoint(self):
        session = PromptCacheSession(
            model=type("FakeLayeredModel", (), {"layers": [object()]})(),
            max_kv_size=None,
            chunk_size=8,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)

        # Exact hit exists but cannot be trimmed to leave a seed token outside cache.
        session._history.insert_cache(
            session._history_key,
            prompt.tolist(),
            [FakeCache(offset=6, trimmable=False)],
        )
        # Checkpoint four tokens before prompt end. This is the usable fallback.
        session._history.insert_cache(
            session._history_key,
            prompt[:2].tolist(),
            [FakeCache(offset=2, trimmable=True)],
        )

        cache, rest = session._restore_cache(prompt)

        self.assertIsNotNone(cache)
        self.assertEqual(cache[0].offset, 2)
        self.assertEqual(rest.tolist(), [3, 4, 5, 6])

    def test_restore_cache_returns_early_for_empty_prompt(self):
        session = PromptCacheSession(
            model=type("FakeLayeredModel", (), {"layers": [object()]})(),
            max_kv_size=None,
            chunk_size=8,
        )
        prompt = mx.array([], dtype=mx.int32)

        with patch.object(
            session._history,
            "fetch_nearest_cache",
            side_effect=AssertionError("history lookup should not run"),
        ):
            cache, rest = session._restore_cache(prompt)

        self.assertIsNone(cache)
        self.assertEqual(rest.tolist(), [])

    def test_fetch_reusable_cache_trims_exact_hit_to_one_seed_token(self):
        session = PromptCacheSession(
            model=type("FakeLayeredModel", (), {"layers": [object()]})(),
            max_kv_size=None,
            chunk_size=8,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)

        session._history.insert_cache(
            session._history_key,
            prompt.tolist(),
            [FakeCache(offset=6, trimmable=True)],
        )

        cache, rest = session._restore_cache(prompt)

        self.assertIsNotNone(cache)
        self.assertEqual(cache[0].offset, 5)
        self.assertEqual(rest.tolist(), [6])

    def test_prefill_splits_chunk_to_store_near_end_checkpoint(self):
        session = PromptCacheSession(
            model=type("FakeLayeredModel", (), {"layers": [object()]})(),
            max_kv_size=None,
            chunk_size=8,
        )
        session._live_cache = [FakeCache(offset=0, trimmable=True)]
        session._live_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)

        model = FakeModel()
        reporter = RecordingReporter()
        stored_prefixes = []

        def store_snapshot(tokens, cache):
            stored_prefixes.append(tokens.tolist())

        session._store_snapshot = store_snapshot

        with (
            patch("mlx_engine.prompt_cache_session.mx.eval"),
            patch("mlx_engine.prompt_cache_session.mx.clear_cache"),
        ):
            session._prefill_cache(
                model=model,
                cache=session._live_cache,
                cache_start=0,
                tokens=mx.array([1, 2, 3, 4, 5, 6, 7], dtype=mx.int32),
                reporter=reporter,
                is_draft=False,
                checkpoint_prefix_len=4,
            )

        self.assertEqual(model.calls, [4, 3])
        self.assertEqual(stored_prefixes, [[1, 2, 3, 4]])

    def test_draft_model_switch_resets_history(self):
        session = PromptCacheSession(
            model=type("FakeLayeredModel", (), {"layers": [object()]})(),
            max_kv_size=None,
            chunk_size=8,
        )
        prompt = mx.array([1, 2, 3], dtype=mx.int32)

        session._history.insert_cache(
            session._history_key,
            prompt.tolist(),
            [FakeCache(offset=3, trimmable=True)],
        )

        session.set_draft_model(type("FakeDraftModel", (), {"layers": [object()]})())

        cache, rest = session._restore_cache(prompt)

        self.assertIsNone(cache)
        self.assertEqual(rest.tolist(), [1, 2, 3])

    def test_quantized_prefill_syncs_live_cache_and_skips_checkpointing(self):
        model = FakeModel()
        model.layers = [object()]
        session = PromptCacheSession(
            model=model,
            max_kv_size=None,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
            chunk_size=8,
        )
        session._live_cache = [FakeCache(offset=0, trimmable=True)]
        reporter = RecordingReporter()
        stored_prefixes = []
        replacement = FakeCache(offset=0, trimmable=True)

        def store_snapshot(tokens, cache):
            stored_prefixes.append(tokens.tolist())

        def quantize_cache(prompt_cache, **_):
            replacement.offset = prompt_cache[0].offset
            prompt_cache[0] = replacement

        session._store_snapshot = store_snapshot

        with (
            patch(
                "mlx_engine.prompt_cache_session.maybe_quantize_kv_cache",
                quantize_cache,
            ),
            patch("mlx_engine.prompt_cache_session.mx.eval"),
            patch("mlx_engine.prompt_cache_session.mx.clear_cache"),
        ):
            prepared = session.prepare(
                mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32),
                reporter,
                num_tokens_to_exclude=1,
            )

        self.assertEqual(model.calls, [7])
        self.assertEqual(stored_prefixes, [])
        self.assertIs(session._live_cache[0], replacement)
        self.assertIs(prepared.cache[0], replacement)


if __name__ == "__main__":
    unittest.main(verbosity=2)
