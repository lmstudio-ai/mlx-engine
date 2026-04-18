import unittest
from contextlib import nullcontext
from unittest.mock import patch

import mlx.core as mx

from mlx_engine.cache_wrapper import (
    DEFAULT_CHECKPOINT_TAIL_TOKENS,
    CacheWrapper,
    StopPromptProcessing,
    _trim_cache_for_snapshot,
)
from mlx_engine.generate import load_model, tokenize
from tests.shared import CancellingReporter, RecordingReporter, model_getter


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

    def advance(self, n):
        self.offset += n


class FakeModel:
    def __init__(self, *, cache_trimmable: bool = True):
        self.layers = [object()]
        self.calls = []
        self.cache_trimmable = cache_trimmable

    def make_cache(self):
        return [FakeCache(trimmable=self.cache_trimmable)]

    def __call__(self, tokens, cache):
        n_tokens = tokens.shape[1]
        self.calls.append(n_tokens)
        for entry in cache:
            entry.offset += n_tokens


def _no_op_stream(_stream):
    return nullcontext()


class TestCacheWrapper(unittest.TestCase):
    def _make_session(self, *, cache_trimmable=True, chunk_size=8, **kwargs):
        model = FakeModel(cache_trimmable=cache_trimmable)
        session = CacheWrapper(
            model=model,
            max_kv_size=None,
            chunk_size=chunk_size,
            **kwargs,
        )
        return session, model

    def _run_update_cache(
        self,
        session,
        prompt_tokens,
        reporter=None,
    ):
        reporter = reporter or RecordingReporter()
        with (
            patch("mlx_engine.cache_wrapper.mx.stream", side_effect=_no_op_stream),
            patch("mlx_engine.cache_wrapper.mx.eval"),
            patch("mlx_engine.cache_wrapper.mx.clear_cache"),
        ):
            result_tokens = session.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=reporter,
            )
        return result_tokens, reporter

    def test_prompt_processing_cancellation(self):
        """Test that progress is saved when processing is cancelled and cache is reused on retry"""

        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        chunk_size = 20  # Small chunk size to ensure multiple progress callbacks
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
            )

        # Second attempt: Reporter that doesn't cancel
        recording_reporter = RecordingReporter()

        result_tokens = model_kit.cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            reporter=recording_reporter,
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
        self.assertEqual(result_tokens.tolist(), prompt_tokens[-1:].tolist())

    def test_full_snapshot_reuse_requires_a_longer_prompt_without_checkpoint(self):
        session, _ = self._make_session(
            cache_trimmable=False,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(
            session,
            mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=mx.int32),
        )
        self.assertEqual(reporter.events[0]["cached_tokens"], 8)
        self.assertEqual(result_tokens.tolist(), [9])

        _, reporter = self._run_update_cache(
            session,
            mx.array([1, 2, 3, 4], dtype=mx.int32),
        )
        self.assertEqual(reporter.events[0]["cached_tokens"], 0)

    def test_same_prompt_reuses_checkpoint_when_full_snapshot_cannot_trim(self):
        session, _ = self._make_session(cache_trimmable=False)
        prompt = mx.array(list(range(1, 16)), dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(
            reporter.events[0]["cached_tokens"],
            len(prompt) - DEFAULT_CHECKPOINT_TAIL_TOKENS,
        )
        self.assertEqual(result_tokens.tolist(), prompt[-1:].tolist())

    def test_empty_prompt_update_returns_an_empty_tail(self):
        session, _ = self._make_session()

        result_tokens, reporter = self._run_update_cache(
            session,
            mx.array([], dtype=mx.int32),
        )

        self.assertEqual(reporter.events[0]["cached_tokens"], 0)
        self.assertEqual(reporter.events[0]["total_prompt_tokens"], 0)
        self.assertEqual(reporter.events[-1]["type"], "finish")
        self.assertEqual(result_tokens.tolist(), [])

    def test_same_prompt_trims_an_exact_hit_to_leave_one_seed_token(self):
        session, _ = self._make_session(
            cache_trimmable=True,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        result_tokens, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(reporter.events[0]["cached_tokens"], 5)
        self.assertEqual(result_tokens.tolist(), [6])

    def test_update_cache_splits_prefill_at_the_checkpoint_boundary(self):
        chunk_size = 8
        session, model = self._make_session(
            cache_trimmable=False,
            chunk_size=chunk_size,
        )
        prompt = mx.array(list(range(1, 16)), dtype=mx.int32)

        result_tokens, _ = self._run_update_cache(session, prompt)

        checkpoint_prefix_len = len(prompt) - DEFAULT_CHECKPOINT_TAIL_TOKENS
        prefillable_tokens = len(prompt) - 1
        remaining_after_checkpoint = prefillable_tokens - checkpoint_prefix_len
        expected_calls = [
            checkpoint_prefix_len,
            chunk_size,
            remaining_after_checkpoint - chunk_size,
        ]

        self.assertEqual(model.calls, expected_calls)
        self.assertEqual(result_tokens.tolist(), prompt[-1:].tolist())

    def test_user_checkpoint_survives_assistant_snapshot_eviction_pressure(self):
        session, _ = self._make_session(
            cache_trimmable=False,
            history_capacity=2,
        )
        prompt = mx.array(list(range(1, 16)), dtype=mx.int32)

        # First request stores a reusable user checkpoint four tokens deep.
        self._run_update_cache(session, prompt)
        # These short follow-ups only flush assistant snapshots, creating eviction pressure.
        self._run_update_cache(session, mx.array([101, 102, 103], dtype=mx.int32))
        self._run_update_cache(session, mx.array([201, 202, 203], dtype=mx.int32))

        _, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(
            reporter.events[0]["cached_tokens"],
            len(prompt) - DEFAULT_CHECKPOINT_TAIL_TOKENS,
        )

    def test_setting_a_draft_model_resets_cached_history(self):
        session, _ = self._make_session(
            cache_trimmable=True,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3], dtype=mx.int32)

        self._run_update_cache(session, prompt)
        session.cache[0].advance(1)
        session.set_draft_model(FakeModel())

        _, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(reporter.events[0]["cached_tokens"], 0)

    def test_unsetting_draft_model_preserves_live_main_cache(self):
        session, _ = self._make_session(
            cache_trimmable=True,
            checkpoint_tail_tokens=100,
        )
        prompt = mx.array([1, 2, 3, 4], dtype=mx.int32)

        session.set_draft_model(FakeModel())
        self._run_update_cache(session, prompt)

        session.unset_draft_model()

        result_tokens, reporter = self._run_update_cache(session, prompt)

        self.assertEqual(reporter.events[0]["cached_tokens"], 3)
        self.assertEqual(len(session.cache), 1)
        self.assertEqual(result_tokens.tolist(), [4])

    def test_quantized_mode_reuses_full_snapshots_and_skips_same_prompt_checkpoint_reuse(
        self,
    ):
        model = FakeModel(cache_trimmable=False)
        session = CacheWrapper(
            model=model,
            max_kv_size=None,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
            chunk_size=8,
        )
        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)
        replacement = FakeCache(offset=0, trimmable=False)

        def quantize_cache(prompt_cache, **_):
            replacement.offset = prompt_cache[0].offset
            prompt_cache[0] = replacement

        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_cache,
        ):
            result_tokens, _ = self._run_update_cache(session, prompt)
        self.assertEqual(model.calls, [7])
        self.assertIs(session.cache[0], replacement)
        self.assertEqual(result_tokens.tolist(), [8])

        session.cache[0].advance(1)
        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_cache,
        ):
            _, reporter = self._run_update_cache(
                session,
                mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=mx.int32),
            )
        self.assertEqual(reporter.events[0]["cached_tokens"], 8)

        session, _ = self._make_session(
            cache_trimmable=False,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=0,
        )
        replacement = FakeCache(offset=0, trimmable=False)

        def quantize_same_prompt_cache(prompt_cache, **_):
            replacement.offset = prompt_cache[0].offset
            prompt_cache[0] = replacement

        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_same_prompt_cache,
        ):
            self._run_update_cache(session, prompt)
        session.cache[0].advance(1)

        with patch(
            "mlx_engine.cache_wrapper.maybe_quantize_kv_cache",
            side_effect=quantize_same_prompt_cache,
        ):
            _, reporter = self._run_update_cache(session, prompt)
        self.assertEqual(reporter.events[0]["cached_tokens"], 0)


class TestTrimCacheForSnapshot(unittest.TestCase):
    def _make_real_kv_entry(self, buffer_size: int = 256, offset: int = 10) -> object:
        """Create a real KVCache entry with known buffer and offset."""
        from mlx_lm.models.cache import KVCache as _KVCache

        entry = _KVCache()
        B, n_heads, dim = 1, 8, 64
        entry.keys = mx.zeros((B, n_heads, buffer_size, dim), dtype=mx.float32)
        entry.values = mx.zeros((B, n_heads, buffer_size, dim), dtype=mx.float32)
        entry.offset = offset
        return entry

    def test_snapshot_stores_trimmed_size_not_full_buffer(self):
        """Snapshot of a KVCache should store only the used portion, not the full buffer."""
        entry = self._make_real_kv_entry(buffer_size=256, offset=10)
        cache = [entry]

        snapshot = _trim_cache_for_snapshot(cache)

        self.assertEqual(len(snapshot), 1)
        trimmed_entry = snapshot[0]
        # The snapshot keys/values should only cover the offset range
        self.assertEqual(trimmed_entry.keys.shape[2], 10)
        self.assertEqual(trimmed_entry.values.shape[2], 10)
        # Full buffer was 256, snapshot is 10 — memory reduced ~25x
        self.assertLess(trimmed_entry.nbytes, entry.nbytes)

    def test_snapshot_tensors_are_independent_from_live_cache(self):
        """Modifying live cache must not affect stored snapshot (P1 fix)."""
        entry = self._make_real_kv_entry(buffer_size=256, offset=10)
        cache = [entry]

        # Write a known pattern into the used portion
        entry.keys[0, 0, :5, 0] = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        snapshot = _trim_cache_for_snapshot(cache)
        snapshot_keys = snapshot[0].keys

        # Mutate the live cache
        entry.keys[0, 0, :5, 0] = mx.array([99.0, 99.0, 99.0, 99.0, 99.0])

        # Snapshot should be unchanged — evaluate before asserting
        mx.eval(snapshot_keys)
        result = snapshot_keys[0, 0, :5, 0]
        mx.eval(result)
        self.assertTrue(
            bool(mx.array_equal(result, mx.array([1.0, 2.0, 3.0, 4.0, 5.0])))
        )

    def test_non_kv_cache_entries_are_deepcopied(self):
        """Non-KVCache entries should still use full deepcopy."""
        from unittest.mock import MagicMock

        fake = MagicMock()
        fake.offset = 5
        cache = [fake]

        snapshot = _trim_cache_for_snapshot(cache)

        self.assertIsNot(snapshot[0], fake)

    def test_zero_offset_produces_empty_tensors(self):
        """A cache with offset=0 should produce empty tensors in the snapshot."""
        entry = self._make_real_kv_entry(buffer_size=256, offset=0)
        cache = [entry]

        snapshot = _trim_cache_for_snapshot(cache)

        self.assertEqual(snapshot[0].keys.shape[2], 0)
        self.assertEqual(snapshot[0].values.shape[2], 0)

    def test_full_buffer_snapshot_does_not_trim(self):
        """When offset == buffer_size, no trimming should occur."""
        entry = self._make_real_kv_entry(buffer_size=256, offset=256)
        cache = [entry]

        snapshot = _trim_cache_for_snapshot(cache)

        self.assertEqual(snapshot[0].keys.shape[2], 256)
        self.assertEqual(snapshot[0].values.shape[2], 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
