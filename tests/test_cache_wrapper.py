import unittest
from unittest.mock import patch
import mlx.core as mx
from mlx_engine.cache_wrapper import CacheWrapper, StopPromptProcessing
from tests.shared import model_getter, RecordingReporter, CancellingReporter
from mlx_engine.generate import load_model, tokenize


class TestCacheWrapper(unittest.TestCase):
    def test_find_common_prefix_with_mismatch(self):
        """Test when there's a mismatch in the tokens"""
        # Create two arrays with a known common prefix [1, 2, 3]
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 6, 7])  # Mismatch at index 3
        num_tokens_to_exclude = 1

        print("\nTest with mismatch:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_common_prefix(
            current_tokens, prompt_tokens, num_tokens_to_exclude
        )
        self.assertEqual(result, 3)  # Should find 3 matching tokens

    def test_find_common_prefix_all_match(self):
        """Test when all tokens match"""
        # Create two identical arrays
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])  # All tokens match
        num_tokens_to_exclude = 1

        print("\nTest with all matching:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_common_prefix(
            current_tokens, prompt_tokens, num_tokens_to_exclude
        )
        self.assertEqual(
            result, 4
        )  # Should find 4 matching tokens (5-1 due to num_tokens_to_exclude)

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
        tokens_to_process = len(prompt_tokens) - num_tokens_to_exclude
        # ceiling division +1 for finish
        expected_chunks = (tokens_to_process + chunk_size - 1) // chunk_size + 1

        # First attempt: Reporter that cancels after 3 events
        cancelling_reporter = CancellingReporter(cancel_after=3)

        with self.assertRaises(StopPromptProcessing):
            model_kit.cache_wrapper.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=cancelling_reporter,
                num_tokens_to_exclude=1,
            )
        first_attempt_event_count = len(cancelling_reporter.events)

        # Second attempt: Reporter that doesn't cancel
        recording_reporter = RecordingReporter()

        result_tokens = model_kit.cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            reporter=recording_reporter,
            num_tokens_to_exclude=1,
        )
        second_attempt_event_count = len(recording_reporter.events)

        self.assertEqual(
            second_attempt_event_count,
            # +1 for finish, +1 for the begin event on retry
            expected_chunks - first_attempt_event_count + 2,
        )

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)

    def test_needs_checkpointing_detection_standard_model(self):
        """Test that standard transformer models have _needs_checkpointing=False"""
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        cache_wrapper = CacheWrapper(
            model_kit.model,
            max_kv_size=4096,
        )

        # Standard transformer models should NOT need checkpointing
        self.assertFalse(cache_wrapper._needs_checkpointing)
        self.assertIsNone(cache_wrapper._checkpoint_store)

    def test_needs_checkpointing_detection_non_trimmable(self):
        """Test that non-trimmable cache triggers _needs_checkpointing=True"""
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        # Patch can_trim_prompt_cache to return False, simulating a hybrid model
        with patch("mlx_engine.cache_wrapper.can_trim_prompt_cache", return_value=False):
            cache_wrapper = CacheWrapper(
                model_kit.model,
                max_kv_size=4096,
            )

        self.assertTrue(cache_wrapper._needs_checkpointing)
        self.assertIsNotNone(cache_wrapper._checkpoint_store)

    def test_checkpoint_cleared_on_draft_model_change(self):
        """Test that checkpoints are cleared when draft model changes"""
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        # Create wrapper with forced checkpointing
        with patch("mlx_engine.cache_wrapper.can_trim_prompt_cache", return_value=False):
            cache_wrapper = CacheWrapper(
                model_kit.model,
                max_kv_size=4096,
            )

        # Manually add a checkpoint
        tokens = mx.array([1, 2, 3])
        cache_wrapper._checkpoint_store.save(
            tokens,
            cache_wrapper.cache[: len(model_kit.model.layers)],
        )
        self.assertEqual(len(cache_wrapper._checkpoint_store), 1)

        # Setting a draft model should clear checkpoints
        cache_wrapper.set_draft_model(model_kit.model)
        self.assertEqual(len(cache_wrapper._checkpoint_store), 0)

        # Unsetting the draft model should also clear checkpoints
        # First add a checkpoint again
        cache_wrapper._checkpoint_store.save(
            tokens,
            cache_wrapper.cache[: len(model_kit.model.layers)],
        )
        self.assertEqual(len(cache_wrapper._checkpoint_store), 1)

        cache_wrapper.unset_draft_model()
        self.assertEqual(len(cache_wrapper._checkpoint_store), 0)


class TestCacheWrapperCheckpointing(unittest.TestCase):
    """Tests for checkpoint integration that don't require model downloads.
    Uses mocks to simulate non-trimmable caches."""

    def test_needs_checkpointing_false_for_trimmable(self):
        """can_trim_prompt_cache returning True means no checkpoint store."""
        with patch("mlx_engine.cache_wrapper.make_prompt_cache") as mock_make, \
             patch("mlx_engine.cache_wrapper.can_trim_prompt_cache", return_value=True):
            mock_make.return_value = []
            mock_model = unittest.mock.MagicMock()
            mock_model.layers = []

            cw = CacheWrapper(mock_model, max_kv_size=None)
            self.assertFalse(cw._needs_checkpointing)
            self.assertIsNone(cw._checkpoint_store)

    def test_needs_checkpointing_true_for_non_trimmable(self):
        """can_trim_prompt_cache returning False creates a checkpoint store."""
        with patch("mlx_engine.cache_wrapper.make_prompt_cache") as mock_make, \
             patch("mlx_engine.cache_wrapper.can_trim_prompt_cache", return_value=False):
            mock_make.return_value = []
            mock_model = unittest.mock.MagicMock()
            mock_model.layers = []

            cw = CacheWrapper(mock_model, max_kv_size=None)
            self.assertTrue(cw._needs_checkpointing)
            self.assertIsNotNone(cw._checkpoint_store)
            self.assertEqual(len(cw._checkpoint_store), 0)

    def test_checkpoint_store_cleared_on_set_draft_model(self):
        """Setting draft model clears checkpoint store."""
        with patch("mlx_engine.cache_wrapper.make_prompt_cache") as mock_make, \
             patch("mlx_engine.cache_wrapper.can_trim_prompt_cache", return_value=False):
            mock_make.return_value = []
            mock_model = unittest.mock.MagicMock()
            mock_model.layers = []

            cw = CacheWrapper(mock_model, max_kv_size=None)

            # Simulate a saved checkpoint
            from tests.test_recurrent_checkpoint_store import make_mock_cache
            cw._checkpoint_store.save(mx.array([1, 2, 3]), make_mock_cache())
            self.assertEqual(len(cw._checkpoint_store), 1)

            # set_draft_model should clear
            mock_draft = unittest.mock.MagicMock()
            cw.set_draft_model(mock_draft)
            self.assertEqual(len(cw._checkpoint_store), 0)

    def test_checkpoint_store_cleared_on_unset_draft_model(self):
        """Unsetting draft model clears checkpoint store."""
        with patch("mlx_engine.cache_wrapper.make_prompt_cache") as mock_make, \
             patch("mlx_engine.cache_wrapper.can_trim_prompt_cache", return_value=False):
            mock_make.return_value = []
            mock_model = unittest.mock.MagicMock()
            mock_model.layers = []

            cw = CacheWrapper(mock_model, max_kv_size=None)
            cw.draft_model = unittest.mock.MagicMock()  # pretend we have a draft model

            from tests.test_recurrent_checkpoint_store import make_mock_cache
            cw._checkpoint_store.save(mx.array([1, 2, 3]), make_mock_cache())
            self.assertEqual(len(cw._checkpoint_store), 1)

            cw.unset_draft_model()
            self.assertEqual(len(cw._checkpoint_store), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
