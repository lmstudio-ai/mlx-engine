import unittest

from mlx_engine.utils.prompt_processing import emit_synthetic_progress


class TestProgressSyntheticUnbounded(unittest.TestCase):
    def test_ticks_are_monotonic_and_end_at_100(self):
        ticks = emit_synthetic_progress(total_tokens=8000, tick_count=5)
        self.assertGreaterEqual(len(ticks), 2)
        self.assertEqual(ticks[0], 0.0)
        self.assertAlmostEqual(ticks[-1], 100.0)
        self.assertTrue(all(t1 <= t2 for t1, t2 in zip(ticks, ticks[1:])))

    def test_tick_count_respected(self):
        tick_count = 7
        ticks = emit_synthetic_progress(total_tokens=16000, tick_count=tick_count)
        self.assertEqual(len(ticks), tick_count)


if __name__ == "__main__":
    unittest.main()
