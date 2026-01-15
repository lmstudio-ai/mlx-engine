import unittest
from mlx_engine.utils.prompt_progress_events import (
    PromptProgressCallbackReporter,
    PromptProgressBeginEvent,
    PromptProgressEvent,
)


class TestPromptProgressCallbackReporter(unittest.TestCase):
    def setUp(self):
        self.events: list = []
        self.percents: list = []

    def progress_callback(self, event, is_draft: bool) -> bool:
        self.events.append({"event": event, "is_draft": is_draft})
        return True

    def percent_callback(self, percent: float) -> None:
        self.percents.append(percent)

    def test_begin_emits_event(self):
        reporter = PromptProgressCallbackReporter(self.progress_callback)
        reporter.begin(
            is_draft=False,
            cached_tokens=50,
            total_prompt_tokens=100,
            prefill_tokens_processed=10,
        )

        self.assertEqual(len(self.events), 1)
        event = self.events[0]["event"]
        self.assertIsInstance(event, PromptProgressBeginEvent)
        self.assertEqual(event.cached_tokens, 50)
        self.assertEqual(event.total_prompt_tokens, 100)
        self.assertEqual(event.prefill_tokens_processed, 10)
        self.assertFalse(self.events[0]["is_draft"])

    def test_update_emits_event(self):
        reporter = PromptProgressCallbackReporter(self.progress_callback)
        reporter.update(is_draft=False, prefill_tokens_processed=25)

        self.assertEqual(len(self.events), 1)
        event = self.events[0]["event"]
        self.assertIsInstance(event, PromptProgressEvent)
        self.assertEqual(event.prefill_tokens_processed, 25)
        self.assertFalse(event.is_final)

    def test_finish_emits_event_with_is_final(self):
        reporter = PromptProgressCallbackReporter(self.progress_callback)
        reporter.finish(is_draft=False, prefill_tokens_processed=50)

        self.assertEqual(len(self.events), 1)
        event = self.events[0]["event"]
        self.assertIsInstance(event, PromptProgressEvent)
        self.assertEqual(event.prefill_tokens_processed, 50)
        self.assertTrue(event.is_final)

    def test_finish_without_prefill_tokens_uses_last_value(self):
        reporter = PromptProgressCallbackReporter(self.progress_callback)
        reporter.begin(
            is_draft=False,
            cached_tokens=20,
            total_prompt_tokens=100,
            prefill_tokens_processed=10,
        )
        reporter.update(is_draft=False, prefill_tokens_processed=50)
        reporter.finish(is_draft=False, prefill_tokens_processed=None)

        event = self.events[2]["event"]
        self.assertEqual(event.prefill_tokens_processed, 50)
        self.assertTrue(event.is_final)

    def test_finish_without_prefill_tokens_no_context_uses_zero(self):
        reporter = PromptProgressCallbackReporter(self.progress_callback)
        reporter.finish(is_draft=False, prefill_tokens_processed=None)

        event = self.events[0]["event"]
        self.assertEqual(event.prefill_tokens_processed, 0)
        self.assertTrue(event.is_final)

    def test_percent_callback_emitted_on_begin(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=50,
            total_prompt_tokens=100,
            prefill_tokens_processed=10,
        )

        self.assertEqual(len(self.percents), 1)
        # percent = prefill / (total - cached) * 100 = 10 / 50 * 100 = 20
        self.assertEqual(self.percents[0], 20.0)

    def test_percent_callback_emitted_on_update(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=20,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        self.percents.clear()

        reporter.update(is_draft=False, prefill_tokens_processed=40)

        self.assertEqual(len(self.percents), 1)
        # percent = prefill / (total - cached) * 100 = 40 / 80 * 100 = 50
        self.assertEqual(self.percents[0], 50.0)

    def test_percent_callback_emitted_on_finish(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=20,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        self.percents.clear()

        reporter.finish(is_draft=False, prefill_tokens_processed=80)

        self.assertEqual(len(self.percents), 1)
        # percent = prefill / (total - cached) * 100 = 80 / 80 * 100 = 100
        self.assertEqual(self.percents[0], 100.0)

    def test_percent_emitted_on_finish_using_last_value(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=20,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        reporter.update(is_draft=False, prefill_tokens_processed=60)
        self.percents.clear()

        reporter.finish(is_draft=False, prefill_tokens_processed=None)

        self.assertEqual(len(self.percents), 1)
        # percent = prefill / (total - cached) * 100 = 60 / 80 * 100 = 75
        self.assertEqual(self.percents[0], 75.0)

    def test_draft_events_do_not_emit_percent(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )

        reporter.begin(
            is_draft=True,
            cached_tokens=50,
            total_prompt_tokens=100,
            prefill_tokens_processed=10,
        )
        reporter.update(is_draft=True, prefill_tokens_processed=25)
        reporter.finish(is_draft=True, prefill_tokens_processed=50)

        self.assertEqual(len(self.events), 3)
        self.assertEqual(len(self.percents), 0)

    def test_draft_events_still_emit_to_progress_callback(self):
        reporter = PromptProgressCallbackReporter(self.progress_callback)

        reporter.begin(
            is_draft=True,
            cached_tokens=50,
            total_prompt_tokens=100,
            prefill_tokens_processed=10,
        )

        self.assertEqual(len(self.events), 1)
        self.assertTrue(self.events[0]["is_draft"])

    def test_update_without_begin_skips_percent(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )

        reporter.update(is_draft=False, prefill_tokens_processed=25)

        self.assertEqual(len(self.events), 1)
        self.assertEqual(len(self.percents), 0)

    def test_finish_without_begin_skips_percent(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )

        reporter.finish(is_draft=False, prefill_tokens_processed=50)

        self.assertEqual(len(self.events), 1)
        self.assertEqual(len(self.percents), 0)

    def test_percent_clamped_to_100(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=80,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        self.percents.clear()

        # tokens_to_prefill = 20, so 50/20 = 250% without clamping
        reporter.update(is_draft=False, prefill_tokens_processed=50)

        self.assertEqual(self.percents[0], 100.0)

    def test_percent_clamped_to_0(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=-10,  # edge case
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        self.percents.clear()

        reporter.update(is_draft=False, prefill_tokens_processed=-20)

        self.assertEqual(self.percents[0], 0.0)

    def test_full_sequence(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )

        reporter.begin(
            is_draft=False,
            cached_tokens=20,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )
        reporter.update(is_draft=False, prefill_tokens_processed=40)
        reporter.update(is_draft=False, prefill_tokens_processed=80)
        reporter.finish(is_draft=False, prefill_tokens_processed=None)

        self.assertEqual(len(self.events), 4)

        # Event 0: Begin event
        begin_event = self.events[0]["event"]
        self.assertIsInstance(begin_event, PromptProgressBeginEvent)
        self.assertEqual(begin_event.cached_tokens, 20)
        self.assertEqual(begin_event.total_prompt_tokens, 100)
        self.assertEqual(begin_event.prefill_tokens_processed, 0)
        self.assertFalse(self.events[0]["is_draft"])

        # Event 1: First update event
        update1_event = self.events[1]["event"]
        self.assertIsInstance(update1_event, PromptProgressEvent)
        self.assertEqual(update1_event.prefill_tokens_processed, 40)
        self.assertFalse(update1_event.is_final)
        self.assertFalse(self.events[1]["is_draft"])

        # Event 2: Second update event
        update2_event = self.events[2]["event"]
        self.assertIsInstance(update2_event, PromptProgressEvent)
        self.assertEqual(update2_event.prefill_tokens_processed, 80)
        self.assertFalse(update2_event.is_final)
        self.assertFalse(self.events[2]["is_draft"])

        # Event 3: Finish event (uses last value of 80)
        finish_event = self.events[3]["event"]
        self.assertIsInstance(finish_event, PromptProgressEvent)
        self.assertEqual(finish_event.prefill_tokens_processed, 80)
        self.assertTrue(finish_event.is_final)
        self.assertFalse(self.events[3]["is_draft"])

        # tokens_to_prefill = 80, so: 0/80=0%, 40/80=50%, 80/80=100%, 80/80=100%
        self.assertEqual(self.percents, [0.0, 50.0, 100.0, 100.0])

    def test_cancellation_propagates_from_progress_callback(self):
        for draft_mode in [False, True]:
            with self.subTest(is_draft=draft_mode):
                self.events.clear()

                def cancelling_callback(event, is_draft: bool) -> bool:
                    self.events.append({"event": event, "is_draft": is_draft})
                    return len(self.events) < 2

                reporter = PromptProgressCallbackReporter(cancelling_callback)

                result1 = reporter.begin(
                    is_draft=draft_mode,
                    cached_tokens=20,
                    total_prompt_tokens=100,
                    prefill_tokens_processed=0,
                )
                result2 = reporter.update(
                    is_draft=draft_mode, prefill_tokens_processed=40
                )

                self.assertTrue(result1)
                self.assertFalse(result2)

    def test_zero_tokens_to_prefill_returns_100_percent(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=0,
            total_prompt_tokens=0,
            prefill_tokens_processed=0,
        )

        # tokens_to_prefill = 0, so returns 100% (nothing to do)
        self.assertEqual(len(self.percents), 1)
        self.assertEqual(self.percents[0], 100.0)

    def test_all_cached_returns_100_percent(self):
        reporter = PromptProgressCallbackReporter(
            self.progress_callback, percent_callback=self.percent_callback
        )
        reporter.begin(
            is_draft=False,
            cached_tokens=100,
            total_prompt_tokens=100,
            prefill_tokens_processed=0,
        )

        # tokens_to_prefill = 0, so returns 100% (everything cached)
        self.assertEqual(len(self.percents), 1)
        self.assertEqual(self.percents[0], 100.0)
