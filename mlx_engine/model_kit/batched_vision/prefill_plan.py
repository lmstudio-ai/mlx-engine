"""Immutable per-request chunk schedule for memory-aware prompt prefill."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillSegment:
    """Use ``step_size`` until the cache reaches ``end_context_length``."""

    end_context_length: int
    step_size: int


@dataclass(frozen=True)
class PrefillPlan:
    """A precomputed segmented schedule for one prepared prompt."""

    prompt_context_length: int
    segments: tuple[PrefillSegment, ...]

    def segment_and_start_for_context(
        self,
        context_length: int,
    ) -> tuple[int, PrefillSegment]:
        if not self.segments:
            raise ValueError("Prefill plan has no segments")
        segment_start = 0
        for segment in self.segments:
            if context_length < segment.end_context_length:
                return segment_start, segment
            segment_start = segment.end_context_length
        last_segment = self.segments[-1]
        previous_end = (
            self.segments[-2].end_context_length if len(self.segments) > 1 else 0
        )
        return previous_end, last_segment
